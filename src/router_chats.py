import requests
import datetime

from src.database import Database
from src.ultils_logger import get_logger

from fastapi import APIRouter, HTTPException, UploadFile

logger = get_logger(__name__)
router = APIRouter()

# Globals injected from main
db: Database | None = None
N8N_WEBHOOK_URL: str = ""

def init_chat_router(database: Database, n8n_url: str):
    global db, N8N_WEBHOOK_URL
    db = database
    N8N_WEBHOOK_URL = n8n_url
    logger.info(f"Chat router initialized with n8n URL: {N8N_WEBHOOK_URL}")


### Helpers ###
def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


### Routes ###

@router.post("/session/{username}")
async def create_chat_session(username: str, session_name:str):
    """Create a new chat session for a user"""
    if db is None:
        raise HTTPException(status_code=500, detail="Chat router not initialized")

    try:
        session_id = db.create_session(username, session_name)
        logger.info(f"New chat session created for {username}: {session_id}; Named {session_name}")
        return {"session_id": session_id}
    
    except ValueError as e:
        logger.warning(f"Failed to create session: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sessions/{username}")
async def get_all_chat_sessionids(username: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Chat router not initialized")
    try:
        return db.list_sessions(username)
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found")
    

@router.get("/session/{username}/{session_id}")
async def get_chat_session(username: str, session_id: str):
    """Get a specific chat session for a user"""
    if db is None:
        raise HTTPException(status_code=500, detail="Chat router not initialized")

    try:
        return db.get_session(username, session_id)
    
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/send/{username}/{session_id}")
async def send_message(username: str, session_id: str, user_message: str):
    """
    Send a user message to n8n -> get LLM response -> store both messages
    """
    if db is None or not N8N_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="Chat router not initialized")

    # Add user message to DB
    db.add_message(username, session_id, {
        "time": _now(),
        "role": "human",
        "message": user_message
    })

    # Send to n8n webhook
    try:
        resp = requests.post(
            N8N_WEBHOOK_URL,
            json={"username": username, "message": user_message, "session_id": session_id},
            timeout=30
        )
        resp.raise_for_status()

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to contact n8n: {e}")
        raise HTTPException(status_code=502, detail="Failed to contact n8n")

    # Parse n8n response
    try:
        n8n_data = resp.json()
    except ValueError:
        n8n_data = {}

    bot_reply = n8n_data.get("reply", "(No response from bot)")

    # Save bot reply
    db.add_message(username, session_id, {
        "time": _now(),
        "role": "bot",
        "message": bot_reply
    })

    logger.info(f"Chat [{session_id}] {username}: {user_message} → {bot_reply}")
    return {"reply": bot_reply}

@router.post("/send-voice/{username}/{session_id}")
async def send_voice(username:str, session_id:str, user_voice:UploadFile):
    if db is None or not N8N_WEBHOOK_URL:
        raise HTTPException(status_code=500, detail="Chat router not initialized")

    try:
        # Read the file bytes to forward to n8n
        file_bytes = await user_voice.read()
    except Exception as e:
        logger.error(f"Failed to read voice file: {e}")
        raise HTTPException(status_code=400, detail="Invalid file upload")

    # Send voice file to n8n as multipart/form-data
    try:
        files = {"file": (user_voice.filename, file_bytes, user_voice.content_type)}
        data = {"username": username, "session_id": session_id}

        resp = requests.post(N8N_WEBHOOK_URL, data=data, files=files, timeout=60) # type: ignore
        resp.raise_for_status()

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to contact n8n for voice message: {e}")
        raise HTTPException(status_code=502, detail="Failed to contact n8n")

    # Parse n8n response
    try:
        n8n_data = resp.json()
    except ValueError:
        n8n_data = {}

    # n8n should return both transcription and reply
    logger.info(n8n_data)
    transcription = n8n_data.get("transcript", "(no transcription)")
    bot_reply = n8n_data.get("reply", "(no reply)")

    # Save transcribed message as human
    db.add_message(username, session_id, {
        "time": _now(),
        "role": "human",
        "message": "[TRANSCRIPTION] " + transcription
    })

    # Save bot reply
    db.add_message(username, session_id, {
        "time": _now(),
        "role": "bot",
        "message": bot_reply
    })

    logger.info(f"[VoiceChat] [{session_id}] [{username}] Transcription: {transcription}; Reply: {bot_reply}")
    return {"transcript": transcription, "reply": bot_reply}

@router.delete("/session/{username}/{session_id}")
async def delete_chat_session(username: str, session_id: str):
    """Delete a specific chat session for a user"""
    if db is None:
        raise HTTPException(status_code=500, detail="Chat router not initialized")

    try:
        db.delete_session(username, session_id)
        return {
            "message": f"Session {session_id} deleted for user {username}"
        }
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")
