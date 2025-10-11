from fastapi import APIRouter, HTTPException
import requests
import datetime

from src.database import Database
from src.ultils_logger import get_logger

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


@router.get("/session/{username}")
async def list_chat_sessions(username: str):
    """List all chat session IDs for a user"""
    if db is None:
        raise HTTPException(status_code=500, detail="Chat router not initialized")

    try:
        sessions = db.list_sessions(username)
        return {"username": username, "session_ids": sessions}
    
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found")

@router.get("/messages/{username}/{session_id}")
async def get_messages(username: str, session_id: str):
    """Get the full message history for a session"""
    if db is None:
        raise HTTPException(status_code=500, detail="Chat router not initialized")

    try:
        messages = db.get_session_messages(username, session_id)
        return {"username": username, "session_id": session_id, "messages": messages}
    
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/send/{username}/{session_id}")
async def send_message(username: str, session_id: str, user_message: str):
    """
    Send a user message to n8n → get LLM response → store both messages
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