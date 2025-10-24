from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import torch

from src.voice_ultils import get_embedding, cosine_score
from src.database import Database
from src.voice_model import ECAPA_TDNN
from src.ultils_logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Globals will be injected from main
db: Database
model: ECAPA_TDNN
device: str
THRESHOLD: float

def init_voice_router(database: Database, mdl: ECAPA_TDNN, dev: str, threshold: float):
    global db, model, device, THRESHOLD
    db = database
    model = mdl
    device = dev
    THRESHOLD = threshold


@router.post("/enroll/{username}")
async def enroll(username: str, password: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"Enroll request received for user: {username}")
    emb = get_embedding(model, file, device)
    try:
        db.add_user(username, password, emb)
        logger.info(f"User enrolled: {username}")
    except ValueError:
        logger.warning(f"Enroll failed - username exists: {username}")
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"status": "enrolled", "username": username}


@router.post("/verify/{username}")
async def verify_user(username: str, password: str = Form(None), file: UploadFile = File(None)):
    """
    Verify a user's identity using either password or voice embedding.
    """
    logger.info(f"[Verify] Request for user: {username}")

    user = db.get_user(username)
    if not user:
        logger.warning(f"Verification failed - user not found: {username}")
        raise HTTPException(status_code=404, detail="User not enrolled")

    # Password verification
    if password is not None:
        logger.info(f"[Password Verify] Attempt for user: {username}")

        if db.verify_password(username, password):
            logger.info(f"Password verification succeeded for user: {username}")
            return {"username": username, "method": "password", "result": "accepted"}

        logger.warning(f"Password verification failed - incorrect password for user: {username}")
        raise HTTPException(status_code=401, detail="Invalid password")

    # Voice verification
    if file is not None:
        logger.info(f"[Voice Verify] Attempt for user: {username}")

        try:
            emb_new = get_embedding(model, file, device)
            emb_ref = db.get_embedding(username, device)
        except Exception as e:
            logger.error(f"Voice verification error for {username}: {e}")
            raise HTTPException(status_code=400, detail="Failed to process voice file")

        score = cosine_score(emb_new, emb_ref)
        result = "accepted" if score > THRESHOLD else "rejected"

        logger.info(f"Voice verification for {username}: score={score:.4f}, result={result}")
        return {"username": username, "method": "voice", "score": score, "result": result}

    logger.warning(f"Verification failed - no input provided for user: {username}")
    raise HTTPException(status_code=400, detail="No verification data provided")


@router.post("/verify/password/{username}")
async def verify_password(username: str, password: str = Form(...)):
    """
    Verify a user's identity using password.
    """
    logger.info(f"[Password Verify] Request for user: {username}")

    user = db.get_user(username)
    if not user:
        logger.warning(f"Password verification failed - user not found: {username}")
        raise HTTPException(status_code=404, detail="User not enrolled")

    # Check password
    if db.verify_password(username, password):
        logger.info(f"Password verification succeeded for user: {username}")
        return {"username": username, "method": "password", "result": "accepted"}

    logger.warning(f"Password verification failed - incorrect password for user: {username}")
    raise HTTPException(status_code=401, detail="Invalid password")

@router.post("/verify/voice/{username}")
async def verify_voice(username: str, file: UploadFile = File(...)):
    """
    Verify a user's identity using voice embedding.
    """
    logger.info(f"[Voice Verify] Request for user: {username}")

    user = db.get_user(username)
    if not user:
        logger.warning(f"Voice verification failed - user not found: {username}")
        raise HTTPException(status_code=404, detail="User not enrolled")

    try:
        emb_new = get_embedding(model, file, device)
        emb_ref = db.get_embedding(username, device)
    except Exception as e:
        logger.error(f"Voice verification error for {username}: {e}")
        raise HTTPException(status_code=400, detail="Failed to process voice file")

    score = cosine_score(emb_new, emb_ref)
    result = "accepted" if score > THRESHOLD else "rejected"

    logger.info(f"Voice verification for {username}: score={score:.4f}, result={result}")
    return {"username": username, "method": "voice", "score": score, "result": result}

@router.get("/users")
async def list_users():
    logger.debug("Listing all users")
    users = list(db.data.keys())
    return {"count": len(users), "users": users}
