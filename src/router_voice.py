from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Callable

from src.voice_ultils import get_embedding, cosine_score
from src.database import Database
from src.voice_model import ECAPA_TDNN
from src.ultils_logger import get_logger
from src.load_assist import infer_assist

logger = get_logger(__name__)

router = APIRouter()

# Globals will be injected from main
db: Database
model: ECAPA_TDNN
assist_model: Callable
device: str
THRESHOLD: float

def init_voice_router(database: Database, mdl: ECAPA_TDNN, assist_mdl:Callable, dev: str, threshold: float):
    global db, model, assist_model, device, THRESHOLD
    db = database
    model = mdl
    assist_model = assist_mdl
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
    Verify a user's identity using BOTH password and voice embedding.
    Both must pass verification.
    """
    logger.info(f"[Verify] Request for user: {username}")

    status = infer_assist(assist_model, file, device)
    if status != "bonafide":
        logger.warning(f"Verification rejected for {username}: spoofed voice detected")
        raise HTTPException(status_code=403, detail="Spoofed or synthetic voice detected")

    user = db.get_user(username)
    if not user:
        logger.warning(f"Verification failed - user not found: {username}")
        raise HTTPException(status_code=404, detail="User not enrolled")

    # Check both fields provided
    if password is None or file is None:
        logger.warning(f"Verification failed - missing password or voice for user: {username}")
        raise HTTPException(status_code=400, detail="Password and voice file are required")

    # Password verification
    logger.info(f"[Password Verify] Attempt for user: {username}")
    if not db.verify_password(username, password):
        logger.warning(f"Password verification failed for user: {username}")
        raise HTTPException(status_code=401, detail="Invalid password")

    # Voice verification
    logger.info(f"[Voice Verify] Attempt for user: {username}")
    try:
        emb_new = get_embedding(model, file, device)
        emb_ref = db.get_embedding(username, device)
    except Exception as e:
        logger.error(f"Voice verification error for {username}: {e}")
        raise HTTPException(status_code=400, detail="Failed to process voice file")

    score = cosine_score(emb_new, emb_ref)
    if score <= THRESHOLD:
        logger.warning(f"Voice verification failed for {username}: score={score:.4f}")
        raise HTTPException(status_code=401, detail="Voice verification failed")

    logger.info(f"Verification succeeded for {username}: score={score:.4f}")
    return {"username": username, "method": "password+voice", "score": score, "result": "accepted"}



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
    Verify a user's identity using voice embedding with anti-spoofing check.
    """
    logger.info(f"[Voice Verify] Request for user: {username}")

    user = db.get_user(username)
    if not user:
        logger.warning(f"Voice verification failed - user not found: {username}")
        raise HTTPException(status_code=404, detail="User not enrolled")

    # 🛡️ Anti-spoofing check
    try:
        status = infer_assist(assist_model, file, device)
        logger.info(f"[Assist Model] Liveness result for {username}: {status}")
        if status != "bonafide":
            raise HTTPException(status_code=403, detail="Spoofed or synthetic voice detected")
    except Exception as e:
        logger.error(f"Assist model inference failed for {username}: {e}")
        raise HTTPException(status_code=400, detail="Assist model failed to process voice")

    # 🎙️ Normal embedding verification
    try:
        emb_new = get_embedding(model, file, device)
        emb_ref = db.get_embedding(username, device)
    except Exception as e:
        logger.error(f"Voice verification error for {username}: {e}")
        raise HTTPException(status_code=400, detail="Failed to process voice file")

    score = cosine_score(emb_new, emb_ref)
    result = "accepted" if score > THRESHOLD else "rejected"

    logger.info(f"Voice verification for {username}: score={score:.4f}, result={result}")
    return {
        "username": username,
        "method": "voice",
        "score": score,
        "assist": status,
        "result": result
    }

@router.post("/spoofcheck")
async def spoof_check(file: UploadFile = File(...)):
    """
    Check if a given voice sample is bonafide (real) or spoofed (synthetic/attack).
    Uses the assist_model only.
    """
    logger.info(f"[SpoofCheck] Received file: {file.filename}")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        result = infer_assist(assist_model, file, device)
        logger.info(f"[SpoofCheck] Result for {file.filename}: {result}")
        return {
            "filename": file.filename,
            "result": result,
            "description": "bonafide = real speaker, spoofed = synthetic or attack sample",
        }
    except Exception as e:
        logger.error(f"[SpoofCheck] Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze voice file")

@router.get("/users")
async def list_users():
    logger.debug("Listing all users")
    users = list(db.data.keys())
    return {"count": len(users), "users": users}
