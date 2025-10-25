from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Callable

from src.voice_ultils import get_embedding, cosine_score
from src.database import Database
from src.voice_model import ECAPA_TDNN
from src.ultils_logger import get_logger
from src.load_assist import infer_assist

logger = get_logger(__name__)
router = APIRouter()

db: Database
model: ECAPA_TDNN
assist_model: Callable
device: str
THRESHOLD: float


def init_voice_router(database: Database, mdl: ECAPA_TDNN, assist_mdl: Callable, dev: str, threshold: float):
    global db, model, assist_model, device, THRESHOLD
    db = database
    model = mdl
    assist_model = assist_mdl
    device = dev
    THRESHOLD = threshold


@router.post("/enroll/{username}")
async def enroll(username: str, password: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"Enroll request received for user: {username}")
    try:
        emb = get_embedding(model, file, device)
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process voice file")

    if db.get_user(username, strict=False):
        logger.warning(f"Enroll failed - username exists: {username}")
        raise HTTPException(status_code=409, detail="Username already exists")
    
    db.add_user(username, password, emb)
    logger.info(f"User enrolled: {username}")
    return {"status": "success", "username": username}


@router.post("/verify/{username}")
async def verify_user(username: str, password: str = Form(None), file: UploadFile = File(None)):
    logger.info(f"[Verify] Request for user: {username}")

    if not file or not password:
        raise HTTPException(status_code=400, detail="Password and voice file are required")

    user = db.get_user(username, strict=False)
    if not user:
        raise HTTPException(status_code=404, detail="User not enrolled")

    try:
        status = infer_assist(assist_model, file, device)
    except Exception as e:
        logger.error(f"Assist model failed: {e}")
        raise HTTPException(status_code=500, detail="Assist model internal error")

    if status != "bonafide":
        raise HTTPException(status_code=403, detail=f"Spoofed or synthetic voice detected ({status})")

    if not db.verify_password(username, password):
        raise HTTPException(status_code=401, detail="Invalid password")

    try:
        emb_new = get_embedding(model, file, device)
        emb_ref = db.get_embedding(username, device)
        score = cosine_score(emb_new, emb_ref)
    except Exception as e:
        logger.error(f"Voice verification error for {username}: {e}")
        raise HTTPException(status_code=500, detail="Voice processing failed")

    if score <= THRESHOLD:
        raise HTTPException(status_code=403, detail=f"Voice verification failed (score={score:.4f})")

    logger.info(f"Verification for {username}: score={score:.4f}, result=accepted")
    return {
        "status": "success",
        "username": username,
        "method": "password+voice",
        "score": score,
        "assist": status,
    }


@router.post("/verify/password/{username}")
async def verify_password(username: str, password: str = Form(...)):
    logger.info(f"[Password Verify] Request for user: {username}")

    user = db.get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not enrolled")

    if not db.verify_password(username, password):
        raise HTTPException(status_code=401, detail="Invalid password")

    return {"status": "success", "username": username, "method": "password"}


@router.post("/verify/voice/{username}")
async def verify_voice(username: str, file: UploadFile = File(...)):
    logger.info(f"[Voice Verify] Request for user: {username}")

    user = db.get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not enrolled")

    try:
        status = infer_assist(assist_model, file, device)
    except Exception as e:
        logger.error(f"Assist model inference failed: {e}")
        raise HTTPException(status_code=500, detail="Assist model internal error")

    if status != "bonafide":
        raise HTTPException(status_code=403, detail=f"Spoofed or synthetic voice detected ({status})")

    try:
        emb_new = get_embedding(model, file, device)
        emb_ref = db.get_embedding(username, device)
        score = cosine_score(emb_new, emb_ref)
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(status_code=500, detail="Voice processing failed")

    if score <= THRESHOLD:
        raise HTTPException(status_code=403, detail=f"Voice verification failed (score={score:.4f})")

    return {
        "status": "success",
        "username": username,
        "method": "voice",
        "score": score,
        "assist": status,
    }


@router.post("/spoofcheck")
async def spoof_check(file: UploadFile = File(...)):
    logger.info(f"[SpoofCheck] File received: {file.filename if file else 'None'}")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        result = infer_assist(assist_model, file, device)
        return {
            "status": "success",
            "filename": file.filename,
            "result": result,
            "description": "bonafide = real, spoofed = synthetic or attack",
        }
    except Exception as e:
        logger.error(f"SpoofCheck error: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze voice file")


@router.get("/users")
async def list_users():
    users = list(db.data.keys())
    return {"status": "success", "count": len(users), "users": users}
