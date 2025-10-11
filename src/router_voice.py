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
async def verify(username: str, password: str = Form(""), file: UploadFile = File(None)):
    logger.info(f"Verification request for user: {username}")

    user = db.get_user(username)
    if not user:
        logger.warning(f"Verification failed - user not found: {username}")
        raise HTTPException(status_code=404, detail="User not enrolled")

    # Password check
    if password and password.strip():
        if db.verify_password(username, password):
            logger.info(f"Password verification succeeded for user: {username}")
            return {"username": username, "method": "password", "result": "accepted"}
        else:
            logger.warning(f"Verification failed - incorrect password for user: {username}")
            raise HTTPException(status_code=401, detail="Invalid password")

    # Voice check
    if file is not None:
        emb_new = get_embedding(model, file, device)
        emb_ref = db.get_embedding(username, device)
        score = cosine_score(emb_new, emb_ref)
        result = "accepted" if score > THRESHOLD else "rejected"
        logger.info(f"Voice verification for {username}: score={score:.4f}, result={result}")
        return {"username": username, "method": "voice", "score": score, "result": result}

    logger.warning(f"Verification failed - neither password nor voice file provided for user: {username}")
    raise HTTPException(status_code=400, detail="Must provide password or voice")


@router.get("/users")
async def list_users():
    logger.debug("Listing all users")
    users = list(db.data.keys())
    return {"count": len(users), "users": users}
