from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form

from src.model import ECAPA_TDNN
from src.ultils import load_parameters, get_embedding, cosine_score
from src.database import Database
from src.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "database.json"
WEIGHT_PATH = BASE_DIR / "assets" / "best_model_epoch9_20251001_064344.pt"

THRESHOLD = 0.54
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model on {device}")
model = ECAPA_TDNN(C=1024).to(device)
load_parameters(model, WEIGHT_PATH, device)
model.eval()
logger.info("Model loaded and set to eval mode")

db = Database(str(DATA_PATH))
app = FastAPI()
logger.info("FastAPI application initialized")

@app.get("/")
async def root():
    logger.debug("Health check endpoint called")
    return {"status": "running"}

@app.post("/enroll/{username}")
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

@app.post("/verify/{username}")
async def verify(username: str, password: str = Form(...), file: UploadFile = File(None)):
    logger.info(f"Verification request for user: {username}")

    # Check user exists
    user = db.get_user(username)
    if not user:
        logger.warning(f"Verification failed - user not found: {username}")
        raise HTTPException(status_code=404, detail="User not enrolled")

    # Verify password
    if not db.verify_password(username, password):
        logger.warning(f"Verification failed - incorrect password for user: {username}")
        raise HTTPException(status_code=401, detail="Invalid password")

    # Voice verification if file is provided
    if file is not None:
        emb_new = get_embedding(model, file, device)
        emb_ref = db.get_embedding(username)
        score = cosine_score(emb_new, emb_ref)
        result = "accepted" if score > THRESHOLD else "rejected"
        logger.info(f"Voice verification for {username}: score={score:.4f}, result={result}")
        return {"username": username, "method": "voice+password", "score": score, "result": result}

    logger.warning(f"Verification failed - no voice file provided for user: {username}")
    raise HTTPException(status_code=400, detail="Must provide voice")

@app.get("/users")
async def list_users():
    logger.debug("Listing all users")
    users = list(db.data.keys())
    return {"count": len(users), "users": users}
