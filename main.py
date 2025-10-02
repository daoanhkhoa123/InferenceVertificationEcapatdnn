import torch
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form

from model import ECAPA_TDNN
from ultils import load_parameters, get_embedding, cosine_score
from database import Database

WEIGHT_PATH = r""
THRESHOLD = 0.4249 # from testing
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ECAPA_TDNN(C=1024).to(device)
load_parameters(model, WEIGHT_PATH)
model.eval()

db = Database("database.json")
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "running"}

@app.post("/enroll/{username}")
async def enroll(username: str, password: str = Form(...), file: UploadFile = File(...)):
    emb = get_embedding(model, file, device)
    try:
        db.add_user(username, password, emb)
    except ValueError:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"status": "enrolled", "username": username}

@app.post("/verify/{username}")
async def verify(
    username: str,
    password: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    user = db.get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not enrolled")

    if file is not None:
        emb_new = get_embedding(model, file, device)
        emb_ref = db.get_embedding(username)
        score = cosine_score(emb_new, emb_ref)
        result = "accepted" if score > THRESHOLD else "rejected"
        return {"username": username, "method": "voice", "score": score, "result": result}

    if password is not None:
        if db.verify_password(username, password):
            return {"username": username, "method": "password", "result": "accepted"}
        else:
            raise HTTPException(status_code=401, detail="Wrong password")

    raise HTTPException(status_code=400, detail="Must provide either voice or password")
