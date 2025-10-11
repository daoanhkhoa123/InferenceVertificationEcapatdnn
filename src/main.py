from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ultils_logger import get_logger
from src.database import Database
from src.voice_model import ECAPA_TDNN
from src.voice_ultils import load_parameters
from src import router_voice, router_chats

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "database.json"
WEIGHT_PATH = BASE_DIR / "assets" / "best_model_epoch9_20251001_064344.pt"

THRESHOLD = 0.54
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
logger.info(f"Loading model on {device}")
model = ECAPA_TDNN(C=1024).to(device)
load_parameters(model, WEIGHT_PATH, device)
model.eval()
logger.info("Model loaded and set to eval mode")

# Initialize database
db = Database(str(DATA_PATH))

# Initialize FastAPI app
app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI application initialized")

# Health check
@app.get("/")
async def root():
    logger.debug("Health check endpoint called")
    return {"status": "running"}

# Register routers
router_voice.init_voice_router(db, model, device, THRESHOLD)

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/chat"  # example URL
router_chats.init_chat_router(db, N8N_WEBHOOK_URL)

app.include_router(
    router_voice.router,
    prefix="/voice",
    tags=["voice"]
)

app.include_router(
    router_chats.router,
    prefix="/chats",
    tags=["chats"]
)
