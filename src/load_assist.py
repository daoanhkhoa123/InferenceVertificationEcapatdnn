import torch
import torch.nn.functional as F
import numpy as np
from fastapi import UploadFile
import json
from pathlib import Path
from typing import Literal

from src.voice_ultils import _load_any_format

# import get_model from your script
from src.aasist.main import get_model

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "aasist" / "config" / "AASIST.conf"
WEIGHT_PATH = BASE_DIR.parent / "assets" / "assist_best_model_epoch8_20251012_052209.pt"

# --- Load config ---
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
model_config = config["model_config"]
print(model_config)

def get_assist_model(device):
    # --- Build model ---
    assist_model = get_model(model_config, device) # type: ignore

    # --- Load weights ---
    state_dict = torch.load(WEIGHT_PATH, map_location=device)
    assist_model.load_state_dict(state_dict, strict=True)

    assist_model.eval()
    return assist_model

def infer_assist(model, file: UploadFile, device) -> Literal["bonafide", "spoofed"]:
    audio, sr = _load_any_format(file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = torch.FloatTensor(audio).to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        emb, logits = model(audio)
        probs = F.softmax(logits, dim=1)
        score = probs[:, 1].item()
        pred = int(score >= 0.5)
    return "bonafide" if pred == 1 else "spoofed"