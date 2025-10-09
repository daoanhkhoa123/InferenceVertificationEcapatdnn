import json
import os
import torch

from src.logger import get_logger
logger = get_logger(__name__)

class Database:
    def __init__(self, path="database.json"):
        self.path = path
        self.data = self._load()
        logger.info(f"Database initialized at {self.path} with {len(self.data)} users")

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} users from {self.path}")
                return data
        logger.info("No existing database found, starting with empty data")
        return {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4)
        logger.debug(f"Database saved to {self.path}")

    def add_user(self, username: str, voice_emb: torch.Tensor):
        if username in self.data:
            logger.warning(f"Attempt to add existing user: {username}")
            raise ValueError("Username already exists")
        emb_list = voice_emb.squeeze().cpu().numpy().tolist()
        self.data[username] = {"voice_emb": emb_list}
        self._save()
        logger.info(f"User added: {username}")

    def get_user(self, username: str):
        user = self.data.get(username)
        if user is None:
            logger.warning(f"User not found: {username}")
        return user

    def get_embedding(self, username: str):
        user = self.get_user(username)
        if not user:
            return None
        logger.debug(f"Retrieved embedding for user: {username}")
        return torch.tensor(user["voice_emb"], dtype=torch.float32)

    def update_embedding(self, username: str, new_emb: torch.Tensor):
        if username not in self.data:
            logger.error(f"Attempt to update non-existent user: {username}")
            raise ValueError("User not found")
        self.data[username]["voice_emb"] = new_emb.squeeze().cpu().numpy().tolist()
        self._save()
        logger.info(f"Updated embedding for user: {username}")
