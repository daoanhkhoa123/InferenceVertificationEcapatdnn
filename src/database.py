import json
import os
import torch

class Database:
    def __init__(self, path="database.json"):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4)

    def add_user(self, username: str, password: str, voice_emb: torch.Tensor):
        if username in self.data:
            raise ValueError("Username already exists")

        # Convert tensor to list (for JSON)
        emb_list = voice_emb.squeeze().cpu().numpy().tolist()

        self.data[username] = {
            "password": password,  # in production: hash this!
            "voice_emb": emb_list
        }
        self._save()

    def get_user(self, username: str):
        return self.data.get(username, None)

    def verify_password(self, username: str, password: str):
        user = self.get_user(username)
        if not user:
            return False
        return user["password"] == password

    def get_embedding(self, username: str):
        user = self.get_user(username)
        if not user:
            return None
        return torch.tensor(user["voice_emb"], dtype=torch.float32)

    def update_embedding(self, username: str, new_emb: torch.Tensor):
        if username not in self.data:
            raise ValueError("User not found")
        self.data[username]["voice_emb"] = new_emb.squeeze().cpu().numpy().tolist()
        self._save()
