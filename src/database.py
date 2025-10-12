import json
import os
import uuid
from typing import Literal, List, Dict, Optional, TypedDict
import torch

from src.ultils_logger import get_logger
logger = get_logger(__name__)

class ChatMessage(TypedDict):
    time: str
    role: Literal["bot", "human"]
    message: str


class SessionData(TypedDict):
    name: str
    messages: List[ChatMessage]


class UserData(TypedDict):
    username: str
    password: str
    voice_emb: List[float]
    sessions: Dict[str, SessionData]


### db ###
class Database:
    def __init__(self, path: str = "database.json"):
        self.path = path
        self.data: Dict[str, UserData] = self._load()
        logger.info(f"Database initialized at {self.path} with {len(self.data)} users")

    ### load/save ###
    def _load(self) -> Dict[str, UserData]:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                raw = json.load(f)
                logger.info(f"Loaded {len(raw)} users from {self.path}")
                return raw
        logger.info("No existing database found, starting with empty data")
        return {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=4)
        logger.debug(f"Database saved to {self.path}")

    ### helpers ###
    def get_username(self, username: str, strict: bool = True) -> Optional[str]:
        if username in self.data:
            return username
        logger.error(f"User not found: {username}")
        if strict:
            raise ValueError("User not found")
        return None

    def get_user(self, username: str, strict: bool = True) -> UserData:
        user = self.data.get(username)
        if user is None:
            logger.error(f"User not found: {username}")
            if strict:
                raise ValueError("User not found")
            return None  # type: ignore
        return user

    ### users ###
    def add_user(self, username: str, password: str, voice_emb: torch.Tensor):
        if username in self.data:
            raise ValueError("Username already exists")

        emb_list = voice_emb.squeeze().cpu().numpy().tolist()
        user: UserData = {
            "username": username,
            "password": password,
            "voice_emb": emb_list,
            "sessions": {}
        }
        self.data[username] = user
        self._save()
        logger.info(f"User added: {username}")

    def verify_password(self, username: str, password: str) -> bool:
        user = self.get_user(username)
        return user["password"] == password

    ### embedding ###
    def get_embedding(self, username: str, device="cpu"):
        user = self.get_user(username)
        return torch.tensor(user["voice_emb"], dtype=torch.float32, device=device)

    def update_embedding(self, username: str, new_emb: torch.Tensor):
        uname = self.get_username(username)
        self.data[uname]["voice_emb"] = new_emb.squeeze().cpu().numpy().tolist() # type: ignore
        self._save()

    ### sessions ###
    def create_session(self, username: str, session_name: str) -> str:
        uname = self.get_username(username)
        session_id = str(uuid.uuid4())
        session: SessionData = {
            "name": session_name,
            "messages": list()
        }
        self.data[uname]["sessions"][session_id] = session # type: ignore
        self._save()
        return session_id

    def add_message(self, username: str, session_id: str, msg: ChatMessage):
        user = self.get_user(username)
        sessions = user["sessions"]
        if session_id not in sessions:
            raise ValueError("Session not found")
        sessions[session_id]["messages"].append(msg)
        self._save()

    def get_session_messages(self, username: str, session_id: str) -> List[ChatMessage]:
        user = self.get_user(username)
        sessions = user["sessions"]
        if session_id not in sessions:
            raise ValueError("Session not found")
        
        return sessions[session_id]["messages"]

    def list_sessions(self, username: str) -> List[str]:
        user = self.get_user(username)
        return list(user["sessions"].keys())

    def get_session(self, username: str, session_id: str) -> SessionData:
        """
        Return a specific chat session by ID.
        Raises ValueError if not found.
        """
        user = self.get_user(username)
        sessions = user["sessions"]
        if session_id not in sessions:
            logger.error(f"Session not found for user={username}, session_id={session_id}")
            raise ValueError("Session not found")
        return sessions[session_id]

    def delete_session(self, username: str, session_id: str):
        """
        Delete a specific chat session by ID.
        Raises ValueError if not found.
        """
        uname = self.get_username(username)
        user = self.get_user(uname) # type: ignore
        sessions = user["sessions"]
        if session_id not in sessions:
            logger.error(f"Tried to delete non-existent session: user={username}, session_id={session_id}")
            raise ValueError("Session not found")

        del sessions[session_id]
        self._save()
        logger.info(f"Deleted session {session_id} for user {username}")
