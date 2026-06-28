"""
chatbot_memory.py
─────────────────
Conversation memory manager for the VentureLens Business Chatbot.

Storage back-ends:
  • **local** (default) — JSON file on disk + in-memory dict.
  • **mongodb** — (future) MongoDB collection.

Data structures are compatible with the project class diagram:
  Chat, Message, AIRequest, AIResponse.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chatbot_config import MEMORY_BACKEND, MEMORY_FILE_PATH, MONGODB_DB_NAME, MONGODB_URI

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════ #
#                        DATA STRUCTURES                                    #
# ═══════════════════════════════════════════════════════════════════════════ #


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def create_chat(
    project_id: str | None = None,
    title: str = "New Chat",
    chat_type: str = "business_advisor",
    context_type: str = "project_validation",
) -> dict[str, Any]:
    """Create a Chat object compatible with the class diagram."""
    return {
        "id": _new_id(),
        "projectId": project_id,
        "title": title,
        "chatType": chat_type,
        "contextType": context_type,
        "createdAt": _now_iso(),
        "messages": [],
    }


def create_message(
    chat_id: str,
    role: str,
    content: str,
    sender_type: str = "user",
) -> dict[str, Any]:
    """Create a Message object compatible with the class diagram."""
    return {
        "id": _new_id(),
        "chatId": chat_id,
        "role": role,
        "content": content,
        "timestamp": _now_iso(),
        "senderType": sender_type,
    }


def create_ai_request(
    chat_id: str,
    prompt: str,
    request_type: str,
    endpoint: str = "",
    payload: dict[str, Any] | None = None,
    context_data: dict[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    """Create an AIRequest object compatible with the class diagram."""
    return {
        "id": _new_id(),
        "chatId": chat_id,
        "prompt": prompt,
        "requestType": request_type,
        "endpoint": endpoint,
        "payload": payload or {},
        "contextData": context_data or {},
        "status": status,
        "createdAt": _now_iso(),
    }


def create_ai_response(
    request_id: str,
    response_text: str,
    response_json: dict[str, Any] | None = None,
    response_type: str = "chatbot_answer",
    score: float | None = None,
    label: str | None = None,
    confidence_score: float | None = None,
    model_name: str = "",
    model_mode: str = "external_llm",
) -> dict[str, Any]:
    """Create an AIResponse object compatible with the class diagram."""
    return {
        "id": _new_id(),
        "requestId": request_id,
        "modelId": None,
        "responseText": response_text,
        "responseJson": response_json or {},
        "responseType": response_type,
        "score": score,
        "label": label,
        "confidenceScore": confidence_score,
        "modelName": model_name,
        "modelMode": model_mode,
        "createdAt": _now_iso(),
    }


# ═══════════════════════════════════════════════════════════════════════════ #
#                     MEMORY MANAGER                                        #
# ═══════════════════════════════════════════════════════════════════════════ #


class ChatMemoryManager:
    """Manage conversation history with pluggable back-ends."""

    def __init__(self) -> None:
        self._backend = MEMORY_BACKEND
        self._chats: dict[str, dict[str, Any]] = {}

        if self._backend == "local":
            self._file_path = Path(MEMORY_FILE_PATH)
            self._load_from_file()
        # Future: elif self._backend == "mongodb": ...

    # ── Public API ───────────────────────────────────────────────────── #

    def get_or_create_chat(
        self,
        chat_id: str | None = None,
        project_id: str | None = None,
        title: str = "New Chat",
    ) -> dict[str, Any]:
        """Return an existing chat or create a new one."""
        if chat_id and chat_id in self._chats:
            return self._chats[chat_id]

        chat = create_chat(project_id=project_id, title=title)
        if chat_id:
            chat["id"] = chat_id
        self._chats[chat["id"]] = chat
        self._persist()
        return chat

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        sender_type: str = "user",
        intent: str | None = None,
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Append a message to a chat and persist."""
        chat = self.get_or_create_chat(chat_id=chat_id)
        msg = create_message(chat_id, role, content, sender_type)
        if intent:
            msg["intent"] = intent
        if reasoning_details:
            msg["reasoning_details"] = reasoning_details
        chat["messages"].append(msg)
        self._persist()
        return msg

    def save_exchange(
        self,
        chat_id: str,
        user_message: str,
        assistant_answer: str,
        intent: str,
        api_results: dict[str, Any] | None = None,
        model_name: str = "",
        model_mode: str = "external_llm",
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Save a full user→assistant exchange with AIRequest/AIResponse."""
        self.add_message(chat_id, "user", user_message, sender_type="user", intent=intent)

        ai_req = create_ai_request(
            chat_id=chat_id,
            prompt=user_message,
            request_type=intent,
        )

        ai_resp = create_ai_response(
            request_id=ai_req["id"],
            response_text=assistant_answer,
            response_json=api_results,
            response_type=intent,
            model_name=model_name,
            model_mode=model_mode,
        )

        self.add_message(
            chat_id, "assistant", assistant_answer,
            sender_type="ai", intent=intent,
            reasoning_details=reasoning_details
        )

        self._persist()
        return {
            "chat_id": chat_id,
            "ai_request": ai_req,
            "ai_response": ai_resp,
        }

    def get_conversation_history(
        self,
        chat_id: str,
        max_messages: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the last *max_messages* messages for a chat."""
        chat = self._chats.get(chat_id)
        if not chat:
            return []
        return chat["messages"][-max_messages:]

    def list_chats(self) -> list[dict[str, Any]]:
        """Return a summary of all chats."""
        summaries = []
        for chat in self._chats.values():
            summaries.append(
                {
                    "id": chat["id"],
                    "title": chat.get("title", ""),
                    "chatType": chat.get("chatType", ""),
                    "createdAt": chat.get("createdAt", ""),
                    "message_count": len(chat.get("messages", [])),
                }
            )
        return summaries

    def clear_chat(self, chat_id: str) -> bool:
        """Delete a chat from memory."""
        if chat_id in self._chats:
            del self._chats[chat_id]
            self._persist()
            return True
        return False

    # ── Persistence ──────────────────────────────────────────────────── #

    def _persist(self) -> None:
        if self._backend == "local":
            self._save_to_file()

    def _load_from_file(self) -> None:
        if self._file_path.exists():
            try:
                with open(self._file_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._chats = {c["id"]: c for c in data} if isinstance(data, list) else data
                logger.info("Loaded %d chats from %s", len(self._chats), self._file_path)
            except Exception as exc:
                logger.warning("Could not load memory file: %s", exc)

    def _save_to_file(self) -> None:
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, "w", encoding="utf-8") as fh:
                json.dump(self._chats, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Could not save memory file: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════ #
#                FUTURE: MongoDB back-end stub                              #
# ═══════════════════════════════════════════════════════════════════════════ #
#
# class MongoDBChatMemory(ChatMemoryManager):
#     def __init__(self):
#         from pymongo import MongoClient
#         self.client = MongoClient(MONGODB_URI)
#         self.db = self.client[MONGODB_DB_NAME]
#         self.chats_col = self.db["chats"]
#         self.messages_col = self.db["messages"]
#         self.ai_requests_col = self.db["ai_requests"]
#         self.ai_responses_col = self.db["ai_responses"]
#
#     def save_exchange(self, ...):
#         # Insert into MongoDB collections instead of JSON file
#         ...
# ═══════════════════════════════════════════════════════════════════════════ #


# Module-level singleton
memory_manager = ChatMemoryManager()
