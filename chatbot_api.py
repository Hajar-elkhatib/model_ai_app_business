"""
chatbot_api.py
──────────────
FastAPI application for VentureLens Advisor.

Run with:
    uvicorn chatbot_api:app --reload --port 8003
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot_config import (
    CHATBOT_API_PORT,
    FASTAPI_BASE_URL,
    RAG_ENABLED,
)
from chatbot_engine import handle_chat_message
from llm_client import get_llm_provider
from chatbot_db import get_business_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════ #
#                          APP SETUP                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

app = FastAPI(
    title="VentureLens Advisor API",
    version="1.0.0",
    description=(
        "AI-powered business advisor chatbot that orchestrates startup "
        "validation, market analysis, sentiment analysis and specialist "
        "recommendation services."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════ #
#                        PYDANTIC MODELS                                    #
# ═══════════════════════════════════════════════════════════════════════════ #


class ChatMessageInput(BaseModel):
    """Input schema for POST /api/v1/chatbot/message."""

    message: str = Field(..., min_length=1, description="User message / question")
    project_data: dict[str, Any] | None = Field(
        default=None,
        description="Project data (sector, financials, opinions, etc.)",
    )
    conversation_history: list[dict[str, Any]] | None = Field(
        default=None,
        description="Previous messages for context",
    )
    chat_id: str | None = Field(
        default=None,
        description="Chat session ID for memory continuity",
    )
    user_id: str | None = Field(
        default=None,
        description="User ID to fetch data from database if project_data is missing",
    )
    fast_mode: bool | None = Field(
        default=None,
        description="Enable fast mode (shorter responses, less context)",
    )


class ChatMessageOutput(BaseModel):
    """Output schema for POST /api/v1/chatbot/message."""

    intent: str
    answer: str
    api_results: dict[str, Any] = Field(default_factory=dict)
    api_errors: dict[str, Any] = Field(default_factory=dict)
    rag_context: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    memory_saved: dict[str, Any] = Field(default_factory=dict)
    fallback_mode: bool = False


# ═══════════════════════════════════════════════════════════════════════════ #
#                         ENDPOINTS                                         #
# ═══════════════════════════════════════════════════════════════════════════ #


@app.get("/health")
def health() -> dict[str, Any]:
    """Health-check endpoint."""
    return {
        "status": "ok",
        "service": "VentureLens Advisor API",
        "version": app.version,
    }


@app.get("/api/v1/chatbot/status")
def chatbot_status() -> dict[str, Any]:
    """Return the current chatbot configuration and diagnostic status."""
    provider_info = get_llm_provider()
    
    # Test business_validation_api health
    main_api_health = "unknown"
    try:
        import requests
        resp = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            main_api_health = "ok"
        else:
            main_api_health = f"error (HTTP {resp.status_code})"
    except Exception as e:
        main_api_health = f"unreachable ({e})"
        
    return {
        "status": "ok",
        "llm_provider": provider_info["provider"],
        "llm_model": provider_info["model"],
        "llm_configured": provider_info["configured"],
        "rag_enabled": RAG_ENABLED,
        "fastapi_base_url": FASTAPI_BASE_URL,
        "fallback_mode": provider_info["fallback_mode"],
        "chatbot_api_port": CHATBOT_API_PORT,
        "main_api_health": main_api_health,
    }


@app.post("/api/v1/chatbot/message", response_model=ChatMessageOutput)
def chat_message(payload: ChatMessageInput) -> dict[str, Any]:
    """Main chatbot endpoint.

    Receives a user message (and optional project data / history),
    detects intent, calls the appropriate AI services, retrieves
    knowledge, generates a response, and returns a structured JSON.
    """
    try:
        project_data = payload.project_data
        
        # If no project data provided but user_id is, try to fetch from DB
        if not project_data and payload.user_id:
            logger.info(f"Fetching business data for user: {payload.user_id}")
            project_data = get_business_data(payload.user_id)
            if project_data:
                logger.info(f"Found business data for user: {payload.user_id}")
            else:
                logger.info(f"No business data found for user: {payload.user_id}")
                
        result = handle_chat_message(
            user_message=payload.message,
            project_data=project_data,
            conversation_history=payload.conversation_history,
            chat_id=payload.chat_id,
            fast_mode=payload.fast_mode,
        )
        return result
    except Exception as exc:
        logger.exception("Error processing chatbot message")
        raise HTTPException(status_code=500, detail=f"Chatbot error: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════ #
#                        STARTUP LOG                                        #
# ═══════════════════════════════════════════════════════════════════════════ #


@app.on_event("startup")
def _log_startup_info() -> None:
    provider_info = get_llm_provider()
    logger.info("=" * 60)
    logger.info("  VentureLens Advisor API started")
    logger.info("  Port        : %s", CHATBOT_API_PORT)
    logger.info("  LLM Provider: %s", provider_info["provider"])
    logger.info("  LLM Model   : %s", provider_info["model"])
    logger.info("  LLM Ready   : %s", provider_info["configured"])
    logger.info("  RAG Enabled : %s", RAG_ENABLED)
    logger.info("  Fallback    : %s", provider_info["fallback_mode"])
    logger.info("  Main API    : %s", FASTAPI_BASE_URL)
    logger.info("=" * 60)
