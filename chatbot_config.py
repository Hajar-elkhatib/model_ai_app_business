"""
chatbot_config.py
─────────────────
Central configuration for the VentureLens Business Chatbot.

All secrets are loaded from environment variables (via python-dotenv).
No API key is ever hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env_path, override=True)
except ImportError:
    pass  # python-dotenv is optional; rely on real env vars


# ─────────────────────────────── API URLs ────────────────────────────────── #

FASTAPI_BASE_URL: str = os.getenv("FASTAPI_BASE_URL", "http://localhost:8004")
CHATBOT_API_PORT: int = int(os.getenv("CHATBOT_API_PORT", "8003"))

# ──────────────────────────── NVIDIA LLM ─────────────────────────────────── #

NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL: str = os.getenv("NVIDIA_MODEL", "")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "600"))
LLM_STREAM: bool = os.getenv("LLM_STREAM", "true").lower() in ("true", "1", "yes")
LLM_REASONING_BUDGET: int = int(os.getenv("LLM_REASONING_BUDGET", "0"))

# ──────────────────────────── Timeouts ───────────────────────────────────── #

API_TIMEOUT_SECONDS: int = int(os.getenv("API_TIMEOUT_SECONDS", "30"))
LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT", os.getenv("LLM_TIMEOUT_SECONDS", "120")))

# ──────────────────────────── RAG Settings ───────────────────────────────── #

RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() in ("true", "1", "yes")
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "2"))
CHATBOT_FAST_MODE: bool = os.getenv("CHATBOT_FAST_MODE", "true").lower() == "true"
KNOWLEDGE_DOCS_PATH: str = os.getenv(
    "KNOWLEDGE_DOCS_PATH",
    str(Path(__file__).resolve().parent / "data" / "knowledge_documents.json"),
)

# ──────────────────────────── Memory Settings ────────────────────────────── #

MEMORY_BACKEND: str = os.getenv("MEMORY_BACKEND", "local")  # "local" | "mongodb"
MEMORY_FILE_PATH: str = os.getenv(
    "MEMORY_FILE_PATH",
    str(Path(__file__).resolve().parent / "data" / "chat_memory.json"),
)
MONGODB_URI: str = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "nexusai")

# ────────────────── Existing API Endpoints ───────────────────────────────── #

ENDPOINTS: dict[str, str] = {
    "health": f"{FASTAPI_BASE_URL}/health",
    "models_status": f"{FASTAPI_BASE_URL}/api/v1/models/status",
    "startup_success": f"{FASTAPI_BASE_URL}/api/v1/startup-success/predict",
    "sentiment": f"{FASTAPI_BASE_URL}/api/v1/sentiment/analyze",
    "market_analysis": f"{FASTAPI_BASE_URL}/api/v1/market-analysis/score",
    "specialists": f"{FASTAPI_BASE_URL}/api/v1/specialists/recommend",
    "business_validation": f"{FASTAPI_BASE_URL}/api/v1/business-validation/score",
}

# ────────────────── Intent Detection Keywords ────────────────────────────── #

INTENT_KEYWORDS: dict[str, list[str]] = {
    "project_question": [
        "what do you think about my project", "what do you think about the project",
        "what do you think about my idea", "is my project a good idea",
        "is my idea good", "will it work in the market", "does it have market potential",
        "does my project have market potential", "can my idea succeed",
        "que penses-tu de mon projet", "que penses tu de mon projet",
        "que penses-tu de mon idee", "que penses tu de mon idee",
        "est-ce que mon idee peut reussir", "est ce que mon idee peut reussir",
        "est-ce que mon projet a un potentiel", "est ce que mon projet a un potentiel",
    ],
    "startup_analysis": [
        "startup", "predict success", "success probability", "startup idea",
        "analyze my startup", "startup score", "chance of success",
        "will my startup succeed", "startup prediction", "predict my startup",
        "idée startup", "analyser mon startup", "probabilité de succès",
    ],
    "market_analysis": [
        "market", "market potential", "market size", "market growth",
        "market score", "market opportunity", "market trends", "competition",
        "marché", "potentiel du marché", "taille du marché", "analyse du marché",
    ],
    "sentiment_analysis": [
        "sentiment", "opinion", "review", "feedback", "customer review",
        "analyze reviews", "customer feedback", "user opinion", "what do users think",
        "avis", "opinions", "sentiments", "commentaires clients",
    ],
    "specialist_recommendation": [
        "specialist", "recommend specialist", "find expert", "which expert",
        "consultant", "hire specialist", "need expert", "who can help",
        "spécialiste", "recommander un spécialiste", "trouver un expert",
    ],
    "business_validation": [
        "validate", "business validation", "validation score", "overall score",
        "validate my business", "business score", "final score", "validate idea",
        "valider", "score de validation", "valider mon projet", "score final",
    ],
    "business_recommendation": [
        "improve", "recommendation", "how can i improve", "what should i change",
        "suggestions", "advice", "optimize", "strengthen", "weakness",
        "améliorer", "recommandation", "conseils", "optimiser",
    ],
    "marketing_strategy": [
        "marketing", "marketing strategy", "go to market", "growth strategy",
        "customer acquisition", "brand strategy", "marketing plan", "promote",
        "stratégie marketing", "plan marketing", "acquisition clients",
    ],
    "business_plan": [
        "business plan", "generate plan", "create plan", "financial plan",
        "roadmap", "pitch deck", "executive summary", "plan d'affaires",
        "plan business", "générer un plan",
    ],
    "report_summary": [
        "report", "summary", "summarize", "generate report", "pdf report",
        "overview", "résumé", "rapport", "synthèse", "générer un rapport",
    ],
    "general_question": [
        "hello", "hi", "help", "what can you do", "explain", "tell me about",
        "how does it work", "what is", "bonjour", "salut", "aide",
    ],
}

# ──────────────────────────── Helpers ────────────────────────────────────── #


def is_llm_configured() -> bool:
    """Return True when NVIDIA has the required runtime configuration."""
    return bool(NVIDIA_API_KEY and NVIDIA_BASE_URL and NVIDIA_MODEL)


def is_fallback_mode() -> bool:
    """True when the chatbot will use template responses instead of an LLM."""
    return not is_llm_configured()
