"""
rag_engine.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Simple RAG (Retrieval-Augmented Generation) engine for the VentureLens chatbot.

First version:
  â€¢ Loads documents from ``data/knowledge_documents.json``.
  â€¢ Falls back to embedded sample documents when the file is absent.
  â€¢ Splits documents into chunks.
  â€¢ Uses TF-IDF (scikit-learn) for retrieval, with a keyword fallback.
  â€¢ Returns the top-k most relevant chunks.

Architecture notes for future upgrades:
  â€¢ The ``retrieve_context`` function is the public interface.  Replace its
    internals with MongoDB Atlas Vector Search when ready â€” callers won't
    change.
  â€¢ ``embeddingVector`` is stored as ``null`` for now; populate it later
    with a real embedding model.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chatbot_config import KNOWLEDGE_DOCS_PATH, RAG_ENABLED, RAG_TOP_K

logger = logging.getLogger(__name__)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#                     SAMPLE DOCUMENTS (EMBEDDED)                           #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #

_SAMPLE_DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "doc_001",
        "title": "Business Validation Score â€” How It Works",
        "content": (
            "The Business Validation Score is a composite metric that evaluates the overall "
            "viability of an entrepreneurial project. It combines four dimensions: Startup "
            "Success Score (40%), Market Sentiment Score (25%), Market Analysis Score (20%), "
            "and Specialist/Risk Score (15%). The final score ranges from 0 to 100. "
            "A score above 75 indicates strong validation, 60-75 is promising but needs "
            "further validation, 45-60 is risky and needs more proof, and below 45 is weak. "
            "The confidence score indicates how reliable the validation is based on data "
            "completeness and sector reliability."
        ),
        "source_type": "faq",
        "metadata": {"module": "business_validation"},
    },
    {
        "id": "doc_002",
        "title": "Startup Success Prediction â€” Model Explanation",
        "content": (
            "The Startup Success Prediction model is a supervised machine learning classifier "
            "trained on historical startup data. It takes as input: sector, funding rounds, "
            "founder experience years, team size, market size (billion $), product traction "
            "(users), burn rate (million $/month), revenue (million $), investor type, and "
            "founder background. The model outputs a success probability (0-100%) and a "
            "prediction label (Success or Failure). Feature engineering includes traction per "
            "employee and burn-to-revenue ratio. The model supports sectors: AI, Climate, "
            "Crypto, Ecommerce, Fintech, Health, and SaaS. For other sectors, a heuristic "
            "fallback is used with reduced confidence."
        ),
        "source_type": "documentation",
        "metadata": {"module": "startup_success"},
    },
    {
        "id": "doc_003",
        "title": "Market Analysis Score â€” Methodology",
        "content": (
            "The Market Analysis Score evaluates market potential using a data-driven scoring "
            "system. It is NOT a machine learning model. Data sources include: World Bank "
            "Indicators API (GDP growth, internet penetration, ease of business), Google Trends "
            "CSV (optional), and project inputs (market size, growth rate, competition level, "
            "traction). Sub-scores include: market size score, market growth score, competition "
            "score, traction score, and search trend score. The final market score is a weighted "
            "average of all sub-scores. The system can automatically collect macroeconomic "
            "indicators from the World Bank API for the specified country."
        ),
        "source_type": "documentation",
        "metadata": {"module": "market_analysis"},
    },
    {
        "id": "doc_004",
        "title": "Sentiment Analysis â€” How Customer Opinions Are Analyzed",
        "content": (
            "The Sentiment Analysis module uses a trained NLP pipeline to analyze customer "
            "reviews and opinions. Each text receives a sentiment score (0-100) and a label "
            "(positive or negative). The system calculates an average sentiment score across "
            "all provided texts and an overall label. When the trained model is unavailable, "
            "a keyword-based fallback is used. Positive keywords include: love, good, great, "
            "excellent, amazing, useful, demand, growing, recommend. Negative keywords include: "
            "bad, poor, expensive, complain, hate, slow, uncertain, problem, difficult. "
            "For best results, provide at least 3-5 customer opinions or reviews."
        ),
        "source_type": "documentation",
        "metadata": {"module": "sentiment_analysis"},
    },
    {
        "id": "doc_005",
        "title": "Specialist Recommendation â€” Matching Algorithm",
        "content": (
            "The Specialist Recommendation engine matches entrepreneurs with relevant experts. "
            "The matching score is computed from: semantic similarity between project description "
            "and specialist bio/expertise (TF-IDF cosine similarity), skills match (overlap "
            "between project needs and specialist skills), sector match (bonus for same sector), "
            "rating (specialist's average rating), availability (available specialists get "
            "priority), budget compatibility (hourly rate vs project budget), language match, "
            "and location proximity. Results are ranked by the composite recommendation score. "
            "In production, specialist profiles come from MongoDB. In demo mode, sample CSV "
            "data is used."
        ),
        "source_type": "documentation",
        "metadata": {"module": "specialist_recommendation"},
    },
    {
        "id": "doc_006",
        "title": "Platform FAQ â€” Frequently Asked Questions",
        "content": (
            "Q: What is VentureLens? A: VentureLens is an intelligent platform for analyzing and "
            "validating entrepreneurial projects using AI, market data, and sentiment analysis.\n\n"
            "Q: How accurate is the prediction? A: The prediction is based on historical data "
            "and should be used as guidance, not as a definitive answer. The confidence score "
            "indicates reliability.\n\n"
            "Q: What data do I need? A: At minimum, you need: project name, description, "
            "sector, and basic financials (funding, team size, revenue, burn rate).\n\n"
            "Q: Can I use it for any sector? A: Yes, but the ML model works best for: AI, "
            "SaaS, Fintech, Health, Ecommerce, Climate, and Crypto. Other sectors use a "
            "heuristic fallback.\n\n"
            "Q: Is my data secure? A: Yes, all data is processed locally on the server and "
            "is not shared with third parties.\n\n"
            "Q: How does the chatbot work? A: The chatbot detects your intent, calls the "
            "appropriate AI services, retrieves relevant knowledge, and generates a "
            "comprehensive response."
        ),
        "source_type": "faq",
        "metadata": {"module": "general"},
    },
    {
        "id": "doc_007",
        "title": "Help Center â€” Getting Started",
        "content": (
            "Welcome to VentureLens! Here's how to get started:\n\n"
            "1. Create a project: Enter your project name, description, and sector.\n"
            "2. Fill in project details: Add financial data (funding, revenue, burn rate), "
            "team information, and market data.\n"
            "3. Run analysis: Use the platform to predict startup success, analyze market "
            "potential, evaluate customer sentiment, and find specialist consultants.\n"
            "4. Get validation: The Business Validation Score combines all analyses into a "
            "single comprehensive score.\n"
            "5. Chat with the AI: Use the chatbot to ask questions, get recommendations, "
            "generate business plans, and create marketing strategies.\n"
            "6. Generate reports: Export your analysis results as structured reports.\n\n"
            "Tips:\n"
            "- Provide as much data as possible for more accurate results.\n"
            "- Include customer opinions for better sentiment analysis.\n"
            "- Specify your country for relevant market data.\n"
            "- Use the chatbot to explore different aspects of your analysis."
        ),
        "source_type": "help_center",
        "metadata": {"module": "general"},
    },
    {
        "id": "doc_008",
        "title": "Understanding Confidence Scores",
        "content": (
            "The confidence score (0-100%) indicates how reliable the validation result is. "
            "It depends on: sector reliability (known sectors score higher), number of opinions "
            "provided (more opinions = higher confidence), and structured data completeness "
            "(more filled fields = higher confidence). A confidence score above 80% means the "
            "analysis is highly reliable. Between 60-80% means moderately reliable. Below 60% "
            "means the analysis should be interpreted with caution. To improve confidence: "
            "use a well-known sector, provide at least 5 customer opinions, and fill in all "
            "project data fields including country, competition level, and market growth rate."
        ),
        "source_type": "documentation",
        "metadata": {"module": "business_validation"},
    },
    {
        "id": "doc_009",
        "title": "Sector Support and Mapping",
        "content": (
            "The platform supports all sectors, but ML model accuracy varies. Fully supported "
            "sectors (trained model): AI, SaaS, Fintech, Health, Ecommerce, Climate, Crypto. "
            "Mapped sectors (keyword detection): Software â†’ SaaS, Machine Learning â†’ AI, "
            "Banking â†’ Fintech, Medical â†’ Health, Retail â†’ Ecommerce, Green Energy â†’ Climate, "
            "Blockchain â†’ Crypto. Other sectors (heuristic fallback): Real Estate, Education, "
            "Agriculture, Food, Transport, Construction, Art. The system uses keyword matching "
            "in both the sector name and project description to find the best mapping. Sector "
            "reliability ranges from 1.0 (exact match) to 0.55 (unknown sector)."
        ),
        "source_type": "documentation",
        "metadata": {"module": "startup_success"},
    },
    {
        "id": "doc_010",
        "title": "API Integration Guide",
        "content": (
            "The VentureLens API provides the following endpoints:\n\n"
            "GET /health â€” Check API health status.\n"
            "GET /api/v1/models/status â€” Check loaded models and their status.\n"
            "POST /api/v1/startup-success/predict â€” Predict startup success probability.\n"
            "POST /api/v1/sentiment/analyze â€” Analyze text sentiment.\n"
            "POST /api/v1/market-analysis/score â€” Score market potential.\n"
            "POST /api/v1/specialists/recommend â€” Get specialist recommendations.\n"
            "POST /api/v1/business-validation/score â€” Get comprehensive business validation.\n\n"
            "All endpoints accept JSON payloads and return JSON responses. "
            "For the current local setup, Spring Boot runs on port 8080, "
            "the main FastAPI AI service runs on port 8004, the chatbot API runs on port 8003, "
            "Angular runs on port 4200, and MongoDB runs on port 27017. "
            "Angular calls Spring Boot, and Spring Boot calls the FastAPI services."
        ),
        "source_type": "documentation",
        "metadata": {"module": "api"},
    },
]

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#                         DOCUMENT LOADING                                  #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #


def load_documents(path: str | None = None) -> list[dict[str, Any]]:
    """Load knowledge documents from the JSON file or fall back to samples.

    Parameters
    ----------
    path : str | None
        Override for the document file path.  Defaults to the value in config.
    """
    doc_path = Path(path or KNOWLEDGE_DOCS_PATH)

    if doc_path.exists():
        try:
            with open(doc_path, "r", encoding="utf-8") as fh:
                docs = json.load(fh)
            logger.info("Loaded %d documents from %s", len(docs), doc_path)
            return docs
        except Exception as exc:
            logger.warning("Failed to load %s: %s â€” using samples", doc_path, exc)

    logger.info("Knowledge file not found at %s â€” using embedded samples.", doc_path)

    # Persist samples so the user can edit them later
    try:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(doc_path, "w", encoding="utf-8") as fh:
            json.dump(_SAMPLE_DOCUMENTS, fh, indent=2, ensure_ascii=False)
        logger.info("Created sample knowledge file at %s", doc_path)
    except Exception as exc:
        logger.warning("Could not write sample file: %s", exc)

    return list(_SAMPLE_DOCUMENTS)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#                          CHUNKING                                         #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    """Split documents into smaller chunks for retrieval.

    Each chunk retains a reference to its source document (``doc_id``,
    ``title``, ``source_type``).
    """
    chunks: list[dict[str, Any]] = []
    for doc in documents:
        content = doc.get("content", "")
        doc_id = doc.get("id", "unknown")
        title = doc.get("title", "")
        source_type = doc.get("source_type", "unknown")
        metadata = doc.get("metadata", {})

        if len(content) <= chunk_size:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_chunk_0",
                    "doc_id": doc_id,
                    "title": title,
                    "content": content,
                    "source_type": source_type,
                    "metadata": metadata,
                }
            )
        else:
            words = content.split()
            start = 0
            chunk_idx = 0
            while start < len(words):
                end = start + chunk_size
                chunk_text = " ".join(words[start:end])
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                        "doc_id": doc_id,
                        "title": title,
                        "content": chunk_text,
                        "source_type": source_type,
                        "metadata": metadata,
                    }
                )
                start = end - chunk_overlap
                chunk_idx += 1

    logger.info("Created %d chunks from %d documents.", len(chunks), len(documents))
    return chunks


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#                         RETRIEVAL                                         #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #

# Module-level cache so we don't reload on every call.
_CHUNKS_CACHE: list[dict[str, Any]] | None = None


def _ensure_chunks() -> list[dict[str, Any]]:
    global _CHUNKS_CACHE
    if _CHUNKS_CACHE is None:
        docs = load_documents()
        _CHUNKS_CACHE = chunk_documents(docs)
    return _CHUNKS_CACHE


def retrieve_context(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for *query*.

    Strategy (in priority order):
    1. scikit-learn TF-IDF cosine similarity (if sklearn is installed).
    2. Simple keyword overlap fallback.

    Parameters
    ----------
    query : str
        The user's question or search query.
    top_k : int | None
        Number of chunks to return.  Defaults to ``RAG_TOP_K`` from config.

    Returns
    -------
    list[dict]
        Each dict has keys: chunk_id, doc_id, title, content, source_type,
        metadata, relevance_score.
    """
    if not RAG_ENABLED:
        return []

    top_k = top_k or RAG_TOP_K
    chunks = _ensure_chunks()
    if not chunks:
        return []

    try:
        return _tfidf_retrieve(query, chunks, top_k)
    except Exception:
        logger.info("TF-IDF retrieval unavailable, using keyword fallback.")
        return _keyword_retrieve(query, chunks, top_k)


def _tfidf_retrieve(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Retrieve using scikit-learn TF-IDF + cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = [c["content"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])

    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            chunk = dict(chunks[idx])
            chunk["relevance_score"] = round(float(similarities[idx]), 4)
            results.append(chunk)
    return results


def _keyword_retrieve(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Fallback keyword-overlap retrieval (no sklearn needed)."""
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored: list[tuple[float, int]] = []
    for idx, chunk in enumerate(chunks):
        chunk_tokens = set(_tokenize(chunk["content"]))
        title_tokens = set(_tokenize(chunk.get("title", "")))
        all_tokens = chunk_tokens | title_tokens
        if not all_tokens:
            continue
        overlap = len(query_tokens & all_tokens)
        score = overlap / (math.sqrt(len(query_tokens)) * math.sqrt(len(all_tokens)))
        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, idx in scored[:top_k]:
        chunk = dict(chunks[idx])
        chunk["relevance_score"] = round(score, 4)
        results.append(chunk)
    return results


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#                        FORMATTING                                         #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #


def format_retrieved_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable string for the LLM prompt."""
    if not chunks:
        return "No relevant internal documents found."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}] {chunk.get('title', 'Untitled')} "
            f"(type: {chunk.get('source_type', 'unknown')}, "
            f"relevance: {chunk.get('relevance_score', 'N/A')})\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#                 FUTURE: MongoDB Atlas Vector Search                       #
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #
#
# To switch to MongoDB Atlas Vector Search:
#
# 1. Store documents with real embedding vectors in MongoDB.
# 2. Create an Atlas Search index with vector search type.
# 3. Replace ``retrieve_context`` internals:
#
#     from pymongo import MongoClient
#
#     client = MongoClient(MONGODB_URI)
#     db = client[MONGODB_DB_NAME]
#     collection = db["knowledge_documents"]
#
#     def retrieve_context_mongo(query, top_k=3):
#         query_embedding = embed(query)  # your embedding function
#         results = collection.aggregate([
#             {
#                 "$vectorSearch": {
#                     "index": "vector_index",
#                     "path": "embeddingVector",
#                     "queryVector": query_embedding,
#                     "numCandidates": top_k * 10,
#                     "limit": top_k,
#                 }
#             }
#         ])
#         return list(results)
#
# The rest of the chatbot code does not need to change.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• #


def reload_documents() -> None:
    """Force-reload documents from disk (useful after adding new docs)."""
    global _CHUNKS_CACHE
    _CHUNKS_CACHE = None
    _ensure_chunks()

