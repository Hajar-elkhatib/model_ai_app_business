from __future__ import annotations

import json
import threading
import time
from typing import Any

import requests
import uvicorn
from fastapi.testclient import TestClient

import business_validation_api
import chatbot_api
from rag_engine import retrieve_context


PROJECT_PAYLOAD: dict[str, Any] = {
    "project_name": "VentureLens Business Validator",
    "project_description": "A SaaS platform that uses AI and market data to validate startup ideas.",
    "sector": "SaaS",
    "funding_rounds": 1,
    "founder_experience_years": 5,
    "team_size": 8,
    "market_size_billion": 25,
    "product_traction_users": 12000,
    "burn_rate_million": 0.8,
    "revenue_million": 0.3,
    "investor_type": "angel",
    "founder_background": "first_time",
    "market_growth_rate_percent": 8,
    "competition_level": "medium",
    "search_trend_score": 65,
    "country": "Morocco",
    "country_code": "MAR",
    "keyword": "business validation",
    "use_world_bank": False,
    "opinions": [
        "Customers like the idea because validation is difficult.",
        "Some users complain that consulting is expensive.",
    ],
}


def compact(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return "..."
    if isinstance(value, dict):
        return {k: compact(v, depth + 1) for k, v in list(value.items())[:12]}
    if isinstance(value, list):
        return [compact(v, depth + 1) for v in value[:3]]
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "..."
    return value


def record(results: list[dict[str, Any]], name: str, response: Any) -> dict[str, Any] | None:
    try:
        body = response.json()
    except Exception:
        body = response.text
    item = {
        "name": name,
        "status_code": response.status_code,
        "ok": 200 <= response.status_code < 300,
        "response": compact(body),
    }
    results.append(item)
    return body if isinstance(body, dict) else None


def main() -> None:
    ai = TestClient(business_validation_api.app)
    bot = TestClient(chatbot_api.app)
    results: list[dict[str, Any]] = []

    record(results, "GET business /health", ai.get("/health"))
    record(results, "GET business /api/v1/models/status", ai.get("/api/v1/models/status"))
    record(results, "POST /api/v1/startup-success/predict", ai.post("/api/v1/startup-success/predict", json=PROJECT_PAYLOAD))
    record(results, "POST /api/v1/sentiment/analyze", ai.post("/api/v1/sentiment/analyze", json={"texts": PROJECT_PAYLOAD["opinions"]}))
    record(results, "POST /api/v1/market-analysis/score", ai.post("/api/v1/market-analysis/score", json=PROJECT_PAYLOAD))
    record(
        results,
        "POST /api/v1/specialists/recommend",
        ai.post(
            "/api/v1/specialists/recommend",
            json={
                "title": PROJECT_PAYLOAD["project_name"],
                "description": PROJECT_PAYLOAD["project_description"],
                "sector": PROJECT_PAYLOAD["sector"],
                "needs": ["market research", "go-to-market", "AI"],
                "budget_per_hour": 80,
                "preferred_language": "fr",
                "location": "Morocco",
                "top_k": 3,
            },
        ),
    )
    analyze = record(results, "POST /api/v1/business-validation/analyze", ai.post("/api/v1/business-validation/analyze", json=PROJECT_PAYLOAD))
    record(results, "POST /api/v1/business-validation/score", ai.post("/api/v1/business-validation/score", json=PROJECT_PAYLOAD))
    record(
        results,
        "POST /api/v1/reports/generate-content",
        ai.post(
            "/api/v1/reports/generate-content",
            json={
                "projectData": PROJECT_PAYLOAD,
                "analysisResult": analyze or {},
                "includeBusinessPlan": True,
            },
        ),
    )
    record(results, "GET chatbot /health", bot.get("/health"))
    record(results, "GET chatbot /api/v1/chatbot/status", bot.get("/api/v1/chatbot/status"))

    server = uvicorn.Server(
        uvicorn.Config(
            business_validation_api.app,
            host="127.0.0.1",
            port=8004,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(30):
        try:
            if requests.get("http://127.0.0.1:8004/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.2)

    record(
        results,
        "POST /api/v1/chatbot/message general RAG",
        bot.post(
            "/api/v1/chatbot/message",
            json={
                "message": "Explique brièvement le score Startup Success dans VentureLens.",
                "project_data": {},
                "conversation_history": [],
                "chat_id": "audit-chat",
                "user_id": "audit-user",
                "fast_mode": True,
            },
        ),
    )
    server.should_exit = True
    thread.join(timeout=5)

    rag = retrieve_context("business validation score startup success market opinion report specialists", top_k=10)
    print(json.dumps({"results": results, "rag_titles": [item.get("title") for item in rag]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
