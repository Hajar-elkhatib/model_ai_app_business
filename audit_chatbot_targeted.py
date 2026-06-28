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


PROJECT_DATA: dict[str, Any] = {
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


def compact(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            "intent": data.get("intent"),
            "fallback_mode": data.get("fallback_mode"),
            "answer_preview": str(data.get("answer", ""))[:500],
            "sources_used": data.get("sources_used"),
            "rag_titles": [item.get("title") for item in data.get("rag_context", [])],
        }
    return data


def main() -> None:
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

    client = TestClient(chatbot_api.app)
    results: list[dict[str, Any]] = []
    for name, message, project_data in [
        ("status", None, {}),
        ("business_plan", "Génère un business plan court pour ce projet.", PROJECT_DATA),
        ("out_of_domain", "Qui a gagné la coupe du monde 2018 ?", {}),
    ]:
        if message is None:
            response = client.get("/api/v1/chatbot/status")
            body = response.json()
        else:
            response = client.post(
                "/api/v1/chatbot/message",
                json={
                    "message": message,
                    "project_data": project_data,
                    "conversation_history": [],
                    "chat_id": f"audit-{name}",
                    "user_id": "audit-user",
                    "fast_mode": True,
                },
            )
            body = compact(response.json())
        results.append({"name": name, "status_code": response.status_code, "ok": response.status_code == 200, "response": body})

    server.should_exit = True
    thread.join(timeout=5)
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
