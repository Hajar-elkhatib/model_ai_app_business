from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from llm_client import generate_json_response, get_llm_provider

logger = logging.getLogger(__name__)

ALLOWED_NEEDS = [
    "finance",
    "business plan",
    "market research",
    "go-to-market",
    "marketing",
    "UX",
    "customer experience",
    "AI",
    "cybersecurity",
    "legal",
]

ALLOWED_DECISIONS = {"increase", "decrease", "keep"}
ALLOWED_RELIABILITY = {"very_low", "low", "medium", "high"}
ALLOWED_DATA_QUALITY = {"very_poor", "poor", "acceptable", "good"}


def review_score_with_llm(
    project_data: dict[str, Any],
    raw_scores: dict[str, Any],
    shap_explanation: dict[str, Any],
) -> dict[str, Any]:
    """Review the raw model score with a bounded NVIDIA contextual adjustment."""
    raw_final = _bounded_score(_number(raw_scores.get("finalScore"), 0.0))
    quality_signals = _detect_data_quality(project_data)
    business_context = {
        "projectStage": _first_non_empty(project_data.get("project_stage"), project_data.get("projectStage"), "UNKNOWN"),
        "country": project_data.get("country"),
        "city": _first_non_empty(project_data.get("city"), project_data.get("region"), ""),
        "sector": project_data.get("sector"),
        "targetMarketScope": _first_non_empty(project_data.get("target_market_scope"), project_data.get("targetMarketScope"), ""),
    }
    system_prompt = """
You are NVIDIA LLM Score Review for VentureLens.
Return only one strict JSON object. Do not include markdown, prose, code fences, comments, or explanations outside JSON.
You may review the raw final score, but you must follow these rules:
- Output keys: adjustment, reviewedFinalScore, scoreReliability, dataQualityLevel, decision, reason, warnings.
- decision must be one of: increase, decrease, keep.
- scoreReliability must be one of: very_low, low, medium, high.
- dataQualityLevel must be one of: very_poor, poor, acceptable, good.
- adjustment must be a number.
- reviewedFinalScore must be a number between 0 and 100.
- warnings must be an array of strings.
- Normal adjustment limit is -15 to +15.
- If dataQualityLevel is very_poor, only keep or decrease; do not increase.
- If projectStage is IDEA_ONLY, missing users/revenue can justify a controlled increase, but only when problem, solution, target customers and context are meaningful.
- If projectStage is ALREADY_LAUNCHED, missing users, revenue or feedback should not receive a strong increase.
- Contradictory, fake, repeated, or meaningless data must reduce reliability and should decrease the score.
- The final reviewed score must be between 0 and 100.
Expected JSON shape:
{"adjustment": 0, "reviewedFinalScore": 0, "scoreReliability": "medium", "dataQualityLevel": "acceptable", "decision": "keep", "reason": "Short business reason.", "warnings": []}
"""
    payload = {
        "projectData": project_data,
        "rawScores": raw_scores,
        "shapExplanation": {
            "positiveFactors": shap_explanation.get("positiveFactors", []),
            "negativeFactors": shap_explanation.get("negativeFactors", []),
        },
        "businessContext": business_context,
        "localQualitySignals": quality_signals,
    }
    response = generate_json_response(system_prompt, json.dumps(payload, ensure_ascii=False))
    if response.get("fallback_mode") or response.get("error"):
        warning = _score_review_warning(response)
        logger.error(
            "NVIDIA score review unavailable | warning=%s | error_type=%s | error=%s | raw_response=%s",
            warning,
            response.get("error_type"),
            response.get("error"),
            response.get("raw_response_preview"),
        )
        return _score_review_fallback(raw_final, warning)

    parsed = response.get("json") or {}
    if not parsed:
        warning = _score_review_warning(response)
        logger.error(
            "NVIDIA score review parsing failed | warning=%s | parse_error=%s | raw_response=%s",
            warning,
            response.get("json_parse_error"),
            (response.get("response_text") or "")[:1000],
        )
        return _score_review_fallback(raw_final, warning)

    review = _sanitize_score_review(parsed, raw_final, project_data, quality_signals)
    provider_info = get_llm_provider()
    review["source"] = f"{provider_info['provider']}_llm" if provider_info["configured"] else "fallback_raw_model"
    return review


def _sanitize_score_review(
    review: dict[str, Any],
    raw_final: float,
    project_data: dict[str, Any],
    quality_signals: dict[str, Any],
) -> dict[str, Any]:
    data_quality = str(review.get("dataQualityLevel") or quality_signals["dataQualityLevel"]).strip().lower()
    if data_quality not in ALLOWED_DATA_QUALITY:
        data_quality = quality_signals["dataQualityLevel"]

    reliability = str(review.get("scoreReliability") or quality_signals["scoreReliability"]).strip().lower()
    if reliability not in ALLOWED_RELIABILITY:
        reliability = quality_signals["scoreReliability"]

    decision = str(review.get("decision") or "keep").strip().lower()
    if decision not in ALLOWED_DECISIONS:
        decision = "keep"

    requested_adjustment = _number(review.get("adjustment"), 0.0)
    project_stage = str(_first_non_empty(project_data.get("project_stage"), project_data.get("projectStage"), "")).upper()
    is_fake = bool(quality_signals.get("isFakeOrInsufficient"))

    min_adjustment, max_adjustment = -15.0, 15.0
    if data_quality == "very_poor":
        min_adjustment, max_adjustment = -30.0, 0.0
    elif project_stage == "IDEA_ONLY" and not is_fake:
        min_adjustment, max_adjustment = -15.0, 25.0

    if is_fake:
        max_adjustment = min(max_adjustment, 0.0)
        if requested_adjustment > 0:
            requested_adjustment = 0.0
        if raw_final > 20 and requested_adjustment == 0:
            requested_adjustment = -min(30.0, raw_final - 20.0)
        decision = "decrease" if requested_adjustment < 0 else "keep"
        data_quality = "very_poor"
        reliability = "very_low"

    bounded_adjustment = max(min_adjustment, min(max_adjustment, requested_adjustment))
    reviewed_score = _bounded_score(raw_final + bounded_adjustment)
    if abs(reviewed_score - raw_final) < 0.01:
        decision = "keep"
        bounded_adjustment = 0.0

    warnings = _clean_list(review.get("warnings"))
    for warning in quality_signals.get("warnings", []):
        if warning not in warnings:
            warnings.append(warning)

    return {
        "adjustment": round(bounded_adjustment, 2),
        "reviewedFinalScore": round(reviewed_score, 2),
        "scoreReliability": reliability,
        "dataQualityLevel": data_quality,
        "decision": decision,
        "reason": str(review.get("reason") or quality_signals["reason"]).strip(),
        "warnings": warnings,
    }


def _score_review_fallback(
    raw_final: float,
    warning: str = "NVIDIA score review unavailable: unknown error",
) -> dict[str, Any]:
    return {
        "source": "fallback_raw_model",
        "adjustment": 0.0,
        "reviewedFinalScore": round(_bounded_score(raw_final), 2),
        "scoreReliability": "medium",
        "dataQualityLevel": "unknown",
        "decision": "keep",
        "reason": f"{warning}; raw model score used.",
        "warnings": [warning],
    }


def _score_review_warning(response: dict[str, Any]) -> str:
    error = str(response.get("error") or "").strip()
    parse_error = str(response.get("json_parse_error") or "").strip()
    if "timeout" in error.lower():
        return "NVIDIA score review unavailable: timeout"
    if "api error" in error.lower() or "http" in error.lower():
        return "NVIDIA score review unavailable: API error"
    if parse_error:
        return f"NVIDIA score review unavailable: invalid JSON ({parse_error})"
    if error:
        return error
    return "NVIDIA score review unavailable: invalid JSON"


def _detect_data_quality(project_data: dict[str, Any]) -> dict[str, Any]:
    text_fields = [
        project_data.get("project_name"),
        project_data.get("project_description"),
        project_data.get("sector"),
        project_data.get("problem"),
        project_data.get("solution"),
        project_data.get("target_customers"),
        project_data.get("main_challenges"),
    ]
    present_fields = [value for value in text_fields if str(value or "").strip()]
    meaningful = [_meaningful_text(value) for value in present_fields]
    short_or_repeated = sum(1 for value in present_fields if _looks_fake_text(value))
    description = str(project_data.get("project_description") or "")
    has_business_context = len(description.split()) >= 10 or any(len(item.split()) >= 5 for item in meaningful)

    warnings: list[str] = []
    if short_or_repeated >= 3 or not has_business_context:
        warnings.append("Project information is too short or generic to support a reliable validation score.")
        return {
            "isFakeOrInsufficient": True,
            "scoreReliability": "very_low",
            "dataQualityLevel": "very_poor",
            "reason": "The project data appears incomplete, repeated, or not meaningful enough for reliable scoring.",
            "warnings": warnings,
        }
    if len(meaningful) < 4:
        warnings.append("Some important business context is missing.")
        return {
            "isFakeOrInsufficient": False,
            "scoreReliability": "low",
            "dataQualityLevel": "poor",
            "reason": "The project has partial context, but key validation inputs are still missing.",
            "warnings": warnings,
        }
    return {
        "isFakeOrInsufficient": False,
        "scoreReliability": "medium",
        "dataQualityLevel": "acceptable",
        "reason": "The project contains enough context for a controlled score review.",
        "warnings": warnings,
    }


def _meaningful_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text


def _looks_fake_text(value: Any) -> bool:
    text = _meaningful_text(value).lower()
    if not text:
        return True
    if len(text) <= 3:
        return True
    compact = re.sub(r"[^a-z0-9]", "", text)
    if compact and len(set(compact)) <= 2 and len(compact) <= 12:
        return True
    tokens = text.split()
    return len(tokens) <= 2 and len(text) < 8


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def build_business_interpretation(
    project_data: dict[str, Any],
    scores: dict[str, Any],
    startup_prediction: dict[str, Any],
    shap_explanation: dict[str, Any],
    market_analysis: dict[str, Any],
    market_opinion: dict[str, Any] | None,
) -> dict[str, Any]:
    deterministic = _deterministic_findings(shap_explanation, market_analysis, market_opinion, scores)
    llm_output = _ask_llm(
        project_data,
        scores,
        startup_prediction,
        shap_explanation,
        market_analysis,
        market_opinion,
        deterministic,
    )

    strengths = _clean_list(llm_output.get("strengths")) or deterministic["strengths"]
    weaknesses = _clean_list(llm_output.get("weaknesses")) or deterministic["weaknesses"]
    recommendations = _clean_list(llm_output.get("recommendations")) or _fallback_recommendations(weaknesses)
    interpretation = str(llm_output.get("interpretation") or deterministic["interpretation"])
    needs = map_weaknesses_to_needs(weaknesses, scores)
    provider_info = get_llm_provider()

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
        "interpretation": interpretation,
        "generatedNeeds": needs,
        "interpretationSource": (
            f"{provider_info['provider']}_llm" if provider_info["configured"] else "fallback_rules"
        ),
    }


def build_warnings(
    feedback_count: int,
    startup_model_mode: str,
    market_analysis: dict[str, Any],
    confidence_score: float | None,
) -> list[str]:
    warnings = []
    if feedback_count == 0:
        warnings.append("Aucun feedback marché n'a été analysé.")
    elif feedback_count < 5:
        warnings.append("Le score d'opinion repose sur peu de feedbacks.")
    if "fallback" in startup_model_mode.lower():
        warnings.append("Le modèle Startup Success fonctionne en mode fallback.")
    if _external_market_data_unavailable(market_analysis):
        warnings.append("Les données marché externes ne sont pas disponibles.")
    if confidence_score is not None and confidence_score < 60:
        warnings.append("La confiance de l'analyse est limitée.")
    return warnings


def map_weaknesses_to_needs(weaknesses: list[str], scores: dict[str, Any]) -> list[str]:
    text = " ".join(weaknesses).lower()
    needs: list[str] = []

    def add(*items: str) -> None:
        for item in items:
            if item in ALLOWED_NEEDS and item not in needs:
                needs.append(item)

    if re.search(r"burn|dépense|depense|revenu|revenue|finance|cash|rentabil", text):
        add("finance", "business plan")
    if re.search(r"traction|acquisition|visibil|marketing", text):
        add("marketing", "go-to-market")
    if re.search(r"concurr|competition|marché faible|market weak|marché.*faible|market.*low|potentiel marché|potentiel market", text):
        add("market research", "go-to-market")
    if re.search(
        r"expérience utilisateur|experience utilisateur|user experience|customer experience|"
        r"\bux\b|utilisabil|usability|interface|parcours|friction|satisfaction|"
        r"usage difficile|difficulté d'usage|difficulte d'usage|expérience client|experience client",
        text,
    ):
        add("UX", "customer experience")
    if re.search(r"expérience du fondateur|experience du fondateur|founder experience|fondateur.*limit|founder.*limited|premier lancement|first.?time", text):
        add("business plan")
    if re.search(r"\\bia\\b|\\bai\\b|intelligence artificielle|machine learning", text):
        add("AI")
    if re.search(r"sécurité|securite|security|cyber", text):
        add("cybersecurity")
    if re.search(r"juridique|legal|contrat|conform", text):
        add("legal")

    if scores.get("marketAnalysisScore", 100) < 50:
        add("market research", "go-to-market")
    return needs


def _ask_llm(
    project_data: dict[str, Any],
    scores: dict[str, Any],
    startup_prediction: dict[str, Any],
    shap_explanation: dict[str, Any],
    market_analysis: dict[str, Any],
    market_opinion: dict[str, Any] | None,
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = """
Tu es un conseiller business. Les scores numériques fournis viennent des modèles IA et du système de scoring.
Tu ne dois jamais les modifier, les recalculer ou les corriger.
Tu dois uniquement interpréter les résultats et générer des forces, faiblesses et recommandations cohérentes.
Retourne uniquement un objet JSON valide avec: strengths, weaknesses, recommendations, interpretation.
"""
    payload = {
        "project": project_data,
        "scores": scores,
        "startupPrediction": startup_prediction,
        "shapExplanation": shap_explanation,
        "marketAnalysis": market_analysis,
        "marketOpinion": market_opinion,
        "deterministicSignals": deterministic,
    }
    response = generate_json_response(system_prompt, json.dumps(payload, ensure_ascii=False))
    return response.get("json") or {}


def _deterministic_findings(
    shap_explanation: dict[str, Any],
    market_analysis: dict[str, Any],
    market_opinion: dict[str, Any] | None,
    scores: dict[str, Any],
) -> dict[str, Any]:
    strengths = [
        f"{item.get('label', item.get('feature'))} favorable."
        for item in shap_explanation.get("positiveFactors", [])[:3]
    ]
    weaknesses = [
        f"{item.get('label', item.get('feature'))} défavorable."
        for item in shap_explanation.get("negativeFactors", [])[:3]
    ]

    market_score = scores.get("marketAnalysisScore")
    opinion_score = scores.get("marketOpinionScore")
    if market_score is not None and market_score >= 60:
        strengths.append("Le potentiel marché est encourageant.")
    elif market_score is not None and market_score < 50:
        weaknesses.append("Le potentiel marché reste à valider.")

    if opinion_score is not None and opinion_score >= 60:
        strengths.append("L'opinion marché est positive.")
    elif opinion_score is not None and opinion_score < 50:
        weaknesses.append("L'opinion marché est mitigée ou négative.")

    if not strengths:
        strengths.append("Le projet dispose de signaux exploitables pour poursuivre la validation.")
    if not weaknesses:
        weaknesses.append("Les principaux risques doivent encore être précisés avec plus de données.")

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "interpretation": "Analyse générée à partir des scores existants, des facteurs explicatifs et des signaux marché.",
    }


def _fallback_recommendations(weaknesses: list[str]) -> list[str]:
    needs = map_weaknesses_to_needs(weaknesses, {})
    if not needs:
        return ["Collecter davantage de données terrain avant les décisions d'investissement."]
    return [f"Travailler le besoin prioritaire: {need}." for need in needs[:5]]


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _external_market_data_unavailable(market_analysis: dict[str, Any]) -> bool:
    features = market_analysis.get("features") or {}
    details = ((market_analysis.get("market_analysis") or {}).get("details") or {})
    inputs = details.get("input") or {}
    return features.get("search_trend_score") is None and inputs.get("search_trend_score") is None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
