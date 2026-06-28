from __future__ import annotations

import json
from typing import Any

from llm_client import generate_json_response, get_llm_provider


INSUFFICIENT = "Insufficient data available."


REPORT_SCHEMA = {
    "title": "Business Validation Report",
    "language": "en",
    "projectName": "",
    "executiveSummary": "",
    "businessPlan": {
        "problem": "",
        "solution": "",
        "targetCustomers": "",
        "valueProposition": "",
        "businessModel": "",
        "marketingStrategy": "",
        "nextSteps": [],
    },
    "analysis": {
        "finalScore": None,
        "startupSuccessScore": None,
        "marketAnalysisScore": None,
        "marketOpinionScore": None,
        "predictionLabel": "",
        "successProbability": None,
        "strengths": [],
        "weaknesses": [],
        "recommendations": [],
        "recommendedSpecialists": [],
        "warnings": [],
    },
}


def generate_report_content(
    project_data: dict[str, Any],
    analysis_result: dict[str, Any],
    include_business_plan: bool = True,
) -> dict[str, Any]:
    project = _normalize_project(project_data)
    analysis = _normalize_analysis(analysis_result)
    fallback = _fallback_report(project, analysis, include_business_plan)

    system_prompt = """
You generate professional English content for a business validation PDF report.
Return only valid JSON matching the expected schema.
Rules:
- Never recalculate or change any provided score.
- NVIDIA/LLM may write narrative text only.
- Do not return raw objects, empty maps, Python/Java list text, or technical fields.
- If source information is missing, write "Insufficient data available."
- Keep each section concise so the final PDF fits in two pages.
- Mention that the final score combines predictive model outputs with a controlled NVIDIA LLM contextual review based on project stage, data quality, SHAP factors, and local market context.
"""
    payload = {
        "project": project,
        "analysis": analysis,
        "includeBusinessPlan": include_business_plan,
        "expectedSchema": REPORT_SCHEMA,
        "fallbackContent": fallback,
    }
    response = generate_json_response(system_prompt, json.dumps(payload, ensure_ascii=False))
    parsed = response.get("json") or {}

    report = _merge_report(fallback, parsed)
    report["title"] = "Business Validation Report"
    report["language"] = "en"
    report["projectName"] = _clean_text(report.get("projectName")) or project["projectName"]
    report["executiveSummary"] = _clean_text(report.get("executiveSummary")) or fallback["executiveSummary"]
    report["businessPlan"] = _clean_business_plan(report.get("businessPlan"), fallback["businessPlan"])
    if not include_business_plan:
        report["businessPlan"] = {}
    report["analysis"] = _clean_analysis(report.get("analysis"), analysis)
    report["generationSource"] = (
        f"{get_llm_provider()['provider']}_llm"
        if get_llm_provider()["configured"]
        else "fallback_rules"
    )
    return report


def _fallback_report(
    project: dict[str, Any],
    analysis: dict[str, Any],
    include_business_plan: bool,
) -> dict[str, Any]:
    name = project["projectName"]
    sector = project["sector"]
    description = project["description"]
    problem = _first_non_empty(project.get("problem"), _infer_problem(description, sector))
    solution = _first_non_empty(project.get("solution"), description)
    customers = _first_non_empty(project.get("targetCustomers"), f"Early customers in the {sector} market.")
    value = _first_non_empty(
        project.get("valueProposition"),
        f"{name} helps customers address a clear business need in the {sector} space.",
    )
    business_model = _first_non_empty(
        project.get("businessModel"),
        _business_model_from_project(project),
    )
    marketing = _first_non_empty(
        project.get("marketingStrategy"),
        "Start with focused customer discovery, direct outreach, expert validation, and content that explains the problem clearly.",
    )
    next_steps = _as_list(analysis.get("recommendations"))[:4] or [
        "Clarify the customer problem and target segment.",
        "Collect more customer feedback and market evidence.",
        "Validate pricing, acquisition channels, and operational assumptions.",
    ]

    summary = (
        f"{name} is a {sector} project. The current validation score is "
        f"{_score_text(analysis.get('finalScore'))}. The final score combines predictive model outputs "
        "with a controlled NVIDIA LLM contextual review based on project stage, data quality, SHAP factors, "
        "and local market context."
    )

    return {
        "title": "Business Validation Report",
        "language": "en",
        "projectName": name,
        "executiveSummary": summary,
        "businessPlan": {
            "problem": problem,
            "solution": solution,
            "targetCustomers": customers,
            "valueProposition": value,
            "businessModel": business_model,
            "marketingStrategy": marketing,
            "nextSteps": next_steps,
        } if include_business_plan else {},
        "analysis": analysis,
    }


def _normalize_project(project_data: dict[str, Any]) -> dict[str, Any]:
    name = _first_non_empty(
        _get(project_data, "project_name"),
        _get(project_data, "projectName"),
        _get(project_data, "title"),
        "Selected project",
    )
    description = _first_non_empty(
        _get(project_data, "project_description"),
        _get(project_data, "projectDescription"),
        _get(project_data, "description"),
        INSUFFICIENT,
    )
    return {
        "projectName": name,
        "description": description,
        "sector": _first_non_empty(_get(project_data, "sector"), "Business"),
        "country": _first_non_empty(_get(project_data, "country"), _get(project_data, "region"), ""),
        "city": _first_non_empty(_get(project_data, "city"), _get(project_data, "region"), ""),
        "stage": _first_non_empty(_get(project_data, "project_stage"), _get(project_data, "projectStage"), _get(project_data, "projectStatus"), ""),
        "teamSize": _get(project_data, "team_size", _get(project_data, "teamSize")),
        "founderExperienceYears": _get(project_data, "founder_experience_years", _get(project_data, "founderExperienceYears")),
        "marketSizeBillion": _get(project_data, "market_size_billion", _get(project_data, "marketSizeBillion")),
        "marketGrowthRatePercent": _get(project_data, "market_growth_rate_percent", _get(project_data, "marketGrowthRatePercent")),
        "revenueMillion": _get(project_data, "revenue_million", _get(project_data, "revenueMillion")),
        "burnRateMillion": _get(project_data, "burn_rate_million", _get(project_data, "burnRateMillion")),
        "problem": _first_non_empty(_get(project_data, "problem"), ""),
        "solution": _first_non_empty(_get(project_data, "solution"), ""),
        "targetCustomers": _first_non_empty(_get(project_data, "target_customers"), _get(project_data, "targetCustomers"), ""),
        "valueProposition": _first_non_empty(_get(project_data, "value_proposition"), _get(project_data, "valueProposition"), ""),
        "businessModel": _first_non_empty(_get(project_data, "business_model"), _get(project_data, "businessModel"), ""),
        "marketingStrategy": _first_non_empty(_get(project_data, "marketing_strategy"), _get(project_data, "marketingStrategy"), ""),
    }


def _normalize_analysis(analysis_result: dict[str, Any]) -> dict[str, Any]:
    scores = _get(analysis_result, "scores", {}) or {}
    startup = _get(analysis_result, "startupPrediction", {}) or {}
    raw = _get(analysis_result, "rawAnalysis", {}) or {}
    raw_scores = _get(raw, "scores", {}) or {}
    raw_startup = _get(raw, "startupPrediction", {}) or {}

    return {
        "finalScore": _number(_first_value(
            _get(analysis_result, "llmReviewedFinalScore"),
            _get(scores, "finalScore"),
            _get(raw_scores, "finalScore"),
            _get(analysis_result, "finalScore"),
            _get(analysis_result, "rawModelFinalScore"),
        )),
        "startupSuccessScore": _number(_first_value(
            _get(scores, "startupSuccessScore"), _get(raw_scores, "startupSuccessScore"), _get(analysis_result, "startupSuccessScore")
        )),
        "marketAnalysisScore": _number(_first_value(
            _get(scores, "marketAnalysisScore"), _get(raw_scores, "marketAnalysisScore"), _get(analysis_result, "marketAnalysisScore")
        )),
        "marketOpinionScore": _number(_first_value(
            _get(scores, "marketOpinionScore"), _get(raw_scores, "marketOpinionScore"), _get(analysis_result, "marketOpinionScore")
        )),
        "predictionLabel": _first_non_empty(
            _get(startup, "predictionLabel"),
            _get(raw_startup, "predictionLabel"),
            _get(analysis_result, "predictionLabel"),
            "",
        ),
        "successProbability": _number(_first_value(
            _get(startup, "successProbability"), _get(raw_startup, "successProbability"), _get(analysis_result, "successProbability")
        )),
        "strengths": _as_list(_first_value(_get(analysis_result, "strengths"), _get(raw, "strengths"))),
        "weaknesses": _as_list(_first_value(_get(analysis_result, "weaknesses"), _get(raw, "weaknesses"))),
        "recommendations": _as_list(_first_value(_get(analysis_result, "recommendations"), _get(raw, "recommendations")))[:4],
        "recommendedSpecialists": _as_list(_first_value(_get(analysis_result, "recommendedSpecialists"), _get(raw, "recommendedSpecialists"))),
        "warnings": _as_list(_first_value(_get(analysis_result, "warnings"), _get(raw, "warnings"))),
        "interpretation": _first_non_empty(_get(analysis_result, "interpretation"), _get(raw, "interpretation"), ""),
        "methodologyNote": "The final score combines predictive model outputs with a controlled NVIDIA LLM contextual review based on project stage, data quality, SHAP factors, and local market context.",
    }


def _merge_report(fallback: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    report = dict(fallback)
    for key in ("projectName", "executiveSummary"):
        if _clean_text(parsed.get(key)):
            report[key] = parsed[key]
    if isinstance(parsed.get("businessPlan"), dict):
        report["businessPlan"] = parsed["businessPlan"]
    if isinstance(parsed.get("analysis"), dict):
        merged_analysis = dict(fallback["analysis"])
        for key, value in parsed["analysis"].items():
            if value not in (None, "", [], {}):
                merged_analysis[key] = value
        report["analysis"] = merged_analysis
    return report


def _clean_business_plan(value: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    plan = {}
    for key in ("problem", "solution", "targetCustomers", "valueProposition", "businessModel", "marketingStrategy"):
        plan[key] = _clean_text(source.get(key)) or _clean_text(fallback.get(key)) or INSUFFICIENT
    next_steps = _as_list(source.get("nextSteps")) or _as_list(fallback.get("nextSteps"))
    plan["nextSteps"] = next_steps[:4] or [INSUFFICIENT]
    return plan


def _clean_analysis(value: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    result = dict(fallback)
    for key in ("finalScore", "startupSuccessScore", "marketAnalysisScore", "marketOpinionScore", "successProbability"):
        result[key] = _number(_first_value(fallback.get(key), source.get(key)))
    result["predictionLabel"] = _clean_text(_first_non_empty(fallback.get("predictionLabel"), source.get("predictionLabel"), ""))
    for key in ("strengths", "weaknesses", "recommendations", "recommendedSpecialists", "warnings"):
        result[key] = _as_list(_first_value(fallback.get(key), source.get(key)))
    result["recommendations"] = result["recommendations"][:4]
    result["interpretation"] = _clean_text(_first_non_empty(fallback.get("interpretation"), source.get("interpretation"), ""))
    result["methodologyNote"] = _clean_text(_first_non_empty(fallback.get("methodologyNote"), source.get("methodologyNote"), ""))
    return result


def _business_model_from_project(project: dict[str, Any]) -> str:
    revenue = _number(project.get("revenueMillion"))
    if revenue and revenue > 0:
        return "The project should refine its pricing, revenue streams, and acquisition economics around the current revenue baseline."
    return "The business model still needs validation around pricing, revenue streams, acquisition costs, and delivery capacity."


def _infer_problem(description: str, sector: str) -> str:
    if description and description != INSUFFICIENT:
        return f"The project addresses a business need described by the founder: {description}"
    return f"The exact customer pain point in the {sector} market needs to be validated with more evidence."


def _score_text(value: Any) -> str:
    number = _number(value)
    return "not available" if number is None else f"{number:.1f}%"


def _get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    text = str(value).strip()
    if text in ("{}", "[]", "null", "None"):
        return ""
    return text


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "", [], {})]
    if isinstance(value, tuple):
        return [item for item in value if item not in (None, "", [], {})]
    text = _clean_text(value)
    return [text] if text else []


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None
