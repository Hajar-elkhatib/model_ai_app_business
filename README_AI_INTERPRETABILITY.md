# IA, NVIDIA et interprétabilité

## Variables `.env`

Remplir manuellement ces variables, sans mettre de vraie clé dans Git :

```env
NVIDIA_API_KEY=your_nvidia_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=your_nvidia_model_here
LLM_TIMEOUT=120
LLM_MAX_TOKENS=600
RAG_ENABLED=true
RAG_TOP_K=2
CHATBOT_FAST_MODE=true
```

## Endpoints ajoutés

`POST /api/v1/startup-success/predict` garde ses champs existants et ajoute `explanation` avec `positiveFactors` et `negativeFactors`.

`POST /api/v1/business-validation/analyze` agrège la prédiction Startup Success, SHAP/fallback, Market Analysis, Sentiment si des feedbacks existent, score final, interprétation métier, needs et spécialistes.

`POST /api/v1/reports/generate-content` génère le contenu texte du rapport. Le PDF reste à gérer côté Spring Boot.

## Règle LLM

Le LLM NVIDIA ne modifie jamais les scores : `finalScore`, `startupSuccessScore`, `marketAnalysisScore`, `marketOpinionScore`, `successProbability`, `predictionLabel`.

## Commandes manuelles

```bash
pip install -r requirements.txt
uvicorn business_validation_api:app --reload --port 8004
uvicorn chatbot_api:app --reload --port 8003
python test_nvidia_llm_connection.py
```

## Exemples Swagger

`POST /api/v1/business-validation/analyze`

```json
{
  "project_name": "VentureLens",
  "project_description": "Plateforme SaaS qui valide les idées startup avec IA.",
  "sector": "SaaS",
  "funding_rounds": 2,
  "founder_experience_years": 5,
  "team_size": 8,
  "market_size_billion": 12,
  "product_traction_users": 12000,
  "burn_rate_million": 0.4,
  "revenue_million": 0.2,
  "investor_type": "angel",
  "founder_background": "technical",
  "market_growth_rate_percent": 8,
  "competition_level": "medium",
  "search_trend_score": 68,
  "country": "Morocco",
  "opinions": [
    "The product is useful and easy to understand.",
    "Pricing needs to be clearer."
  ]
}
```

`POST /api/v1/reports/generate-content`

```json
{
  "projectData": {
    "project_name": "VentureLens",
    "project_description": "Plateforme SaaS qui valide les idées startup avec IA.",
    "sector": "SaaS"
  },
  "analysisResult": {
    "scores": {
      "finalScore": 72,
      "startupSuccessScore": 70,
      "marketAnalysisScore": 75,
      "marketOpinionScore": 65
    },
    "strengths": ["Traction utilisateur favorable."],
    "weaknesses": ["Ratio dépenses / revenus défavorable."],
    "recommendations": ["Optimiser les dépenses avant d'accélérer."],
    "warnings": []
  },
  "includeBusinessPlan": true
}
```
