# Rapport Global - Partie IA VentureLens

Ce document explique ce qui a ete fait dans la partie IA du projet **VentureLens - AI Business Validation Platform**. Il sert a comprendre les fichiers, les datasets, les modeles, les APIs et ce qu'il faut presenter devant le professeur.

## 1. Idee Generale Du Projet IA

L'objectif de la partie IA est d'aider un entrepreneur a evaluer une idee business avec plusieurs signaux:

| Module                     | Question a laquelle il repond                                                  |
| -------------------------- | ------------------------------------------------------------------------------ |
| Startup Success Prediction | Est-ce que ce projet ressemble a des startups qui reussissent ?                |
| Sentiment Analysis         | Les avis/opinions autour du sujet sont-ils positifs ou negatifs ?              |
| Market Analysis            | Le marche semble-t-il attractif selon des donnees economiques et de tendance ? |
| Specialist Recommendation  | Quels specialistes peuvent aider le porteur du projet ?                        |
| Business Validation Score  | Quel score global peut-on donner en combinant plusieurs signaux ?              |

Important: tous les modules ne sont pas des modeles ML supervises. Certains sont des moteurs de scoring ou de recommandation. Ce n'est pas un probleme si on le presente honnetement.

## 2. Architecture Actuelle

L'architecture actuelle est la suivante:

1. Les notebooks entrainent ou documentent les modules IA.
2. Les fichiers `.py` contiennent le code reusable des moteurs IA.
3. FastAPI expose les modeles sous forme d'endpoints HTTP.
4. Streamlit teste les APIs comme le feront plus tard Angular ou Spring Boot.

Le backend final de l'equipe pourra appeler FastAPI depuis Spring Boot.

## 3. Sources De Donnees Utilisees

| Source                           | Type                             | Utilisation                                            |
| -------------------------------- | -------------------------------- | ------------------------------------------------------ |
| `startup_success_dataset.csv`  | CSV local                        | Entrainement du modele Startup Success                 |
| UCI Sentiment Labelled Sentences | Dataset public                   | Entrainement du modele Sentiment Analysis              |
| World Bank Indicators API        | API publique officielle          | Indicateurs economiques pour Market Analysis           |
| Google Trends CSV export         | Fichier CSV exporte manuellement | Tendance de recherche pour Market Analysis             |
| `data/specialists_sample.csv`  | CSV synthetique temporaire       | Tester la recommandation de specialistes avant MongoDB |

La World Bank Indicators API donne acces a des milliers de series temporelles et permet des appels API v2. Source officielle: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation

Google Trends permet d'exporter les donnees de tendances en CSV depuis un graphique. Source officielle: https://support.google.com/trends/answer/4365538

Le dataset UCI contient 3000 phrases labellisees positives ou negatives, venant d'Amazon, IMDb et Yelp. Source officielle: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

## 4. Modele 1 - Startup Success Prediction

### Objectif

Predire si un projet/startup a une probabilite de succes.

Le target original etait:

| Valeur originale | Transformation |
| ---------------- | -------------- |
| IPO              | Success        |
| Acquisition      | Success        |
| Failure          | Failure        |

Le notebook transforme donc le probleme en classification binaire:

```text
target_binary = Success ou Failure
```

### Dataset

Fichier utilise:

```text
startup_success_dataset.csv
```

Colonnes principales du dataset:

| Feature                      | Signification                             |
| ---------------------------- | ----------------------------------------- |
| `funding_rounds`           | Nombre de tours de financement            |
| `founder_experience_years` | Experience du fondateur                   |
| `team_size`                | Taille de l'equipe                        |
| `market_size_billion`      | Taille du marche en milliards             |
| `product_traction_users`   | Nombre d'utilisateurs ou traction produit |
| `burn_rate_million`        | Argent consomme par periode               |
| `revenue_million`          | Revenus en millions                       |
| `investor_type`            | Type d'investisseur                       |
| `sector`                   | Secteur de la startup                     |
| `founder_background`       | Profil du fondateur                       |
| `outcome`                  | Target original                           |

### Feature Engineering

Le notebook cree des features derivees:

| Feature generee           | Formule                                          | Pourquoi                                  |
| ------------------------- | ------------------------------------------------ | ----------------------------------------- |
| `funding_per_round`     | `funding_total / funding_rounds` si disponible | Mesurer financement moyen par round       |
| `experience_per_round`  | `founder_experience_years / funding_rounds`    | Relier experience et maturite financement |
| `traction_per_employee` | `product_traction_users / team_size`           | Mesurer efficacite de l'equipe            |
| `burn_to_revenue_ratio` | `burn_rate_million / revenue_million`          | Mesurer risque financier                  |

Attention: `revenue_million` brut a ete retire du modele pour reduire le risque de fuite de donnees. Si le revenu est connu apres l'evenement final, il ne faut pas l'utiliser directement.

### Modeles testes

Le notebook compare:

| Modele              | Role                               |
| ------------------- | ---------------------------------- |
| Logistic Regression | Baseline interpretable             |
| Random Forest       | Modele arbre robuste               |
| Gradient Boosting   | Modele ensemble performant         |
| XGBoost             | Modele boosting souvent performant |

### Output

Le modele produit:

| Output                  | Signification               |
| ----------------------- | --------------------------- |
| `success_probability` | Probabilite de succes en %  |
| `prediction_label`    | Success ou Failure          |
| `confidence_score`    | Confiance globale du moteur |

### Fichiers lies

| Fichier                                                | Role                                                  |
| ------------------------------------------------------ | ----------------------------------------------------- |
| `startup_success_binary_pipeline.ipynb`              | Notebook d'entrainement                               |
| `artifacts/startup_success_binary_best_model.joblib` | Modele sauvegarde                                     |
| `business_validation_score_engine.py`                | Charge et utilise le modele                           |
| `business_validation_api.py`                         | Expose l'endpoint `/api/v1/startup-success/predict` |

### Limites

Le modele depend des secteurs presents dans son dataset. S'il recoit un secteur comme immobilier, restauration ou peinture, il peut calculer un score, mais la fiabilite diminue. C'est pour cela que le module Market Analysis a ete ajoute.

## 5. Modele 2 - Sentiment Analysis

### Objectif

Classer des avis ou opinions en:

```text
0 = negatif
1 = positif
```

Ce modele ne predit pas directement la reussite d'un business. Il mesure la perception des utilisateurs ou du marche.

### Dataset

Dataset utilise:

```text
sentiment labelled sentences/
```

Fichiers:

| Fichier                       | Source      |
| ----------------------------- | ----------- |
| `amazon_cells_labelled.txt` | Avis Amazon |
| `imdb_labelled.txt`         | Avis IMDb   |
| `yelp_labelled.txt`         | Avis Yelp   |

Le dataset contient 3000 phrases au total, avec 500 phrases positives et 500 phrases negatives par source.

### Features

Le texte est transforme avec TF-IDF:

| Etape                   | Role                                      |
| ----------------------- | ----------------------------------------- |
| Lowercase               | Uniformiser les mots                      |
| Suppression ponctuation | Reduire le bruit                          |
| Suppression espaces     | Nettoyer le texte                         |
| Stopwords               | Retirer certains mots peu informatifs     |
| TF-IDF                  | Convertir le texte en vecteurs numeriques |
| Unigrams/Bigrams        | Utiliser mots seuls et paires de mots     |

### Modeles testes

| Modele                  | Role                                   |
| ----------------------- | -------------------------------------- |
| Logistic Regression     | Modele interpretable pour texte        |
| Linear SVM              | Tres utilise pour classification texte |
| SGDClassifier           | Modele lineaire efficace               |
| Multinomial Naive Bayes | Baseline classique NLP                 |

### Output

| Output                      | Signification                |
| --------------------------- | ---------------------------- |
| `sentiment_label`         | positive ou negative         |
| `sentiment_score`         | score de positivite          |
| `average_sentiment_score` | moyenne sur plusieurs textes |

### Fichiers lies

| Fichier                                          | Role                                 |
| ------------------------------------------------ | ------------------------------------ |
| `sentiment_analysis_model.ipynb`               | Notebook d'entrainement              |
| `artifacts/sentiment_analysis_pipeline.joblib` | Pipeline TF-IDF + modele             |
| `business_validation_score_engine.py`          | Utilise le pipeline sentiment        |
| `business_validation_api.py`                   | Expose `/api/v1/sentiment/analyze` |

### Limites

Le dataset est propre mais petit. Il est suffisant pour une baseline de projet, mais pour un produit reel il faudra ajouter des avis plus proches des domaines business: Google Play, Trustpilot, Reddit, reviews d'applications, enquêtes utilisateurs, etc.

## 6. Module 3 - Market Analysis

### Point Important

Ce module n'est pas un modele ML supervise. Il n'y a pas un target universel public du type:

```text
market_success = 0 ou 1
```

Donc on ne doit pas dire:

> J'ai entraine un modele ML d'analyse de marche.

Il faut dire:

> J'ai construit un module de scoring marche data-driven. Il collecte des indicateurs economiques reels via API, peut integrer Google Trends, puis calcule un score explicable d'attractivite du marche.

### Pourquoi World Bank ?

World Bank est une source officielle, stable et publique. Elle permet d'obtenir des indicateurs macro-economiques par pays, par exemple:

| Feature                                   | Indicateur World Bank            |
| ----------------------------------------- | -------------------------------- |
| `gdp_current_usd`                       | GDP actuel en dollars            |
| `gdp_growth_percent`                    | Croissance GDP                   |
| `population_total`                      | Population                       |
| `internet_users_percent`                | % utilisateurs Internet          |
| `services_value_added_percent_gdp`      | Poids des services dans le GDP   |
| `industry_value_added_percent_gdp`      | Poids de l'industrie dans le GDP |
| `manufacturing_value_added_percent_gdp` | Poids manufacturing              |
| `urban_population_percent`              | Population urbaine               |

Ces donnees aident a estimer si un pays offre un contexte favorable pour un projet.

### Pourquoi Google Trends ?

Google Trends mesure l'interet de recherche pour un mot-cle. Pour une idee business, cela donne un signal de demande ou de curiosite du marche.

Exemple:

| Projet                 | Mot-cle                   |
| ---------------------- | ------------------------- |
| Plateforme immobiliere | `immobilier`            |
| Plateforme education   | `online learning`       |
| Food delivery          | `healthy food delivery` |

Google Trends n'est pas une verite absolue, mais c'est utile comme signal complementaire.

### CSV generes

| CSV                                        | Origine                                                        |
| ------------------------------------------ | -------------------------------------------------------------- |
| `data/market_projects_sample.csv`        | Ecrit manuellement comme exemples de projets a analyser        |
| `data/market_signals_dataset.csv`        | Genere par `market_data_collector.py` apres appel World Bank |
| `data/market_signals_scored_dataset.csv` | Genere apres calcul des scores marche                          |

Le fichier `market_projects_sample.csv` n'est pas un vrai dataset marche. C'est une liste de projets exemples. Le vrai enrichissement vient de l'API World Bank.

### Features utilisees pour le score marche

| Feature                        | Source                                                           |
| ------------------------------ | ---------------------------------------------------------------- |
| `market_size_billion`        | Estime depuis GDP et part sectorielle, ou fourni par utilisateur |
| `market_growth_rate_percent` | World Bank GDP growth ou input utilisateur                       |
| `competition_level`          | Input utilisateur ou futur module concurrentiel                  |
| `product_traction_users`     | Donnee projet                                                    |
| `search_trend_score`         | Google Trends CSV ou input manuel                                |
| `country`                    | Donnee projet                                                    |

### Formule du score

| Signal                | Poids |
| --------------------- | ----: |
| Taille du marche      |   25% |
| Croissance du marche  |   25% |
| Concurrence           |   20% |
| Traction              |   15% |
| Tendance de recherche |   10% |
| Fit geographique      |    5% |

### Fichiers lies

| Fichier                               | Role                                          |
| ------------------------------------- | --------------------------------------------- |
| `market_data_collector.py`          | Collecte World Bank et lit Google Trends CSV  |
| `market_analysis_score_engine.py`   | Calcule le score marche                       |
| `market_analysis_score_model.ipynb` | Notebook qui documente collecte + scoring     |
| `business_validation_api.py`        | Expose `/api/v1/market-analysis/score`      |
| `data/google_trends/README.md`      | Explique comment ajouter un CSV Google Trends |

### Ce qu'il faut dire devant le prof

Le module marche n'est pas un modele ML classique. C'est un moteur de scoring explicable base sur des donnees reelles. Cette approche est defendable parce que l'objectif est de produire un score interpretable, pas de predire une target artificielle.

## 7. Module 4 - Specialist Recommendation

### Objectif

Recommander les specialistes les plus adaptes a un projet.

### Pourquoi pas un dataset public ?

Parce que dans le produit final, on ne doit pas recommander des specialistes d'un dataset externe. On doit recommander les specialistes inscrits dans la plateforme VentureLens, donc les donnees viendront de MongoDB.

Le CSV actuel est seulement un jeu de test.

### Dataset temporaire

Fichier:

```text
data/specialists_sample.csv
```

Colonnes:

| Feature                 | Signification       |
| ----------------------- | ------------------- |
| `specialist_id`       | Identifiant         |
| `full_name`           | Nom                 |
| `expertise_domain`    | Domaine d'expertise |
| `skills`              | Competences         |
| `sectors`             | Secteurs connus     |
| `industry_experience` | Experience          |
| `hourly_rate`         | Tarif               |
| `languages`           | Langues             |
| `location`            | Localisation        |
| `average_rating`      | Note moyenne        |
| `reviews_count`       | Nombre d'avis       |
| `availability_status` | Disponibilite       |
| `bio`                 | Description         |
| `completed_projects`  | Projets termines    |

### Formule de recommandation

| Signal                              | Poids |
| ----------------------------------- | ----: |
| Similarite semantique projet/profil |   40% |
| Matching des competences            |   25% |
| Matching du secteur                 |   15% |
| Rating                              |   10% |
| Disponibilite                       |    5% |
| Budget/langue/localisation          |    5% |

### Output

| Output                | Signification                    |
| --------------------- | -------------------------------- |
| `recommended_score` | Score de compatibilite           |
| `reason`            | Explication de la recommandation |
| `score_details`     | Details des sous-scores          |

### Fichiers lies

| Fichier                                   | Role                                     |
| ----------------------------------------- | ---------------------------------------- |
| `specialist_recommendation_engine.py`   | Moteur de recommandation                 |
| `specialist_recommendation_model.ipynb` | Notebook de demonstration                |
| `data/specialists_sample.csv`           | Donnees test                             |
| `business_validation_api.py`            | Expose `/api/v1/specialists/recommend` |

## 8. Score Final Business Validation

Le score final combine actuellement:

| Module             | Poids |
| ------------------ | ----: |
| Startup Success    |   40% |
| Sentiment Analysis |   25% |
| Market Analysis    |   20% |
| Specialist/Risk    |   15% |

Le fichier responsable est:

```text
business_validation_score_engine.py
```

L'endpoint est:

```text
POST /api/v1/business-validation/score
```

Remarque produit: on peut choisir de separer la recommandation specialiste du score final. Dans ce cas, on retire le bloc 15% Specialist/Risk et on redistribue les poids. Pour l'instant, le code le garde pour tester l'idee d'un score global multi-signal.

## 9. APIs FastAPI

Fichier:

```text
business_validation_api.py
```

Endpoints:

| Endpoint                                   | Role                                            |
| ------------------------------------------ | ----------------------------------------------- |
| `GET /health`                            | Verifier que l'API fonctionne                   |
| `GET /api/v1/models/status`              | Voir si les modeles sont charges ou en fallback |
| `POST /api/v1/startup-success/predict`   | Prediction Startup Success                      |
| `POST /api/v1/sentiment/analyze`         | Analyse de sentiment                            |
| `POST /api/v1/market-analysis/score`     | Analyse marche avec collecte World Bank         |
| `POST /api/v1/specialists/recommend`     | Recommandation specialistes                     |
| `POST /api/v1/business-validation/score` | Score final                                     |

Commande:

```bash
uvicorn business_validation_api:app --reload --port 8004
```

## 10. Streamlit

Il existe deux apps Streamlit:

| Fichier                  | Role                                   | Recommandation                   |
| ------------------------ | -------------------------------------- | -------------------------------- |
| `streamlit_api_app.py` | Teste les modeles en appelant FastAPI  | A garder pour la demo principale |
| `streamlit_app.py`     | Teste les moteurs directement sans API | Optionnel, utile en debug        |

Pour le projet final, il vaut mieux utiliser:

```bash
streamlit run streamlit_api_app.py
```

Pourquoi ? Parce que cela simule mieux l'architecture finale: Angular/Spring Boot appelle une API IA.

## 11. Quels Fichiers Executer ?

### Pour entrainer ou regenerer les modeles

| Ordre | Fichier                                   | Pourquoi                                      |
| ----- | ----------------------------------------- | --------------------------------------------- |
| 1     | `startup_success_binary_pipeline.ipynb` | Genere le modele startup                      |
| 2     | `sentiment_analysis_model.ipynb`        | Genere le pipeline sentiment                  |
| 3     | `market_analysis_score_model.ipynb`     | Genere les datasets marche avec API et score  |
| 4     | `specialist_recommendation_model.ipynb` | Documente/teste la recommandation specialiste |

### Pour lancer l'application IA

Terminal 1:

```bash
pip install -r requirements.txt
uvicorn business_validation_api:app --reload --port 8004
```

Terminal 2:

```bash
streamlit run streamlit_api_app.py
```

### Fichiers `.py` a ne pas executer directement

Ces fichiers sont surtout importes par l'API:

| Fichier                                 | Execute directement ?   |
| --------------------------------------- | ----------------------- |
| `business_validation_score_engine.py` | Non, sauf test rapide   |
| `market_analysis_score_engine.py`     | Non, sauf test rapide   |
| `market_data_collector.py`            | Non, sauf test collecte |
| `specialist_recommendation_engine.py` | Non, sauf test rapide   |

## 12. Fichiers A Garder, Supprimer Ou Ne Pas Pousser

### A garder et pousser

| Fichier/dossier                            | Pourquoi                             |
| ------------------------------------------ | ------------------------------------ |
| `*.ipynb`                                | Preuve de travail, EDA, entrainement |
| `*.py` principaux                        | Code reutilisable API/modeles        |
| `requirements.txt`                       | Dependances                          |
| `AI_PROJECT_REPORT_FR.md`                | Rapport d'explication                |
| `startup_success_dataset.csv`            | Dataset startup                      |
| `sentiment labelled sentences/`          | Dataset sentiment                    |
| `data/specialists_sample.csv`            | Demo specialistes                    |
| `data/market_projects_sample.csv`        | Inputs exemples pour collecte marche |
| `data/market_signals_dataset.csv`        | Dataset marche genere depuis API     |
| `data/market_signals_scored_dataset.csv` | Dataset marche + scores              |

### A ne pas pousser sur GitHub normal

| Fichier                                                | Pourquoi                     |
| ------------------------------------------------------ | ---------------------------- |
| `artifacts/startup_success_binary_best_model.joblib` | Trop gros pour GitHub normal |

Ce modele fait environ 803 MB. Il faut soit le regenerer avec le notebook, soit utiliser Git LFS, soit le stocker dans Google Drive/OneDrive.

### Peut etre supprime ou ignore

| Fichier/dossier                  | Decision                                                  |
| -------------------------------- | --------------------------------------------------------- |
| `__pycache__/`                 | Supprimer/ignorer                                         |
| `.DS_Store`                    | Supprimer/ignorer                                         |
| `startup_pipeline_results.png` | Optionnel                                                 |
| `streamlit_app.py`             | Optionnel si tu gardes seulement `streamlit_api_app.py` |

Ne supprime pas les notebooks. Ils sont importants pour montrer l'EDA, l'entrainement et la methode.

## 13. EDA Et Evaluation

Oui, il faut analyser les graphes EDA. Le prof peut demander:

| Graphe              | Ce qu'il faut commenter                   |
| ------------------- | ----------------------------------------- |
| Distribution target | Est-ce que les classes sont equilibrees ? |
| Missing values      | Est-ce qu'il y a des valeurs manquantes ? |
| Correlation matrix  | Quelles features sont redondantes ?       |
| Boxplots            | Y a-t-il des outliers ?                   |
| Model comparison    | Quel modele marche le mieux et pourquoi ? |
| Confusion matrix    | Quelles classes sont mal predites ?       |

Pour les scores modeles, il faut commenter:

| Metrique  | Signification                                    |
| --------- | ------------------------------------------------ |
| Accuracy  | Proportion globale de predictions correctes      |
| Precision | Quand le modele predit Success, est-ce fiable ?  |
| Recall    | Le modele retrouve-t-il bien les vrais Success ? |
| F1-score  | Equilibre precision/recall                       |

Le F1-score est souvent plus important que l'accuracy si les classes sont desequilibrees.

## 14. Ce Qui Est Bon Actuellement

Le projet a maintenant une base correcte:

1. Deux vrais modeles ML supervises: Startup Success et Sentiment.
2. Un module Market Analysis plus honnete, base sur collecte de donnees reelles.
3. Un recommender specialistes coherent avec MongoDB futur.
4. Une API FastAPI utilisable par Spring Boot.
5. Une app Streamlit qui teste les APIs.
6. Un debut de structure compatible MLOps.

## 15. Ce Qui Reste A Faire

| Tache                                                               | Priorite             |
| ------------------------------------------------------------------- | -------------------- |
| Verifier que les notebooks s'executent du debut a la fin            | Haute                |
| Ajouter captures/interpretations des graphes EDA dans les notebooks | Haute                |
| Tester FastAPI avec `streamlit_api_app.py`                        | Haute                |
| Decider si Specialist/Risk reste dans le score final                | Moyenne              |
| Ajouter Git LFS ou retirer le gros `.joblib` du push              | Haute                |
| Ajouter monitoring/MLOps plus tard                                  | Moyenne              |
| Connecter Spring Boot a FastAPI                                     | Tache equipe backend |

## 16. Phrase Simple Pour La Soutenance

> Notre plateforme ne base pas la validation d'une idee sur un seul modele. Elle combine un modele de prediction startup, un modele NLP de sentiment, un module d'analyse marche base sur des donnees economiques reelles, et un moteur de recommandation de specialistes. Le score final est donc un score d'aide a la decision, pas une verite absolue.
