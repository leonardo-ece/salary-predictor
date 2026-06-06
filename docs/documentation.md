# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section

Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:
> See *Data Preprocessing* in [`analysis.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/01_eda.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `app.py`, line 42:
> [`app.py`, line 42](https://github.com/leonardo-ece/salary-predictor/blob/main/app/app.py#L42)

---

## Project Metadata

- Project title: **SalaryScope AI — Job Salary Band Predictor**
- Student: leonardo-ece
- GitHub repository URL: https://github.com/leonardo-ece/salary-predictor
- Deployment URL: https://huggingface.co/spaces/eceleo/salaryscope-ai
- Submission date: 2026-06-07

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [ ] Computer Vision

Primary blocks used for core solution:

- Primary block 1: **NLP** — extracts structured features from raw job description text
- Primary block 2: **ML Numeric Data** — classifies salary band from structured + NLP-derived features

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- **Problem statement:** Job seekers and recruiters often lack transparency on whether a posted salary range is competitive. Raw job descriptions contain implicit signals (required skills, seniority cues, domain) that are hard to parse manually at scale.
- **Goal:** Build a system that takes a job posting (free text + basic structured fields) and predicts the salary band (Low / Mid / High / Very High), then explains the prediction in plain language.
- **Scope and assumptions:** Predictions are limited to US-based job postings with annual salaries between $30k–$200k. Job descriptions are assumed to be in English. The model predicts salary band only — not exact salary. Equity, bonuses, and benefits are not accounted for in the salary figures.
- **Success criteria:** Classification accuracy ≥ 55% on held-out test set (4-class problem, random baseline = 25%); qualitatively coherent LLM explanations for predicted bands. Both criteria were met: RF 60.4%, XGBoost 60.5%.

### 1.2 Integration Logic

- **How the selected blocks interact:** The NLP block processes raw job description text and produces two types of features: (1) TF-IDF + SVD numeric vectors capturing vocabulary patterns, and (2) LLM-extracted structured fields (domain, implied seniority, remote flag). Both feed directly into the ML classifier as input features alongside structured metadata. After prediction, the ML output (band + confidence) is passed back to the NLP block to generate a plain-language explanation.
- **Data and output flow between blocks:**

```
Job Description Text ──► [NLP Block]
  ├── TF-IDF (200 terms) → SVD (30 dims) → numeric feature vector
  └── LLM prompt extraction → domain, seniority, remote_friendly

Structured metadata ──────────────────────────────────────────┐
(experience_level, work_type)                                  ▼
                                              [ML Block] → salary_band prediction
                                                        → class probabilities
                                                              │
                                                              ▼
                                              [NLP Block] → plain-language explanation
```

> See pipeline diagram in [`README.md`](https://github.com/leonardo-ece/salary-predictor/blob/main/README.md)

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
|---|---|---|---|---|
| 1 | [LinkedIn Job Postings — Kaggle (arshkon)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Structured CSV | 123,849 rows, 27 columns | Primary structured features: experience level, work type, normalized salary |
| 2 | NLP block output (TF-IDF + SVD features) | Numeric matrix | 30 dimensions per row | Text-derived numeric features fed into classifier |
| 3 | NLP block output (LLM extraction) | Structured fields | 3 categorical + 1 binary | Domain, seniority, remote flag as additional features |

#### 2A.2 Preprocessing and Features

- **Cleaning steps:** Used `normalized_salary` column (already annualized) — 36,073 rows with salary info out of 123,849. Removed outliers outside 5th–95th percentile, leaving 26,866 rows in salary range $29,640–$197,000. Target variable `salary_band` created by binning into 4 bands: Low <50k, Mid 50–90k, High 90–140k, Very High >140k.
  > See *Create Target Variable* in [`notebooks/01_eda.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/01_eda.ipynb)
- **Preprocessing steps:** Label-encoded `formatted_experience_level` and `formatted_work_type`. LLM-extracted fields (domain, seniority) also label-encoded. Remote flag binarized to integer.
- **Feature engineering and selection:** Combined structured features (2) + LLM-extracted features (3) + TF-IDF/SVD text features (30) = 35 total features. Random Forest feature importance used post-hoc for analysis.
  > See *Combine All Features* in [`notebooks/03_modeling.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/03_modeling.ipynb)

#### 2A.3 Model Selection

- **Models tested:** Random Forest, XGBoost
- **Why these models were chosen:** Both handle mixed feature types (encoded categoricals + dense numeric vectors) well without scaling. Random Forest is robust to noisy features; XGBoost achieves higher accuracy via boosting. Comparing both gives a clear baseline vs. boosted model contrast on the same feature set.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
|---|---|---|---|---|---|
| 1 | Baseline with structured features only | `exp_encoded`, `work_encoded` only | RF, XGBoost | Accuracy | — |
| 2 | Add TF-IDF/SVD text features | Append 30 SVD components from job descriptions | RF, XGBoost | Accuracy | Text features improve discrimination of domain-specific salary signals |
| 3 | Add LLM-extracted features | Append domain, seniority, remote fields | RF, XGBoost | Accuracy | LLM fields add explicit semantic structure not captured by TF-IDF alone |

> See full comparison in [`notebooks/03_modeling.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/03_modeling.ipynb) and [`docs/ml_metrics.csv`](https://github.com/leonardo-ece/salary-predictor/blob/main/docs/ml_metrics.csv)

#### 2A.5 Evaluation and Error Analysis

- **Metrics used:** Accuracy, per-class Precision / Recall / F1, confusion matrix (80/20 train/test split, stratified)
- **Final results:**

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Random Forest | 60.4% | 0.57 | 0.59 |
| XGBoost | 60.5% | 0.59 | 0.60 |

Random Forest per-class results:
- High (90–140k): P=0.57, R=0.65, F1=0.61
- Low (<50k): P=0.76, R=0.46, F1=0.57
- Mid (50–90k): P=0.58, R=0.75, F1=0.65
- Very High (>140k): P=0.73, R=0.34, F1=0.47

XGBoost per-class results:
- High (90–140k): P=0.57, R=0.61, F1=0.59
- Low (<50k): P=0.70, R=0.54, F1=0.61
- Mid (50–90k): P=0.60, R=0.70, F1=0.65
- Very High (>140k): P=0.63, R=0.44, F1=0.52

- **Error patterns and likely causes:** Both models struggle most with "Very High (>140k)" — low recall (RF: 34%, XGB: 44%) due to class imbalance (only 913 test samples vs 1953 for Mid). Most misclassifications occur between adjacent bands (Mid↔High), which is expected as salary boundaries are arbitrary. "Low" precision is high but recall is low — the model is conservative about predicting low salaries.
  > See [`docs/figures/05_confusion_matrices.png`](https://github.com/leonardo-ece/salary-predictor/blob/main/docs/figures/05_confusion_matrices.png)

#### 2A.6 Integration with Other Block(s)

- **Inputs received from NLP block:** TF-IDF/SVD feature vectors (30 dims) and LLM-extracted fields (domain, seniority_implied, remote_friendly) — both computed in [`notebooks/02_nlp_features.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/02_nlp_features.ipynb)
- **Outputs provided to NLP block:** Predicted salary band + class probabilities → passed as context to the LLM explanation prompt in [`app/app.py`](https://github.com/leonardo-ece/salary-predictor/blob/main/app/app.py)

---

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
|---|---|---|---|---|
| 1 | [LinkedIn Job Postings — Kaggle (arshkon)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Text (description column) | 26,866 job descriptions after cleaning | TF-IDF feature extraction |
| 2 | [Job Description Dataset — Kaggle (ravindrasinghrana)](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) | Text CSV | 1,615,940 rows | Additional text corpus for NLP analysis and keyword-based domain inference |
| 3 | OpenAI GPT-4o-mini | LLM via API | 500 rows extracted + 10-sample prompt comparison | Structured field extraction + salary band explanation generation |

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:** Lowercased, HTML tags removed (`<[^>]+>`), non-alphabetic characters stripped, whitespace normalized. Applied consistently in both training and inference.
  > See `clean_text()` in [`notebooks/02_nlp_features.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/02_nlp_features.ipynb) and [`app/app.py`](https://github.com/leonardo-ece/salary-predictor/blob/main/app/app.py)
- **Prompt design:** Two extraction prompt versions tested. Prompt V1: verbose with full field descriptions and examples. Prompt V2: concise with minimal instructions. Final explanation prompt receives predicted band, confidence, domain, seniority, and skills to generate a 3–4 sentence career-advisor-style explanation.
  > See prompt templates in [`notebooks/02_nlp_features.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/02_nlp_features.ipynb)

#### 2B.3 Approach Selection

- **Approach used:** Two NLP approaches combined — (1) classical NLP (TF-IDF + SVD/LSA) for numeric feature extraction feeding the ML model; (2) prompt engineering with GPT-4o-mini for structured field extraction and natural language explanation generation.
- **Alternatives considered:** Sentence-transformers / BERT embeddings (richer semantic features but significant inference overhead); RAG over a salary database (out of scope given data availability).

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
|---|---|---|---|---|---|
| 1 | TF-IDF baseline | 200 features, unigrams only | TF-IDF → RF classifier | Downstream ML accuracy | — |
| 2 | Add bigrams + SVD | ngram_range=(1,2), TruncatedSVD 30 components | TF-IDF + SVD → RF | Variance explained: captures phrases like "machine learning", "team lead" | Bigrams improve semantic coverage |
| 3 | LLM Prompt V1 vs V2 | Verbose vs. concise prompt | GPT-4o-mini | Domain agreement: 6/10 (60%), Seniority agreement: 0/10 (0%) | V1 extracts more varied domains; V2 defaults to "other"/"unknown" |

> See [`docs/nlp_prompt_comparison.csv`](https://github.com/leonardo-ece/salary-predictor/blob/main/docs/nlp_prompt_comparison.csv)

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy:** TF-IDF features evaluated indirectly via downstream ML accuracy. LLM extraction evaluated by comparing domain/seniority agreement between Prompt V1 and V2 on 10 samples.
- **Results:** Domain agreement between prompts: 6/10 (60%). Seniority agreement: 0/10 (0%) — Prompt V2 consistently returned "unknown" for seniority, while V1 inferred mid/senior/junior from context. This confirms V1 is the better prompt for structured extraction. Prompt V1 was used for the full 500-row extraction and in the deployed app.
- **Error patterns and likely causes:** Prompt V2 over-generalizes (defaults to "other"/"unknown") because its concise instructions lack sufficient guidance for edge cases. LLM domain extraction struggles with generalist roles (e.g. "Operations Manager" at a tech company). TF-IDF struggles with short or template-style descriptions lacking discriminative vocabulary.

#### 2B.6 Integration with Other Block(s)

- **Inputs received from ML block:** Predicted salary band + per-class probabilities → passed to the explanation prompt as context in [`app/app.py`](https://github.com/leonardo-ece/salary-predictor/blob/main/app/app.py)
- **Outputs provided to ML block:** (1) TF-IDF/SVD vector (30 numeric dims) as input features; (2) LLM-extracted domain, seniority, remote flag as additional categorical features

---

### 2C. Computer Vision

N/A — Computer Vision block not selected for this project.

---

## 3. Deployment

- **Deployment URL:** https://huggingface.co/spaces/eceleo/salaryscope-ai
- **Main user flow:**
  1. User enters a job title, selects experience level and work type
  2. User pastes the full job description text
  3. App runs NLP pipeline: cleans text → TF-IDF → SVD (30 dims), and GPT-4o-mini prompt for domain/seniority/skills extraction
  4. Random Forest model predicts salary band with per-class probabilities
  5. GPT-4o-mini generates a plain-language explanation of the prediction
  6. Results displayed: predicted band + confidence, confidence breakdown per band, AI explanation
- **Screenshot:** See `docs/figures/app_screenshot.png`

> See [`app/app.py`](https://github.com/leonardo-ece/salary-predictor/blob/main/app/app.py) for full Gradio implementation

---

## 4. Execution Instructions

- **Environment setup:**
```bash
git clone https://github.com/leonardo-ece/salary-predictor
cd salary-predictor
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r app/requirements.txt
```

- **Data setup:**
  Download the following from Kaggle and place in `data/raw/`:
  - [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) → `postings.csv`
  - [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) → `job_descriptions.csv`

- **Set API key (Windows):**
```bash
set OPENAI_API_KEY=sk-your-key-here
```

- **Training — run notebooks in order:**
```bash
jupyter notebook
# Run: notebooks/01_eda.ipynb
# Run: notebooks/02_nlp_features.ipynb  (requires OPENAI_API_KEY)
# Run: notebooks/03_modeling.ipynb
```

- **Launch app locally:**
```bash
python app/app.py
# Opens at http://127.0.0.1:7860
```

- **Reproducibility notes:** All random seeds set to 42. Python 3.10+. Key dependencies: scikit-learn, xgboost, openai, gradio. Trained model artifacts in `src/*.pkl` — if available, skip notebook runs and launch app directly.

---

## 5. Optional Bonus Evidence

- [ ] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [ ] Extended evaluation
- [ ] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

**Evidence for selected bonus items:**

Three distinct data sources used with clearly different roles: (1) LinkedIn structured job data for ML features and target variable, (2) ravindrasinghrana text-heavy dataset for additional NLP corpus and keyword-based domain inference on the non-LLM portion of the data, (3) OpenAI GPT-4o-mini API used in two distinct roles — structured feature extraction during training and natural language explanation generation at inference time.
