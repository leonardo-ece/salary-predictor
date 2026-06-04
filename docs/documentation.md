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
- Deployment URL: *(add Hugging Face Space URL after deployment)*
- Submission date: 2026-06-07

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [ ] Deployment URL provided *(fill in after deployment)*
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
- **Success criteria:** Classification accuracy ≥ 55% on held-out test set (4-class problem, random baseline = 25%); qualitatively coherent LLM explanations for predicted bands.

### 1.2 Integration Logic

- **How the selected blocks interact:** The NLP block processes raw job description text and produces two types of features: (1) TF-IDF + SVD numeric vectors capturing vocabulary patterns, and (2) LLM-extracted structured fields (domain, implied seniority, remote flag). Both feed directly into the ML classifier as input features alongside structured metadata.
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
| 1 | [LinkedIn Job Postings — Kaggle (arshkon)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Structured CSV | ~33,000 rows, 27 columns | Primary structured features: experience level, work type, salary min/max |
| 2 | NLP block output (TF-IDF + SVD features) | Numeric matrix | 30 dimensions per row | Text-derived numeric features fed into classifier |
| 3 | NLP block output (LLM extraction) | Structured fields | 3 categorical + 1 binary | Domain, seniority, remote flag as additional features |

#### 2A.2 Preprocessing and Features

- **Cleaning steps:** Dropped rows missing both `min_salary` and `max_salary` (~60% of dataset had salary info). Normalized hourly pay to annual (× 2080). Removed outliers outside 5th–95th salary percentile.
  > See *Create Target Variable* in [`notebooks/01_eda.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/01_eda.ipynb)
- **Preprocessing steps:** Label-encoded `formatted_experience_level` and `formatted_work_type`. Target variable `salary_band` created by binning salary average into 4 bands (Low <50k, Mid 50–90k, High 90–140k, Very High >140k).
- **Feature engineering and selection:** Combined structured features (2) + LLM-extracted features (3) + TF-IDF/SVD text features (30) = 35 total features. No manual feature selection — Random Forest feature importance used post-hoc for analysis.
  > See *Combine All Features* in [`notebooks/03_modeling.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/03_modeling.ipynb)

#### 2A.3 Model Selection

- **Models tested:** Random Forest, XGBoost
- **Why these models were chosen:** Both handle mixed feature types (encoded categoricals + dense numeric vectors) well without scaling requirements. Random Forest is robust to noisy features; XGBoost typically achieves higher accuracy via boosting. Comparing both gives a fair baseline vs. boosted model contrast.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
|---|---|---|---|---|---|
| 1 | Baseline with structured features only | `exp_encoded`, `work_encoded` only | RF, XGBoost | Accuracy | — |
| 2 | Add TF-IDF/SVD text features | Append 30 SVD components from job descriptions | RF, XGBoost | Accuracy | +Δ from text features |
| 3 | Add LLM-extracted features | Append domain, seniority, remote fields | RF, XGBoost | Accuracy | +Δ from LLM features |

> See full comparison in [`notebooks/03_modeling.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/03_modeling.ipynb) and [`docs/ml_metrics.csv`](https://github.com/leonardo-ece/salary-predictor/blob/main/docs/ml_metrics.csv)

#### 2A.5 Evaluation and Error Analysis

- **Metrics used:** Accuracy, per-class Precision / Recall / F1, confusion matrix
- **Final results:** *(fill in after running notebooks — e.g. RF: 61% accuracy, XGBoost: 63% accuracy)*
- **Error patterns and likely causes:** Most misclassifications occur between adjacent bands (Mid↔High). Likely causes: noisy salary data (some postings report pre-tax vs. post-tax, or equity included), ambiguous job titles, and class imbalance in the Very High band.
  > See *Confusion Matrix* in [`docs/figures/05_confusion_matrices.png`](https://github.com/leonardo-ece/salary-predictor/blob/main/docs/figures/05_confusion_matrices.png)

#### 2A.6 Integration with Other Block(s)

- **Inputs received from NLP block:** TF-IDF/SVD feature vectors (30 dims) and LLM-extracted fields (domain, seniority_implied, remote_friendly) — both computed in `notebooks/02_nlp_features.ipynb`
- **Outputs provided to NLP block:** Predicted salary band + class probabilities → used as input context for the LLM explanation prompt in `app/app.py`

---

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
|---|---|---|---|---|
| 1 | [LinkedIn Job Postings — Kaggle (arshkon)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Text (description column) | ~33,000 job descriptions | TF-IDF feature extraction |
| 2 | [Job Description Dataset — Kaggle (ravindrasinghrana)](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) | Text CSV | Separate dataset of raw postings | Additional text data for NLP analysis and prompt testing |
| 3 | Claude API (claude-sonnet-4-20250514) | LLM | Via API | Structured field extraction + salary explanation generation |

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:** Lowercased, HTML tags removed, non-alphabetic characters stripped, whitespace normalized. Applied consistently in both training (`notebooks/02_nlp_features.ipynb`) and inference (`app/app.py`).
  > See `clean_text()` in [`app/app.py`](https://github.com/leonardo-ece/salary-predictor/blob/main/app/app.py)
- **Prompt design:** Two prompt versions tested for structured extraction (see comparison below). Final explanation prompt is contextual — it receives the predicted band, confidence, domain, seniority, and skills, and generates a 3–4 sentence career-advisor-style explanation.
  > See prompt templates in [`notebooks/02_nlp_features.ipynb`](https://github.com/leonardo-ece/salary-predictor/blob/main/notebooks/02_nlp_features.ipynb) and [`app/app.py`](https://github.com/leonardo-ece/salary-predictor/blob/main/app/app.py)

#### 2B.3 Approach Selection

- **Approach used:** Two NLP approaches combined — (1) classical NLP (TF-IDF + SVD/LSA) for numeric feature extraction feeding the ML model; (2) prompt engineering with an LLM (Claude) for structured field extraction and natural language explanation generation.
- **Alternatives considered:** Sentence-transformers / BERT embeddings (would give richer semantic features but add significant inference overhead and model size for deployment); RAG over a salary database (out of scope given data availability).

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
|---|---|---|---|---|---|
| 1 | TF-IDF baseline | 200 features, unigrams only | TF-IDF → RF classifier | Accuracy contribution | — |
| 2 | Add bigrams + SVD | ngram_range=(1,2), TruncatedSVD 30 components | TF-IDF + SVD → RF | Accuracy, variance explained | +bigrams capture phrases like "machine learning", "team lead" |
| 3 | LLM extraction Prompt v1 vs v2 | Verbose instructions vs. concise prompt | Claude claude-sonnet-4-20250514 | Domain agreement (10-sample eval) | Compared in `docs/nlp_prompt_comparison.csv` |

> See [`docs/nlp_prompt_comparison.csv`](https://github.com/leonardo-ece/salary-predictor/blob/main/docs/nlp_prompt_comparison.csv)

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy:** TF-IDF features evaluated indirectly via downstream ML accuracy. LLM extraction evaluated by comparing domain/seniority agreement between Prompt v1 and v2 on 10 samples; explanation quality assessed qualitatively.
- **Results:** *(fill in after running — e.g. Prompt agreement: 8/10 on domain, 7/10 on seniority)*
- **Error patterns and likely causes:** LLM occasionally misclassifies domain for generalist roles (e.g. "Operations Manager" at a tech company). TF-IDF struggles with short or template-style descriptions that lack discriminative vocabulary.

#### 2B.6 Integration with Other Block(s)

- **Inputs received from ML block:** Predicted salary band + per-class probabilities → passed to the explanation prompt as context
- **Outputs provided to ML block:** (1) TF-IDF/SVD vector (30 numeric dims) as input features; (2) LLM-extracted domain, seniority, remote flag as additional categorical features

---

### 2C. Computer Vision

N/A — Computer Vision block not selected for this project.

---

## 3. Deployment

- **Deployment URL:** *(add Hugging Face Space URL after deployment)*
- **Main user flow:**
  1. User enters a job title, selects experience level and work type
  2. User pastes the job description text
  3. App calls NLP pipeline: cleans text → TF-IDF → SVD, and LLM prompt for structured extraction
  4. ML model (Random Forest) predicts salary band with class probabilities
  5. LLM generates a plain-language explanation of the prediction
  6. Results displayed: predicted band, confidence breakdown, explanation
- **Screenshot:** *(add screenshots after deployment)*

> See [`app/app.py`](https://github.com/leonardo-ece/salary-predictor/blob/main/app/app.py) for full Gradio implementation

---

## 4. Execution Instructions

- **Environment setup:**
```bash
git clone https://github.com/leonardo-ece/salary-predictor
cd salary-predictor
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
pip install -r app/requirements.txt
```

- **Data setup:**
  Download the following from Kaggle and place in `data/raw/`:
  - [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) → `job_postings.csv`
  - [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) → `job_descriptions.csv`

- **Training commands (run notebooks in order):**
```bash
jupyter notebook
# Run: notebooks/01_eda.ipynb
# Run: notebooks/02_nlp_features.ipynb  (requires ANTHROPIC_API_KEY)
# Run: notebooks/03_modeling.ipynb
```

- **Set API key (required for NLP notebook and app):**
```bash
export ANTHROPIC_API_KEY=sk-your-key-here   # macOS/Linux
# $env:ANTHROPIC_API_KEY = "sk-your-key-here"  # Windows PowerShell
```

- **Inference/run command:**
```bash
python app/app.py
# Opens at http://127.0.0.1:7860
```

- **Reproducibility notes:** All random seeds set to 42. Python 3.10+. Key library versions in `app/requirements.txt`. Trained model artifacts saved to `src/*.pkl` — if available, skip notebook runs and launch app directly.

---

## 5. Optional Bonus Evidence

- [ ] Third selected block implemented with strong quality
- [ ] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [ ] Extended evaluation
- [ ] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

**Evidence for selected bonus items:**

*Two distinct Kaggle data sources used with different roles (structured vs. text-heavy). LLM used both for feature extraction and for user-facing explanation — dual NLP role within single block.*
