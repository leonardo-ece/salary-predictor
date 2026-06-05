# 💼 SalaryScope AI — Job Salary Band Predictor

An AI-powered tool that combines **NLP** and **ML on Structured Data** to predict salary bands from job postings and explain the predictions in plain language.

## Blocks Used

| Block | Task |
|---|---|
| **NLP** | Extracts features from raw job description text (TF-IDF + LLM prompt engineering) |
| **ML Numeric Data** | Random Forest + XGBoost classify salary band from structured + NLP-derived features |

## Pipeline

```
Job Posting Text ──► [NLP Block] ──► Text features (TF-IDF / LLM-extracted) ──┐
                                                                                 ▼
Structured Data ─────────────────────────────────────────────────────────► [ML Block] ──► Salary Band + Explanation
(experience level, work type, domain)
```

## Data Sources

| Source | Type | Description |
|---|---|---|
| [LinkedIn Job Postings (arshkon)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Structured CSV | ~33,000 job postings with salary, seniority, location, company size |
| [Job Description Dataset (ravindrasinghrana)](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) | Text CSV | Raw job description text for NLP feature extraction |

## Project Structure

```
salary-predictor/
├── notebooks/
│   ├── 01_eda.ipynb              # EDA on structured job data
│   ├── 02_nlp_features.ipynb    # NLP preprocessing + feature extraction
│   └── 03_modeling.ipynb        # ML training, comparison, evaluation
├── src/
│   ├── rf_model.pkl              # Trained Random Forest
│   ├── xgb_model.pkl             # Trained XGBoost
│   ├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
│   └── scaler.pkl                # StandardScaler
├── app/
│   ├── app.py                    # Gradio deployment app
│   └── requirements.txt
├── data/
│   ├── raw/                      # Kaggle datasets (not in repo)
│   └── processed/                # Cleaned + merged CSVs
└── docs/
    ├── figures/                  # EDA + evaluation plots
    └── documentation.md
```

## Deployment

Live demo: https://huggingface.co/spaces/eceleo/salaryscope-ai

## Setup & Execution Instructions

### 1. Prerequisites

- Python 3.10+
- Git
- An [Anthropic API key](https://console.anthropic.com/)

### 2. Clone the repository

```bash
git clone https://github.com/leonardo-ece/salary-predictor
cd salary-predictor
```

### 3. Create a virtual environment

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r app/requirements.txt
```

### 5. Add datasets

Download the following datasets from Kaggle and place the CSVs in `data/raw/`:
- [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) → save as `job_postings.csv`
- [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) → save as `job_descriptions.csv`

### 6. Run notebooks in order

```bash
jupyter notebook
```

Run in this order: `01_eda.ipynb` → `02_nlp_features.ipynb` → `03_modeling.ipynb`

> Note: `02_nlp_features.ipynb` requires `ANTHROPIC_API_KEY` to be set as an environment variable.

### 7. Launch the app locally

```bash
export ANTHROPIC_API_KEY=sk-your-key-here
python app/app.py
```

Opens at http://127.0.0.1:7860

---

> ⚠️ **Disclaimer:** Student project for educational purposes only. Salary predictions are estimates based on historical data and should not be used for real hiring or job application decisions.
