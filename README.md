# 💼 SalaryScope AI — Job Salary Band Predictor

An AI-powered tool that combines **NLP** and **ML on Structured Data** to predict salary bands from job postings and explain the predictions in plain language.

## Blocks Used

| Block | Task |
|---|---|
| **NLP** | Extracts features from raw job description text (TF-IDF + GPT-4o-mini prompt engineering) |
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
| [LinkedIn Job Postings (arshkon)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Structured CSV | 123,849 job postings with salary, seniority, location, company size |
| [Job Description Dataset (ravindrasinghrana)](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) | Text CSV | 1.6M raw job descriptions for NLP feature extraction |

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 60.4% | 0.57 |
| XGBoost | 60.5% | 0.59 |

## Deployment

Live demo: https://huggingface.co/spaces/eceleo/salaryscope-ai

## Project Structure

```
salary-predictor/
├── notebooks/
│   ├── 01_eda.ipynb              # EDA on structured job data
│   ├── 02_nlp_features.ipynb     # NLP preprocessing + feature extraction
│   └── 03_modeling.ipynb         # ML training, comparison, evaluation
├── src/
│   ├── rf_model.pkl              # Trained Random Forest
│   ├── xgb_model.pkl             # Trained XGBoost
│   ├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
│   └── svd.pkl                   # Fitted SVD reducer
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

## Setup & Execution Instructions

### 1. Prerequisites

- Python 3.10+
- Git
- An [OpenAI API key](https://platform.openai.com/)

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

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r app/requirements.txt
```

### 5. Add datasets

Download from Kaggle and place in `data/raw/`:
- [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) → `postings.csv`
- [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) → `job_descriptions.csv`

### 6. Set your OpenAI API key

**Windows:**
```bash
set OPENAI_API_KEY=sk-your-key-here
```

**macOS / Linux:**
```bash
export OPENAI_API_KEY=sk-your-key-here
```

### 7. Run notebooks in order

```bash
jupyter notebook
```

Run in this order: `01_eda.ipynb` → `02_nlp_features.ipynb` → `03_modeling.ipynb`

### 8. Launch the app locally

```bash
python app/app.py
```

Opens at http://127.0.0.1:7860

---

> ⚠️ **Disclaimer:** Student project for educational purposes only. Salary predictions are estimates based on historical data and should not be used for real hiring or job application decisions.

## Acknowledgements
Claude (Anthropic) was used as an AI assistant for coding support, debugging, and documentation.