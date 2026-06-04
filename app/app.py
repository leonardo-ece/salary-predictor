import gradio as gr
import pandas as pd
import numpy as np
import pickle
import re
import json
import os
import anthropic

# ── Load models & encoders ──────────────────────────────────────────────────
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, '..', 'src')

rf_model     = load_pickle(os.path.join(SRC, 'rf_model.pkl'))
tfidf        = load_pickle(os.path.join(SRC, 'tfidf_vectorizer.pkl'))
svd          = load_pickle(os.path.join(SRC, 'svd.pkl'))
le_target    = load_pickle(os.path.join(SRC, 'label_encoder_target.pkl'))
le_exp       = load_pickle(os.path.join(SRC, 'label_encoder_exp.pkl'))
le_work      = load_pickle(os.path.join(SRC, 'label_encoder_work.pkl'))
le_domain    = load_pickle(os.path.join(SRC, 'label_encoder_domain.pkl'))
le_seniority = load_pickle(os.path.join(SRC, 'label_encoder_seniority.pkl'))

client = anthropic.Anthropic()

# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

EXTRACT_PROMPT = """You are a structured data extractor. Given a job description, extract the following fields.
Respond ONLY with valid JSON, no explanation.

Fields:
- domain: one of [tech, finance, healthcare, marketing, operations, education, other]
- seniority_implied: one of [junior, mid, senior, lead, unknown]
- top_skills: list of up to 5 key skills mentioned
- remote_friendly: true or false

Job description:
{description}
"""

EXPLAIN_PROMPT = """You are a helpful career advisor. A machine learning model just predicted the salary band for a job posting.

Job title: {title}
Predicted salary band: {band}
Model confidence: {confidence:.0%}
Job domain: {domain}
Implied seniority: {seniority}
Top skills found: {skills}

In 3-4 sentences, explain why this salary band makes sense given the job details. Be specific and helpful.
"""

def extract_llm_features(description):
    try:
        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=300,
            messages=[{'role': 'user', 'content': EXTRACT_PROMPT.format(description=description[:1500])}]
        )
        return json.loads(response.content[0].text.strip())
    except:
        return {'domain': 'other', 'seniority_implied': 'unknown', 'top_skills': [], 'remote_friendly': False}

def safe_encode(encoder, value, default='Unknown'):
    classes = list(encoder.classes_)
    val = value if value in classes else default
    if val not in classes:
        val = classes[0]
    return encoder.transform([val])[0]

def predict(job_title, job_description, experience_level, work_type):
    if not job_description.strip():
        return "Please enter a job description.", "", ""

    # NLP: extract LLM features
    llm_feats = extract_llm_features(job_description)
    domain     = llm_feats.get('domain', 'other')
    seniority  = llm_feats.get('seniority_implied', 'unknown')
    remote     = int(bool(llm_feats.get('remote_friendly', False)))
    skills     = ', '.join(llm_feats.get('top_skills', []))

    # TF-IDF features
    clean = clean_text(job_description)
    tfidf_vec = tfidf.transform([clean])
    tfidf_reduced = svd.transform(tfidf_vec)

    # Structured features
    exp_enc  = safe_encode(le_exp, experience_level)
    work_enc = safe_encode(le_work, work_type)
    dom_enc  = safe_encode(le_domain, domain)
    sen_enc  = safe_encode(le_seniority, seniority)

    structured = np.array([[exp_enc, work_enc, dom_enc, sen_enc, remote]])
    X = np.hstack([structured, tfidf_reduced])

    # Predict
    proba = rf_model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    pred_band = le_target.inverse_transform([pred_idx])[0]
    confidence = proba[pred_idx]

    # Confidence bar
    band_probs = {le_target.inverse_transform([i])[0]: f'{p:.0%}' for i, p in enumerate(proba)}
    probs_str = '\n'.join([f'{b}: {p}' for b, p in sorted(band_probs.items())])

    # LLM explanation
    try:
        explain_resp = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=300,
            messages=[{'role': 'user', 'content': EXPLAIN_PROMPT.format(
                title=job_title or 'Unknown',
                band=pred_band,
                confidence=confidence,
                domain=domain,
                seniority=seniority,
                skills=skills or 'not specified'
            )}]
        )
        explanation = explain_resp.content[0].text.strip()
    except Exception as e:
        explanation = f"Could not generate explanation: {e}"

    return pred_band, probs_str, explanation

# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="💼 SalaryScope AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 💼 SalaryScope AI
    ### Predict the salary band for any job posting using NLP + Machine Learning
    Paste a job description and let AI extract features, predict the salary band, and explain its reasoning.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            job_title = gr.Textbox(label="Job Title", placeholder="e.g. Senior Data Scientist")
            experience_level = gr.Dropdown(
                label="Experience Level",
                choices=["Entry level", "Mid-Senior level", "Associate", "Director", "Executive", "Internship", "Unknown"],
                value="Mid-Senior level"
            )
            work_type = gr.Dropdown(
                label="Work Type",
                choices=["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Unknown"],
                value="Full-time"
            )
            job_description = gr.Textbox(
                label="Job Description",
                placeholder="Paste the full job description here...",
                lines=12
            )
            predict_btn = gr.Button("🔍 Predict Salary Band", variant="primary")

        with gr.Column(scale=1):
            pred_output = gr.Textbox(label="📊 Predicted Salary Band", interactive=False)
            probs_output = gr.Textbox(label="📈 Confidence per Band", interactive=False, lines=5)
            explain_output = gr.Textbox(label="💡 AI Explanation", interactive=False, lines=8)

    predict_btn.click(
        fn=predict,
        inputs=[job_title, job_description, experience_level, work_type],
        outputs=[pred_output, probs_output, explain_output]
    )

    gr.Markdown("""
    ---
    *Student project — predictions are estimates based on historical LinkedIn data and should not be used for real hiring decisions.*
    """)

if __name__ == "__main__":
    demo.launch()
