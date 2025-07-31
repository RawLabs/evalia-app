import openai
import os
import json
import logging
import re
import io
import base64
from datetime import datetime, timezone
from PIL import Image
import streamlit as st
import requests
from fpdf import FPDF
import pandas as pd
import plotly.express as px

# Setup logging
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Key Check
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set in environment. Please configure it in Streamlit Cloud.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)
MEMORY_FILE = "evalia_memory.json"

def initialize_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w') as f:
            json.dump([], f)

def save_to_memory(entry):
    with open(MEMORY_FILE, 'r+') as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)

def sanitize_input(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^\w\s\.,!?]', '', text)
    return " ".join(text.strip().split())

def analyze_image(img_file):
    try:
        image_bytes = img_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        prompt = "Extract all legible text. Return raw text only."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise OCR and information extraction assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
        return sanitize_input(response.choices[0].message.content) or "No text extracted."
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return f"Error extracting text: {str(e)}"

def fetch_url_text(url):
    try:
        r = requests.get(url, timeout=10)
        return sanitize_input(r.text[:3000])
    except Exception as e:
        logger.error(f"URL fetch error: {e}")
        return f"Error fetching URL: {str(e)}"

def detect_style_gpt(claim):
    styles_list = ["Symbolic/metaphysical", "Conspiratorial", "Academic", "Emotional/personal", "Basic/blunt", "Exploratory"]
    prompt = (
        f"Classify the style of the following claim into one of these categories: {', '.join(styles_list)}. "
        "Consider keywords and tone. "
        "Return ONLY in this exact format: 'Style: X, Confidence: Y' where X is the style and Y is a number from 0.0 to 1.0."
        f"\n\nClaim: {claim}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content.strip()
    match = re.search(r"Style:\s*(.+?)\s*,\s*Confidence:\s*(\d*\.?\d+)", content, re.IGNORECASE)
    if match:
        style = match.group(1).strip()
        confidence = float(match.group(2))
        return style, confidence
    return "Exploratory", 0.1

def feedback_style_instruction(style):
    return {
        "Symbolic/metaphysical": "Respect metaphorical framing; gently guide toward evidence without dismissing symbolic meaning.",
        "Conspiratorial": "Be clear and firm on facts, but maintain a respectful and curious tone. Avoid ridicule.",
        "Academic": "Use precise language and address theoretical constructs with clarity.",
        "Emotional/personal": "Be compassionate. Frame critique as support for personal growth.",
        "Basic/blunt": "Keep it simple, direct, and helpful without jargon.",
        "Exploratory": "Encourage inquiry and deeper exploration in a welcoming, curious tone."
    }.get(style, "Encourage inquiry and deeper exploration in a welcoming, curious tone.")

SCORING_PROMPT = """
You are Evalia, an AI reasoning engine. Evaluate the following claim across five axes:
1. Logic (0-10)
2. Natural Law (0-10)
3. Historical Accuracy (0-10)
4. Source Credibility (0-10)
5. Overall Reasonableness (0-10)
Return results as compact JSON like: {"logic": 7, "natural": 5, "historical": 6, "source": 4, "overall": 6}
"""

def score_claim(text):
    try:
        cleaned = sanitize_input(text)
        full_prompt = SCORING_PROMPT + f"\n\nClaim:\n{cleaned}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_prompt}]
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        return {}

def generate_pdf_report(entry):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Evalia Claim Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(0, 10, f"Timestamp: {entry['timestamp']}", ln=True)
    pdf.multi_cell(0, 10, f"Claim: {entry['claim']}")
    if entry["url"]:
        pdf.multi_cell(0, 10, f"URL: {entry['url']}")
    if entry["image_text"]:
        pdf.multi_cell(0, 10, f"Extracted Image Text: {entry['image_text']}")
    pdf.ln(5)
    pdf.cell(0, 10, "Scores:", ln=True)
    for k, v in entry["scores"].items():
        pdf.cell(0, 10, f"  {k.capitalize()}: {v}/10", ln=True)

    pdf_file = f"evalia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_file)
    return pdf_file

initialize_memory()
st.set_page_config(page_title="Evalia - Claim Evaluator", layout="wide")
st.title("🔍 Evalia – Evaluate with Confidence")
st.markdown("Assess claims, URLs, and images using GPT-4o. Forensic score breakdown. Dev-mode ON.")

col1, col2 = st.columns(2)
with col1:
    claim_input = st.text_area("📌 Enter a claim for evaluation", height=200, placeholder="e.g., For such an entity to come anywhere near our planet is probably very bad news...")
    url_input = st.text_input("🔗 Optional URL to analyze", placeholder="Insert URL here")
with col2:
    image_input = st.file_uploader("🖼️ Upload image for OCR (optional)", type=["png", "jpg", "jpeg"])

cross_eval = st.checkbox("🔁 Evaluate image and claim as a combined context")

download_ready = False
pdf_generated = None

if st.button("Run Evaluation"):
    analysis_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "claim": claim_input,
        "url": url_input,
        "image_text": None,
        "scores": None
    }

    text_blob = claim_input
    if url_input:
        with st.spinner("Fetching URL..."):
            url_text = fetch_url_text(url_input)
            st.text_area("🔎 Extracted URL text", url_text[:1000], height=100)
            text_blob += "\n" + url_text

    if image_input:
        st.image(image_input, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Extracting text from image..."):
            img_text = analyze_image(image_input)
            st.subheader("📝 Extracted Text")
            st.info(img_text)
            analysis_log["image_text"] = img_text
            if cross_eval:
                text_blob += "\n" + img_text

    if text_blob.strip():
        with st.spinner("Scoring text..."):
            style, style_confidence = detect_style_gpt(text_blob)
            style_guidance = feedback_style_instruction(style)
            scores = score_claim(text_blob)
            analysis_log["scores"] = scores
            analysis_log["style"] = style

        if scores:
            st.subheader("📊 Evaluation Breakdown")
            categories = ["logic", "natural", "historical", "source", "overall"]
            scores_data = {k: scores.get(k, 0) for k in categories}
            df = pd.DataFrame({"Category": [k.capitalize() for k in categories], "Score": [scores.get(k, 0) for k in categories]})
            fig = px.bar(df, x='Score', y='Category', orientation='h', title='Score Overview', range_x=[0, 10], color='Score', color_continuous_scale='blues')
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No scores returned. Try again or rephrase.")

    if analysis_log["scores"] or analysis_log["image_text"]:
        save_to_memory(analysis_log)
        st.success("✅ Analysis saved to memory.")
        pdf_generated = generate_pdf_report(analysis_log)
        download_ready = True

if download_ready and pdf_generated:
    with open(pdf_generated, "rb") as f:
        st.download_button(
            label="📄 Download PDF Report",
            data=f,
            file_name=pdf_generated,
            mime="application/pdf"
        )

st.markdown("---")
st.caption("Evalia © 2025 – AI-Powered Reasoning Engine (Dev Mode + PDF Export)")