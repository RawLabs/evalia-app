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

SCORING_PROMPT = """
You are Evalia, an AI reasoning engine. Evaluate the following claim and provide a detailed analysis:
- 🔥 Verdict: Plausible / Implausible / Speculative / Unknown / Proven
- 📊 Bar-style Score Overview: Logic (0-10), Natural Law (0-10), Historical Accuracy (0-10), Source Credibility (0-10), Overall Reasonableness (0-10)
- 🌺 Grounding Meter: Unverified ←─█ to ░─→ Fact (describe position)
- 🧠 Emotion Meter: Neutral ←─█ to ░─→ Charged (describe intensity)
- 🤖 AI Origin: Human ←─█ to ░─→ AI (assess likelihood)
- 📝 Detected Style: Symbolic/metaphysical, Conspiratorial, Academic, Emotional/personal, Basic/blunt, Exploratory (with confidence 0.0-1.0)
- 🧪 Reasoning per category: Brief explanation for each score
- 📚 Relevant Sources & Background: 1-2 reputable sources
- 📌 Suggested Further Research: Guidance on next steps
- 🧽 Final Commentary: Human, persuasive, encouraging self-verification
- 📾 Confidence Level: Percentage with rationale
- 🎯 Truth Drift Score: Grounded / Speculative / Detached
- 📊 Claim Length: Word count
- ⏳ Temporal Reference: Recent/timeless/historical/future-focused
Return the response as well-structured Markdown.
"""

def score_claim(text):
    try:
        cleaned = sanitize_input(text)
        full_prompt = SCORING_PROMPT + f"\n\nClaim:\n{cleaned}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        return "Error: Unable to score claim due to an issue."

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
st.markdown("Evaluate claims, images, or videos for plausibility, credibility, and AI origin. Paste a claim or upload media to get started.")

col1, col2 = st.columns(2)
with col1:
    claim_input = st.text_area("Paste your claim here:", placeholder="e.g., Ever since 5G towers went up, I’ve seen fewer bees...", height=200)
    url_input = st.text_input("Enter source URL (optional):", placeholder="Insert URL here")
with col2:
    image_file = st.file_uploader("Upload an image or meme (optional)", type=["png", "jpg", "jpeg"])
    video_file = st.file_uploader("Upload a video (optional)", type=["mp4", "mov", "mpeg4"])

download_ready = False
pdf_generated = None

if st.button("Run Evaluation"):
    analysis_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "claim": claim_input,
        "url": url_input,
        "image_text": None,
        "scores": {}
    }

    text_blob = claim_input
    if url_input:
        with st.spinner("Fetching URL..."):
            url_text = fetch_url_text(url_input)
            st.text_area("🔎 Extracted URL text", url_text[:1000], height=100)
            text_blob += "\n" + url_text

    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Extracting text from image..."):
            img_text = analyze_image(image_file)
            st.subheader("📝 Extracted Text")
            st.info(img_text)
            analysis_log["image_text"] = img_text
            text_blob += "\n" + img_text

    if text_blob.strip():
        with st.spinner("Scoring text..."):
            result = score_claim(text_blob)
            st.markdown(result)
            # Parse scores for chart (simplified parsing, adjust based on actual response format)
            categories = ["Logic", "Natural Law", "Historical Accuracy", "Source Credibility", "Overall Reasonableness"]
            scores = {}
            for cat in categories:
                match = re.search(rf"{cat}: (\d+)/10", result)
                if match:
                    scores[cat.lower()] = int(match.group(1))
            analysis_log["scores"] = scores

        if scores:
            df = pd.DataFrame({"Category": categories, "Score": [scores.get(cat.lower(), 0) for cat in categories]})
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