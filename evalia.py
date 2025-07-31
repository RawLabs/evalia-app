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
import logging.handlers

# Configure robust logging
log_file = "evalia_debug.log"
logger = logging.getLogger("EvaliaLogger")
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - [File: %(pathname)s, Line: %(lineno)d] - %(exc_info)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
if os.getenv("ENV") != "STREAMLIT_CLOUD":
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set in environment. Please configure it in Streamlit Cloud.")
    st.error("OPENAI_API_KEY not set in environment. Please configure it in Streamlit Cloud.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)
MEMORY_FILE = "evalia_memory.json"

def initialize_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w') as f:
            json.dump([], f)
        logger.info("Initialized memory file: %s", MEMORY_FILE)

def save_to_memory(entry):
    try:
        with open(MEMORY_FILE, 'r+') as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
        logger.info("Saved to memory: %s", entry["claim"][:50] + "..." if len(entry["claim"]) > 50 else entry["claim"])
    except Exception as e:
        logger.error("Failed to save to memory", exc_info=True)

def sanitize_input(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^\w\s\.,!?]', '', text)
    return " ".join(text.strip())

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
        logger.error("Image analysis error for file %s", img_file.name if img_file else "unknown", exc_info=True)
        return f"Error extracting text: {str(e)}"

def fetch_url_text(url):
    try:
        r = requests.get(url, timeout=10)
        return sanitize_input(r.text[:3000])
    except Exception as e:
        logger.error("URL fetch error for %s", url, exc_info=True)
        return f"Error fetching URL: {str(e)}"

SCORING_PROMPT = """
You are Evalia, an AI reasoning engine. Evaluate the following claim and provide a detailed analysis in Markdown:
- 🔥 Verdict: Plausible / Implausible / Speculative / Unknown / Proven
- 📊 Bar-style Score Overview (use exactly: Category: ███░░░░░░░ 3/10 format for each):
  - Logic: ███░░░░░░░ 3/10
  - Natural Law: ██░░░░░░░░ 2/10
  - Historical Accuracy: ████░░░░░░ 4/10
  - Source Credibility: █░░░░░░░░░ 1/10
  - Overall Reasonableness: ███░░░░░░░ 3/10
- 🌺 Grounding Meter: Unverified ←─███░░░░░░─→ Fact (describe position, e.g., Leaning Unverified)
- 🧠 Emotion Meter: Neutral ←─████░░░░░░─→ Charged (describe intensity)
- 🤖 AI Origin: Human ←─█████░░░░─→ AI (assess likelihood)
- 📝 Detected Style: e.g., Symbolic/metaphysical (with confidence 0.0-1.0)
- 🧪 Reasoning per category: Brief explanation for each
- 📚 Relevant Sources & Background: 1-2 reputable sources
- 📌 Suggested Further Research: Guidance on next steps
- 🧽 Final Commentary: Human-like, persuasive, encouraging self-verification
- 📾 Confidence Level: Percentage with rationale
- 🎯 Truth Drift Score: Grounded / Speculative / Detached
- 📊 Claim Length: Word count
- ⏳ Temporal Reference: Recent/timeless/historical/future-focused
Ensure scores are in the exact "Category: ███░░░░░░░ 3/10" format for parsing.
"""

def score_claim(text):
    try:
        cleaned = sanitize_input(text)
        full_prompt = SCORING_PROMPT + f"\n\nClaim:\n{cleaned}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_prompt}]
        )
        raw_response = response.choices[0].message.content.strip()
        logger.info("Raw GPT response for claim '%s': %s", cleaned[:50] + "..." if len(cleaned) > 50 else cleaned, raw_response)
        return raw_response
    except Exception as e:
        logger.error("Scoring error for claim '%s'", text[:50] + "..." if len(text) > 50 else text, exc_info=True)
        return "Error: Unable to score claim due to an issue."

def generate_pdf_report(entry):
    try:
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
        logger.info("Generated PDF report: %s", pdf_file)
        return pdf_file
    except Exception as e:
        logger.error("Failed to generate PDF report", exc_info=True)
        return None

initialize_memory()
st.set_page_config(page_title="Evalia - Claim Evaluator", layout="wide")

# Load background image
background_image = None
try:
    image_path = "68887836-metal-background-texture-of-titanium-sheet-of-metal-surface-steel.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        background_image = f"data:image/jpeg;base64,{encoded_string}"
        logger.info("Background image loaded successfully: %s", image_path)
    else:
        logger.warning("Background image not found at %s", image_path)
except Exception as e:
    logger.error("Failed to load background image", exc_info=True)
if not background_image:
    logger.warning("Falling back to gradient due to missing image.")
    background_image = "linear-gradient(to bottom, #A9A9A9, #FFFFFF)"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #FFFFFF;
    }}
    .output-box {{
        background-color: rgba(40, 40, 40, 0.8);
        color: #E0E0E0;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Roboto', sans-serif;
    }}
    .stTextArea, .stFileUploader, .stTextInput {{
        background-color: #282828;
        color: #E0E0E0;
        border: 1px solid #003087;
        border-radius: 8px;
    }}
    .stButton>button {{
        background-color: #003087;
        color: white;
        border-radius: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Evalia – Evaluate with Confidence")
st.markdown("Evaluate claims, images, or videos for plausibility, credibility, and AI origin. Paste a claim or upload media to get started.")

col1, col2 = st.columns(2)
with col1:
    claim_input = st.text_area("Paste your claim here:", placeholder="e.g., Ever since 5G towers went up...", height=200)
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
            st.markdown(f'<div class="output-box">{result}</div>', unsafe_allow_html=True)
            categories = ["Logic", "Natural Law", "Historical Accuracy", "Source Credibility", "Overall Reasonableness"]
            scores = {}
            for cat in categories:
                match = re.search(rf"{re.escape(cat)}:.*?(\d+)/10", result, re.IGNORECASE | re.DOTALL)
                if match:
                    scores[cat.lower()] = int(match.group(1))
            analysis_log["scores"] = scores
            logger.info(f"Parsed scores for claim '{text_blob[:50]}...': {scores}")

        if scores:
            df = pd.DataFrame({"Category": categories, "Score": [scores.get(cat.lower(), 0) for cat in categories]})
            fig = px.bar(df, x='Score', y='Category', orientation='h', title='Score Overview', range_x=[0, 10], color='Score', color_continuous_scale='blues')
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            logger.warning("No scores parsed from response for claim '%s'", text_blob[:50] + "..." if len(text_blob) > 50 else text_blob)
            st.warning("No scores parsed from response. Check logs for details or rephrase.")

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