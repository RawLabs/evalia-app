# evalia.py (Full Evalia Logic Restored for Streamlit Cloud)

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

# Ensure memory exists
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

# Clean up input
def sanitize_input(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^\w\s\.,!?]', '', text)
    return " ".join(text.strip().split())

# Scoring visualization
def bar_chart(score):
    full = round(score)
    empty = 10 - full
    return "[" + "█" * full + "░" * empty + "]"

# GPT-4o image-to-text
def analyze_image(img_file):
    try:
        image_bytes = img_file.read()
        image = Image.open(io.BytesIO(image_bytes))
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
        text = sanitize_input(response.choices[0].message.content)
        return text if text else "No text extracted."
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return f"Error extracting text: {str(e)}"

# Claim scoring
SCORING_PROMPT = """
You are Evalia, an AI reasoning engine. Evaluate the following claim across five axes:
1. Logic (0-10)
2. Natural Law (0-10)
3. Historical Accuracy (0-10)
4. Source Credibility (0-10)
5. Overall Reasonableness (0-10)
Return results as compact JSON like: {"logic": 7, "natural": 5, ...}
"""

def score_claim(text):
    try:
        cleaned = sanitize_input(text)
        full_prompt = SCORING_PROMPT + f"\n\nClaim: {cleaned}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_prompt}]
        )
        raw = response.choices[0].message.content
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        return {}

# Streamlit interface
initialize_memory()
st.set_page_config(page_title="Evalia - Claim Evaluator", layout="centered")
st.title("🔍 Evalia – Evaluate with Confidence")
st.markdown("Assess claims and extract text from images using GPT-4o. Full forensic score breakdown.")

claim_input = st.text_area("📌 Enter a claim for evaluation", height=150)
image_input = st.file_uploader("🖼️ Upload image for OCR (optional)", type=["png", "jpg", "jpeg"])

if st.button("Run Evaluation"):
    analysis_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "claim": claim_input,
        "image_text": None,
        "scores": None
    }

    if claim_input:
        with st.spinner("Scoring claim..."):
            scores = score_claim(claim_input)
        analysis_log["scores"] = scores

        if scores:
            st.subheader("📊 Evaluation Breakdown")
            for category in ["logic", "natural", "historical", "source", "overall"]:
                label = category.capitalize() if category != "natural" else "Natural Law"
                score = scores.get(category, 0)
                st.write(f"**{label}**: {bar_chart(score)} {score}/10")
        else:
            st.warning("No scores returned. Try again or rephrase.")

    if image_input:
        st.image(image_input, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Extracting text from image..."):
            img_text = analyze_image(image_input)
        st.subheader("📝 Extracted Text")
        st.info(img_text)
        analysis_log["image_text"] = img_text

    if analysis_log["scores"] or analysis_log["image_text"]:
        save_to_memory(analysis_log)
        st.success("✅ Analysis saved to memory.")

st.markdown("---")
st.caption("Evalia © 2025 – AI-Powered Reasoning Engine")
