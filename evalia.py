# evalia.py — Streamlit app with Guided Quest UI + fixed logger duplication
import openai
import os
import json
import logging
import logging.handlers
import re
import io
import base64
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import requests
from fpdf import FPDF
import pandas as pd
import plotly.express as px
import unicodedata

# ----------------------- Logging -----------------------
log_file = "evalia_debug.log"
logger = logging.getLogger("EvaliaLogger")

if not logger.handlers:  # Prevent duplicate handlers on Streamlit rerun
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s - [File: %(pathname)s, Line: %(lineno)d] - %(exc_info)s'
    )

    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if os.getenv("ENV") != "STREAMLIT_CLOUD":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False

# ----------------------- API -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set in environment.")
    st.error("OPENAI_API_KEY not set. Please configure it in Streamlit Cloud.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)
MEMORY_FILE = "evalia_memory.json"

# ----------------------- Helpers -----------------------
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
        claim_preview = entry.get("claim", "") or entry.get("type", "entry")
        if isinstance(claim_preview, str) and len(claim_preview) > 50:
            claim_preview = claim_preview[:50] + "..."
        logger.info("Saved to memory: %s", claim_preview)
    except Exception:
        logger.error("Failed to save to memory", exc_info=True)

def sanitize_input(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^\w\s\.,!?:/\-\(\)\[\]]', '', text)
    return " ".join(text.strip().split())

# ----------------------- Fetchers -----------------------
def analyze_image(img_file):
    try:
        image_bytes = img_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        prompt = """
        Analyze the provided image for misinformation detection:
        1) Extract all legible text verbatim.
        2) Describe content/style/visual elements.
        3) Assess meme/AI/manipulation likelihood and flag telltales.
        Return JSON: {"extracted_text": "...", "description": "...", "assessment": "..."}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise image analysis assistant using GPT-4 Vision."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
        raw_content = response.choices[0].message.content
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            return {
                "extracted_text": sanitize_input(raw_content),
                "description": "Error parsing description.",
                "assessment": "Error in assessment."
            }
    except Exception:
        logger.error("Image analysis error", exc_info=True)
        return {"extracted_text": "Error extracting text.", "description": "", "assessment": ""}

def fetch_url_text(url):
    try:
        r = requests.get(url, timeout=10)
        return sanitize_input(r.text[:3000])
    except Exception as e:
        logger.error("URL fetch error for %s", url, exc_info=True)
        return f"Error fetching URL: {str(e)}"

# ----------------------- Prompts & Scoring -----------------------
STOIC_SCORING_PROMPT = """..."""  # (keep full prompt text)
BRUTAL_SCORING_PROMPT = """..."""

def score_claim(text, brutality_mode=False):
    try:
        cleaned = sanitize_input(text)
        selected_prompt = BRUTAL_SCORING_PROMPT if brutality_mode else STOIC_SCORING_PROMPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": selected_prompt},
                {"role": "user", "content": f"Claim:\n{cleaned}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.error("Scoring error", exc_info=True)
        return "Error: Unable to score claim."

# ----------------------- UI Text Helpers -----------------------
def spicy_tldr(analysis_text: str) -> str:
    v = re.search(r"^-\s*🔥\s*Verdict:\s*(.+)$", analysis_text, re.MULTILINE)
    s = re.search(r"^-\s*🔑\s*Claim Summary:\s*(.+)$", analysis_text, re.MULTILINE)
    verdict = v.group(1).strip() if v else "Result"
    summary = s.group(1).strip() if s else analysis_text.strip().split("\n", 1)[0][:240]
    return (f"{verdict}: {summary}")[:280]

def extract_reasoning(text, category):
    m = re.search(rf"{category}:\s*(.+?)(?=\n\S+:|\Z)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None

def extract_change_hint(text):
    m = re.search(r"(Suggested Further Research|Further Research)[^\n]*\n(.+)", text, re.IGNORECASE)
    return m.group(2).strip() if m else None

# ----------------------- PDF & Report -----------------------
def sanitize_for_pdf(text):
    # replace symbols & normalize
    ...
def generate_pdf_report(entry):
    ...
# ----------------------- App UI -----------------------
initialize_memory()
st.set_page_config(page_title="Evalia - Claim Evaluator", layout="wide")
# (keep your background CSS)

# Persona toggle
brutality_mode = st.checkbox("⚔️ Enable Brutality Mode")
# Inputs (claim, url, image)
...
# Run Evaluation button
...
# Tabs: Verdict Quest Flow, Evidence, Export
...
# Verdict tab content (quest stages)
...
# Evidence tab
...
# Export tab
...
# Footer
...
