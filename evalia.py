# evalia.py (Streamlit Cloud Safe Version)

import openai
import os
import json
import logging
from datetime import datetime, timezone
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from scipy import fftpack
from statistics import stdev, mean
import re
import tempfile
import base64
import io
import requests
import pandas as pd
import plotly.express as px

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

# Environment Setup
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set.")
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
        image = Image.open(io.BytesIO(image_bytes))
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        prompt = "Extract any text from this image."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise text extractor."},
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

def evaluate_claim(text):
    try:
        text = sanitize_input(text)
        prompt = f"Assess this claim for logic, plausibility, and emotional tone: {text}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Claim analysis error: {e}")
        return f"Error analyzing claim: {str(e)}"

# Streamlit UI
initialize_memory()
st.set_page_config(page_title="Evalia", layout="wide")
st.title("🔍 Evalia - Evaluate with Confidence")

col1, col2 = st.columns(2)
with col1:
    claim_text = st.text_area("Paste your claim:", height=200)
with col2:
    image_file = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])

if st.button("Evaluate"):
    if claim_text:
        with st.spinner("Analyzing claim..."):
            result = evaluate_claim(claim_text)
        st.markdown("### Result")
        st.markdown(result)

    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing image..."):
            img_result = analyze_image(image_file)
        st.markdown("### Extracted Text from Image")
        st.markdown(img_result)

st.markdown("---")
st.markdown("Evalia © 2025 – AI-assisted forensic analysis")
