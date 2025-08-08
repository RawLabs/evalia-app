# evalia.py — Streamlit app with Guided Quest UI
import openai
import os
import json
import logging
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
import logging.handlers
import unicodedata  # For PDF Unicode sanitization

# ----------------------- Logging (fixed for Streamlit reruns) -----------------------
def _configure_evalia_logger():
    """
    Make logging idempotent under Streamlit's rerun model:
    - Clear any pre-existing handlers on our named logger.
    - Add a rotating file handler and (optionally) a console handler once.
    - Keep formatter simple; rely on exc_info=True when needed.
    """
    if st.session_state.get("_evalia_logger_configured"):
        return logging.getLogger("EvaliaLogger")

    logger = logging.getLogger("EvaliaLogger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't bubble to root (prevents duplicate prints)

    # Remove any handlers that might have been added by prior reruns
    for h in list(logger.handlers):
        logger.removeHandler(h)

    log_file = "evalia_debug.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setLevel(logging.DEBUG)

    # NOTE: no %(exc_info)s in the format — use logger.error(..., exc_info=True) when desired
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s - [File: %(pathname)s, Line: %(lineno)d]"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler only when not on Streamlit Cloud
    if os.getenv("ENV") != "STREAMLIT_CLOUD":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Tame noisy third‑party loggers a bit
    for noisy in ("urllib3", "httpx", "PIL", "requests"):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass

    st.session_state["_evalia_logger_configured"] = True
    logger.debug("Evalia logger configured.")
    return logger

logger = _configure_evalia_logger()

# ----------------------- API -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set in environment. Please configure it in Streamlit Cloud.")
    st.error("OPENAI_API_KEY not set in environment. Please configure it in Streamlit Cloud.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)
MEMORY_FILE = "evalia_memory.json"

# ----------------------- Helpers: Memory & Sanitization -----------------------
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
        # tolerate entries that aren't "claim" objects (feedback, etc.)
        claim_preview = entry.get("claim", "") or entry.get("type", "entry")
        claim_preview = (claim_preview[:50] + "...") if isinstance(claim_preview, str) and len(claim_preview) > 50 else claim_preview
        logger.info("Saved to memory: %s", claim_preview)
    except Exception:
        logger.error("Failed to save to memory", exc_info=True)

def sanitize_input(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)        # strip non-BMP
    text = re.sub(r'[^\w\s\.,!?:/\-\(\)\[\]]', '', text)        # keep common punctuation/URLs
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
STOIC_SCORING_PROMPT = """
You are Evalia, disciplined and precise. Clinical tone; no fluff.
Return a Markdown analysis in this exact structure:
- 🔥 Verdict: <Plausible|Implausible|Speculative|Unknown|Proven>
- 🔑 Claim Summary: ...
- 📊 Bar-style Score Overview:
  - Logic: ███░░░░░░░ 3/10
  - Natural Law: ██░░░░░░░░ 2/10
  - Historical Accuracy: ████░░░░░░ 4/10
  - Source Credibility: █░░░░░░░░░ 1/10
  - Overall Reasonableness: ███░░░░░░░ 3/10
- 🌺 Grounding Meter: ...
- 🧠 Emotion Meter: ...
- 🤖 AI Origin: ...
- 📝 Detected Style: ...
- 🧪 Reasoning per category:
  Logic: ...
  Natural Law: ...
  Historical Accuracy: ...
  Source Credibility: ...
  Overall Reasonableness: ...
- 📚 Relevant Sources & Background: [Title](https://...)
- 📌 Suggested Further Research: ...
- 🧽 Final Commentary: ...
- 📾 Confidence Level: ...
- 🎯 Truth Drift Score: ...
- 📊 Claim Length: ...
- ⏳ Temporal Reference: ...
Ensure exact 'Category: █... 3/10' formatting for parsing.
"""

BRUTAL_SCORING_PROMPT = """
You are Evalia — witty, biting, correct. Same structure as stoic; tone is sharp.
Return the Markdown in the exact structure listed in the stoic prompt (scores same format).
"""

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
        raw_response = response.choices[0].message.content.strip()
        logger.info(
            "Raw GPT response for claim '%s' (Brutal: %s): %s",
            cleaned[:50] + "..." if len(cleaned) > 50 else cleaned,
            brutality_mode,
            raw_response[:300]
        )
        return raw_response
    except Exception:
        logger.error("Scoring error", exc_info=True)
        return "Error: Unable to score claim due to an issue."

# ----------------------- UI Text Helpers -----------------------
def spicy_tldr(analysis_text: str) -> str:
    """One-liner hook from Verdict + Claim Summary."""
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

# ----------------------- PDF (kept simple) -----------------------
def sanitize_for_pdf(text):
    special_chars = {
        '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
        '⁻': '^-', '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4', '₅': '_5', '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9',
        '🔥': '[Fire]', '🔑': '[Key]', '📊': '[Chart]', '🌺': '[Flower]', '🧠': '[Brain]', '🤖': '[Robot]', '📝': '[Note]',
        '🧪': '[Test]', '📚': '[Book]', '📌': '[Pin]', '🧽': '[Sponge]', '📾': '[Envelope]', '🎯': '[Target]', '⏳': '[Hourglass]'
    }
    for char, repl in special_chars.items():
        text = text.replace(char, repl)
    text = unicodedata.normalize('NFKD', text).encode('latin1', 'replace').decode('latin1')
    return text.replace('?', '')

def generate_pdf_report(entry):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Evalia Claim Analysis Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(0, 10, f"Timestamp: {entry['timestamp']}", ln=True)
        sanitized_claim = sanitize_for_pdf(entry.get("claim",""))
        pdf.multi_cell(0, 10, f"Claim: {sanitized_claim}")
        if entry.get("url"):
            pdf.multi_cell(0, 10, f"URL: {entry['url']}")
        if entry.get("image_analysis"):
            pdf.multi_cell(0, 10, f"Image Analysis: {sanitize_for_pdf(str(entry['image_analysis']))}")
        pdf.ln(5)
        pdf.cell(0, 10, "Scores:", ln=True)
        for k, v in entry.get("scores", {}).items():
            pdf.cell(0, 10, f"  {k.capitalize()}: {v}/10", ln=True)
        if "analysis" in entry:
            sanitized_analysis = sanitize_for_pdf(entry["analysis"])
            pdf.ln(5)
            pdf.cell(0, 10, "Analysis:", ln=True)
            pdf.multi_cell(0, 10, sanitized_analysis)

        pdf_file = f"evalia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(pdf_file)
        logger.info("Generated PDF report: %s", pdf_file)
        return pdf_file
    except Exception:
        logger.error("Failed to generate PDF report", exc_info=True)
        return None

# ----------------------- App UI -----------------------
initialize_memory()
st.set_page_config(page_title="Evalia - Claim Evaluator", layout="wide")

# Background image / gradient
background_image = None
try:
    image_path = "raw-cast-enterprises-backdrop.png"
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        background_image = f"data:image/png;base64,{encoded_string}"
        logger.info("Background image loaded: %s", image_path)
except Exception:
    logger.error("Failed to load background image", exc_info=True)
if not background_image:
    background_image = "linear-gradient(to bottom, #1f1f1f, #2b2b2b)"

# Simple theme CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #EAEAEA;
    }}
    .stButton>button {{ border-radius: 10px; background:#8a1c1c; color:#fff; }}
    a {{ color:#7ec8ff; }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Evalia – Evaluate with Confidence")
st.caption("A rite-of-passage style evaluation. Hook → Gates → Artifacts → Missing Piece → Seal.")

# Persona toggle
brutality_mode = st.checkbox("⚔️ Enable Brutality Mode")

# Inputs
col1, col2 = st.columns(2)
with col1:
    claim_input = st.text_area("Paste your claim here:", placeholder="e.g., Ever since 5G towers went up...", height=180)
    url_input = st.text_input("Enter source URL (optional):", placeholder="https://...")
with col2:
    image_file = st.file_uploader("Upload an image or meme (optional)", type=["png", "jpg", "jpeg"])

download_ready = False
pdf_generated = None

# ----------------------- Run Evaluation -----------------------
if st.button("Cross the Threshold (Run Evaluation)", use_container_width=True):
    if not (claim_input.strip() or image_file or url_input):
        st.error("Please provide a claim, URL, or image to evaluate.")
    else:
        analysis_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claim": claim_input,
            "url": url_input,
            "image_analysis": None,
            "scores": {},
            "brutality_mode": brutality_mode
        }

        text_blob = claim_input
        url_text_display = None

        if url_input:
            with st.spinner("Fetching URL artifact..."):
                url_text = fetch_url_text(url_input)
                url_text_display = url_text[:1000]
                text_blob += "\n[URL Content]: " + url_text

        img_analysis = None
        if image_file:
            st.image(image_file, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Analyzing image artifact..."):
                img_analysis = analyze_image(image_file)
                analysis_log["image_analysis"] = img_analysis
                text_blob += (
                    f"\n[Image Extracted Text]: {img_analysis.get('extracted_text','')}"
                    f"\n[Image Description]: {img_analysis.get('description','')}"
                    f"\n[Image Assessment]: {img_analysis.get('assessment','')}"
                )

        result, scores = "", {}
        categories = ["Logic", "Natural Law", "Historical Accuracy", "Source Credibility", "Overall Reasonableness"]
        if text_blob.strip():
            with st.spinner("Passing through the Gates..."):
                result = score_claim(text_blob, brutality_mode=brutality_mode)
                for cat in categories:
                    match = re.search(rf"-\s*\**\s*{re.escape(cat)}\s*\**:\s*█+░+\s*(\d+)/10", result, re.IGNORECASE)
                    if match:
                        scores[cat.lower()] = int(match.group(1))
                analysis_log["scores"] = scores
                analysis_log["analysis"] = result
                logger.info("Parsed scores: %s", scores)

        # --------- TABBED OUTPUT (Quest Flow in Verdict) ----------
        tabs = st.tabs(["Verdict (Quest)", "Evidence", "Export"])

        # ===== QUEST FLOW VERDICT TAB =====
        with tabs[0]:
            if scores:
                persona_name = "Brutal" if brutality_mode else "Stoic"

                # Stage 1 – Hook
                st.markdown("## 🗝 Step into the Hall of Reason")
                st.caption(f"Persona: {persona_name}")
                verdict_score = scores.get("overall reasonableness", 0)
                st.metric("Overall Reasonableness", f"{verdict_score}/10")
                st.progress(verdict_score/10)
                if result:
                    st.markdown(f"**TL;DR:** {spicy_tldr(result)}")

                # Stage 2 – Gates of Reason
                st.markdown("### 🚪 Gates of Reason")
                gates = ["Logic", "Natural Law", "Historical Accuracy", "Source Credibility"]
                for gate in gates:
                    score_val = scores.get(gate.lower(), 0)
                    with st.expander(f"{gate} Gate — {score_val}/10", expanded=False):
                        snippet = extract_reasoning(result, gate)
                        st.write(snippet if snippet else "_No detail available._")

                # Stage 3 – Trial of Evidence (preview here too)
                st.markdown("### 📜 Trial of Evidence")
                if url_text_display:
                    st.subheader("Artifact: Extracted URL Text")
                    st.text_area("", url_text_display, height=150)
                if img_analysis:
                    st.subheader("Artifact: Image Analysis")
                    st.info(
                        f"Extracted Text: {img_analysis.get('extracted_text','')}\n\n"
                        f"Description: {img_analysis.get('description','')}\n\n"
                        f"Assessment: {img_analysis.get('assessment','')}"
                    )
                if not url_text_display and not img_analysis:
                    st.write("_No artifacts found in this quest._")

                # Stage 4 – The Missing Piece
                st.markdown("### 🔍 The Missing Piece")
                change_hint = extract_change_hint(result)
                st.write(change_hint if change_hint else "_No suggestions provided._")

                # Stage 5 – Seal of Passage
                st.markdown("## 🏆 Seal of Passage")
                verdict_line = spicy_tldr(result) if result else "Evaluation Complete"
                # Generate seal image
                img = Image.new("RGB", (500, 500), color=(28, 28, 28))
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
                except:
                    font = ImageFont.load_default()
                border_color = (200, 0, 0) if brutality_mode else (0, 100, 200)
                draw.ellipse((25, 25, 475, 475), outline=border_color, width=8)
                draw.text((250, 220), "Seal of Passage", font=font, fill="white", anchor="mm")
                draw.text((250, 270), verdict_line, font=font, fill="white", anchor="mm")
                seal_buf = io.BytesIO()
                img.save(seal_buf, format="PNG")
                st.image(seal_buf.getvalue(), caption="Your Seal of Passage", use_container_width=False)
                st.download_button(
                    "Download Seal as PNG",
                    data=seal_buf.getvalue(),
                    file_name="seal_of_passage.png",
                    mime="image/png"
                )

                with st.expander("Full analysis (for the record)"):
                    st.markdown(result if result else "_No analysis generated._")
            else:
                st.info("No scores parsed. Try a clearer claim or include a URL/image.")

        # ===== Evidence Tab =====
        with tabs[1]:
            if url_text_display:
                st.subheader("Extracted URL Text")
                st.text_area("", url_text_display, height=180)
            if img_analysis:
                st.subheader("Image Analysis")
                st.info(
                    f"Extracted Text: {img_analysis.get('extracted_text','')}\n\n"
                    f"Description: {img_analysis.get('description','')}\n\n"
                    f"Assessment: {img_analysis.get('assessment','')}"
                )
            if not url_text_display and not img_analysis:
                st.write("_No evidence inputs provided._")

        # ===== Export Tab =====
        with tabs[2]:
            if scores or img_analysis:
                save_to_memory(analysis_log)
                st.success("✅ Analysis saved to memory.")
                pdf_generated = generate_pdf_report(analysis_log)
                if pdf_generated and os.path.exists(pdf_generated):
                    with open(pdf_generated, "rb") as f:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=f,
                            file_name=pdf_generated,
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.warning("PDF generation failed.")
        # -----------------------------------------------------------

# Optional: refine loop
if st.button("Refine Claim"):
    st.text_area("Edit your claim:", value=claim_input, height=180, key="refine_claim")
    st.info("Update your claim and cross the threshold again.")

st.markdown("---")
st.caption("Evalia © 2025 – Raw Cast Labs | Guided evaluation: initiate → test → reveal → seal")
