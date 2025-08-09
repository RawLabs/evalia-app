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
    logger.propagate = False  # don't bubble to root (prevents duplicates)

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
You are Evalia, disciplined and precise. Your goal is to produce a comprehensive,
research-backed claim evaluation that is both deeply analytical and useful for further investigation.

Return a Markdown analysis in this exact structure and order:

- 🔥 Verdict: <Plausible|Implausible|Speculative|Unknown|Proven>
- 🔑 Claim Summary: (One sentence neutral summary of the claim)
- 📊 Bar-style Score Overview:
  - Logic: ███░░░░░░░ 3/10
  - Natural Law: ██░░░░░░░░ 2/10
  - Historical Accuracy: ████░░░░░░ 4/10
  - Source Credibility: █░░░░░░░░░ 1/10
  - Overall Reasonableness: ███░░░░░░░ 3/10
- 🌺 Grounding Meter: (Brief qualitative measure of how well-founded the claim is)
- 🧠 Emotion Meter: (Brief assessment of emotional vs. rational tone)
- 🤖 AI Origin: (If applicable)
- 📝 Detected Style: (E.g., formal, sensationalist, satirical, technical)
- 🧪 Reasoning per category:
  Logic:
    Provide 2–4 paragraphs of structured reasoning, clearly explaining the logical strengths and weaknesses.
    Use examples, analogies, or known logical fallacies if applicable.
  Natural Law:
    Provide 2–4 paragraphs detailing how the claim aligns or conflicts with known scientific or economic principles.
  Historical Accuracy:
    Provide 2–4 paragraphs comparing the claim to documented historical events or timelines.
  Source Credibility:
    Provide 2–4 paragraphs assessing the reliability of the sources behind the claim, citing specifics.
  Overall Reasonableness:
    Provide a synthesis judgment — weigh logic, evidence, and plausibility.
- 📚 Relevant Sources & Background:
    Provide 3–6 clickable markdown links to credible, primary, or authoritative sources (gov, edu, peer-reviewed research, or reputable investigative journalism — avoid Snopes/FactCheck-style).
    Each link should have a short annotation on why it’s relevant.
- 📌 Suggested Further Research:
    2–3 suggestions for next steps in investigating or verifying the claim.
- 🧽 Final Commentary:
    Concise wrap-up for the reader.
- 📾 Confidence Level: (0–100%)
- 🎯 Truth Drift Score: (0–100, higher means further from likely truth)
- 📊 Claim Length: (Word count)
- ⏳ Temporal Reference: (Time period referred to in the claim)

Ensure exact 'Category: █... 3/10' formatting for parsing.
"""

BRUTAL_SCORING_PROMPT = """
You are Evalia — sharp-tongued, witty, and brutally honest. Same structure and detail requirements as the stoic prompt above,
but with biting commentary where warranted. Still provide the same depth, research links, and structured paragraphs.
Do not water down criticism. Maintain accuracy.
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

def extract_reasoning_map(text: str) -> dict:
    """
    Parse ONLY the 'Reasoning per category:' section into a dict:
      { 'Logic': '...', 'Natural Law': '...', 'Historical Accuracy': '...', 'Source Credibility': '...', 'Overall Reasonableness': '...' }
    - Captures multi-line paragraphs until the next category OR the next major section.
    - If the section is missing, returns {}.
    """
    # 1) Find the 'Reasoning per category' anchor (emoji optional, bullet optional)
    anchor = re.search(
        r"^\s*[-•]?\s*🧪?\s*Reasoning per category\s*:\s*$",
        text, re.IGNORECASE | re.MULTILINE
    )
    if not anchor:
        return {}

    # 2) Slice after the anchor line
    sub = text[anchor.end():]

    # 3) Where a reasoning block should stop:
    SECTION_BREAK = (
        r"(?:^\s*(?:Logic|Natural\s+Law|Historical\s+Accuracy|Source\s+Credibility|Overall\s+Reasonableness)\s*:|"
        r"^\s*[-•]?\s*(?:📚\s*Relevant|📌\s*Suggested|🧽\s*Final|📾\s*Confidence|🎯\s*Truth|📊\s*Claim|⏳\s*Temporal|🔥\s*Verdict|🔑\s*Claim|📊\s*Bar)|\Z)"
    )

    cat_regex = re.compile(
        rf"^\s*(Logic|Natural\s+Law|Historical\s+Accuracy|Source\s+Credibility|Overall\s+Reasonableness)\s*:\s*(.+?)(?={SECTION_BREAK})",
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    out = {}
    for m in cat_regex.finditer(sub):
        key_raw = m.group(1).strip().lower()
        key_map = {
            "logic": "Logic",
            "natural law": "Natural Law",
            "historical accuracy": "Historical Accuracy",
            "source credibility": "Source Credibility",
            "overall reasonableness": "Overall Reasonableness",
        }
        key = key_map.get(key_raw, m.group(1).strip())
        val = re.sub(r"\n{2,}", "\n\n", m.group(2).strip())
        out[key] = val

    return out

# ----------------------- Seal Renderer (Evalia-branded) -----------------------
def _text_width(draw, text, font):
    """Measure text width using textbbox for broad Pillow compatibility."""
    if not text:
        return 0
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0

def _wrap_text_bbox(draw, text, font, max_width_px):
    """Word-wrap text to fit max width using textbbox (no textlength dependency)."""
    words = text.split()
    if not words:
        return ""
    lines, line = [], words[0]
    for w in words[1:]:
        candidate = f"{line} {w}"
        if _text_width(draw, candidate, font) <= max_width_px:
            line = candidate
        else:
            lines.append(line)
            line = w
    lines.append(line)
    return "\n".join(lines)

def render_evalia_seal(verdict_text: str, brutality_mode: bool, logo_path: str = "Evalia Logo Silver.png") -> bytes:
    """
    Builds an 800x800 PNG 'Seal of Passage' using the Evalia logo (if present).
    - verdict_text: short TL;DR or verdict line
    - brutality_mode: ring color changes with persona
    - logo_path: PNG with transparent background preferred
    Returns PNG bytes.
    """
    import io
    from PIL import Image, ImageDraw, ImageFont

    W, H = 800, 800
    img = Image.new("RGBA", (W, H), color=(24, 24, 24, 255))
    draw = ImageDraw.Draw(img)

    # Subtle vignette
    vignette = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    vdraw = ImageDraw.Draw(vignette)
    vdraw.ellipse((40, 40, W - 40, H - 40), fill=(0, 0, 0, 0))
    vdraw.rectangle((0, 0, W, H), outline=None, fill=(0, 0, 0, 120))
    img = Image.alpha_composite(img, vignette)
    draw = ImageDraw.Draw(img)

    # Colors
    ring = (200, 40, 40, 255) if brutality_mode else (60, 120, 220, 255)
    accent_dark = (80, 80, 80, 255)
    text_color = (235, 235, 235, 255)
    accent_text = (190, 190, 190, 255)

    # Outer rings
    draw.ellipse((60, 60, W - 60, H - 60), outline=ring, width=14)
    draw.ellipse((82, 82, W - 82, H - 82), outline=accent_dark, width=3)

    # Try to place the Evalia logo
    try:
        if os.path.exists(logo_path):
            logo = Image.open(logo_path).convert("RGBA")
            max_logo_w, max_logo_h = 440, 200
            scale = min(max_logo_w / logo.width, max_logo_h / logo.height, 1.0)
            logo = logo.resize((int(logo.width * scale), int(logo.height * scale)))
            lx = (W - logo.width) // 2
            ly = 140
            img.alpha_composite(logo, (lx, ly))
    except Exception:
        pass  # safe fail: render without logo

    # Fonts
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 50)
        verdict_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 38)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        title_font = ImageFont.load_default()
        verdict_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Title
    draw.text((W // 2, 75), "Seal of Passage", font=title_font, fill=text_color, anchor="mm")

    # Verdict text (wrapped)
    max_text_w = int(W * 0.72)
    wrapped = _wrap_text_bbox(draw, verdict_text or "Evaluation Complete", verdict_font, max_text_w)
    draw.multiline_text((W // 2, 390), wrapped, font=verdict_font, fill=text_color, anchor="mm", align="center", spacing=8)

    # Footer
    draw.text((W // 2, H - 80), "Evalia • Validation Through Inquiry", font=small_font, fill=accent_text, anchor="mm")

    # Export PNG bytes
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

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
if 'background_image' not in st.session_state:
    try:
        image_path = "raw-cast-enterprises-backdrop.png"
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            st.session_state.background_image = f"data:image/png;base64,{encoded_string}"
            logger.info("Background image loaded and cached: %s", image_path)
    except Exception:
        logger.error("Failed to load background image", exc_info=True)
    if 'background_image' not in st.session_state:
        st.session_state.background_image = "linear-gradient(to bottom, #1f1f1f, #2b2b2b)"

background_image = st.session_state.background_image

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
if st.button("Cross the Threshold (Run Evaluation)", key="eval_button", use_container_width=True):
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
            st.image(image_file, caption="Uploaded Image", use_column_width=True)
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
        reasoning_map = {}
        if text_blob.strip():
            with st.spinner("Passing through the Gates..."):
                result = score_claim(text_blob, brutality_mode=brutality_mode)
                # Parse numeric scores from the Bar section
                for cat in categories:
                    match = re.search(rf"-\s*\**\s*{re.escape(cat)}\s*\**:\s*█+░+\s*(\d+)/10", result, re.IGNORECASE)
                    if match:
                        scores[cat.lower()] = int(match.group(1))
                # Parse reasoning paragraphs only from the Reasoning section
                reasoning_map = extract_reasoning_map(result)

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

                # Guard: if parsing failed or section missing, keep it empty dict
                _reasoning = reasoning_map or {}

                for gate in gates:
                    score_val = scores.get(gate.lower(), 0)
                    with st.expander(f"{gate} Gate — {score_val}/10", expanded=False):
                        snippet = _reasoning.get(gate, "")
                        st.write(snippet if snippet else "_No detail available for this gate._")

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
                change_hint = re.search(r"(?:Suggested Further Research|Further Research)[^\n]*\n(.+)", result, re.IGNORECASE)
                st.write(change_hint.group(1).strip() if change_hint else "_No suggestions provided._")

                # Stage 5 – Seal of Passage (Evalia-branded)
                st.markdown("## 🏆 Seal of Passage")
                verdict_line = spicy_tldr(result) if result else "Evaluation Complete"

                seal_png = render_evalia_seal(
                    verdict_text=verdict_line,
                    brutality_mode=brutality_mode,
                    logo_path="Evalia Logo Silver.png"  # ensure file exists in same folder
                )

                st.image(seal_png, caption="Evalia Seal of Passage", use_column_width=True)
                st.download_button(
                    "Download Seal as PNG",
                    data=seal_png,
                    file_name="evalia_seal_of_passage.png",
                    mime="image/png",
                    key="png_download"
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
                            use_container_width=True,
                            key="pdf_dl"
                        )
                else:
                    st.warning("PDF generation failed.")

# Optional: refine loop
if st.button("Refine Claim", key="refine_button"):
    st.text_area("Edit your claim:", value=claim_input, height=180, key="refine_claim")
    st.info("Update your claim and cross the threshold again.")

st.markdown("---")
st.caption("Evalia © 2025 – Raw Cast Labs | Guided evaluation: initiate → test → reveal → seal")