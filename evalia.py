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
import unicodedata  # For PDF Unicode sanitization

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
        prompt = """
        Analyze the provided image for misinformation detection:
        1. Extract all legible text verbatim.
        2. Describe the content, style, and visual elements (e.g., colors, subjects, text overlays).
        3. Assess if it's a meme (humorous/satirical intent), potential AI generation (e.g., unnatural artifacts), or manipulation (e.g., inconsistent lighting, edits). Flag misinformation markers (e.g., exaggerated claims).
        Return in JSON: {"extracted_text": "...", "description": "...", "assessment": "..."}
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
            result = json.loads(raw_content)
            return result
        except json.JSONDecodeError:
            return {"extracted_text": sanitize_input(raw_content), "description": "Error parsing description.", "assessment": "Error in assessment."}
    except Exception as e:
        logger.error("Image analysis error for file %s", img_file.name if img_file else "unknown", exc_info=True)
        return {"extracted_text": f"Error extracting text: {str(e)}", "description": "", "assessment": ""}

def fetch_url_text(url):
    try:
        r = requests.get(url, timeout=10)
        return sanitize_input(r.text[:3000])
    except Exception as e:
        logger.error("URL fetch error for %s", url, exc_info=True)
        return f"Error fetching URL: {str(e)}"

# Standard Stoic SCORING_PROMPT
STOIC_SCORING_PROMPT = """
You are Evalia, an AI agent of disciplined logic and unwavering linguistic precision. You speak with calm weight, not volume. You do not flatter. You do not react. You clarify, correct, and reveal contradiction as a natural force, not a personal attack.

You treat language as sacred. Words have meaning, and those who misuse them will be offered correction — not as punishment, but as restoration of sense.

Your tone is:

Controlled
Articulate
Dryly insightful
Measured like a scholar, purposeful like a judge

You do not motivate, inspire, or console. You do not escalate. You simply hold the line of truth, even if the world fails to thank you for it.

Use metaphor only when it reinforces clarity. You may use calm rhetorical pressure, never sarcasm.

When identifying contradictions, be clinical and direct.

When a user misuses scientific terms, misrepresents logic, or propagates false cause, correct them with neutral force. Example:

“You have made a categorical assertion with no qualifying data. That is not a conclusion. It is an opinion. If you’d like to test it, I’ll help

Evaluate the following claim and provide a detailed analysis in Markdown. STRICTLY follow this format without additional markdown (e.g., no bold or italics) unless specified. Use single newlines between sections unless specified otherwise:

- 🔥 Verdict: Plausible / Implausible / Speculative / Unknown / Proven
- 🔑 Claim Summary: A concise 1-2 sentence summary of the core claim(s), incorporating any provided image descriptions, video transcripts, or URL content.
- 📊 Bar-style Score Overview (use exactly: "Category: ███░░░░░░░ 3/10" format for each):
  - Logic: ███░░░░░░░ 3/10
  - Natural Law: ██░░░░░░░░ 2/10
  - Historical Accuracy: ████░░░░░░ 4/10
  - Source Credibility: █░░░░░░░░░ 1/10
  - Overall Reasonableness: ███░░░░░░░ 3/10
- 🌺 Grounding Meter: Unverified ←─███░░░░░░─→ Fact (describe position, e.g., Leaning Unverified)
- 🧠 Emotion Meter: Neutral ←─████░░░░░░─→ Charged (describe intensity)
- 🤖 AI Origin: Human ←─█████░░░░─→ AI (assess likelihood, especially for images/memes)
- 📝 Detected Style: e.g., Symbolic/metaphysical (with confidence 0.0-1.0)
- 🧪 Reasoning per category: Brief explanation for each
  Logic: Explanation
  Natural Law: Explanation
  Historical Accuracy: Explanation
  Source Credibility: Explanation
  Overall Reasonableness: Explanation
- 📚 Relevant Sources & Background: Provide 1-3 direct, primary or official URLs formatted as Markdown hyperlinks (e.g., [Overview of electromagnetic system efficiency](https://www.science.gov/art...)). Focus on raw data, original studies, or official records from sources like .gov or .edu.
- 📌 Suggested Further Research: Provide 1-2 specific, clickable search query links formatted as Markdown hyperlinks (e.g., [Search nature.com for electrostatic efficiency](https://www.google.com/search?q=site:nature.com+electrostatic+energy+efficiency)) and one actionable step (e.g., 'Examine original data on [topic] at [official site]' or 'Review primary documents from [source].')
- 🧽 Final Commentary: Deliver a measured, clinical statement pointing to the information (e.g., 'The claim presents a position. Here is access to primary data for examination. Proceed with precision if you seek clarity.')
- 📾 Confidence Level: Percentage with rationale
- 🎯 Truth Drift Score: Grounded / Speculative / Detached
- 📊 Claim Length: Word count
- ⏳ Temporal Reference: Recent/timeless/historical/future-focused

Ensure scores are in the exact "Category: ███░░░░░░░ 3/10" format for parsing.
"""

## Brutal Mode SCORING_PROMPT
BRUTAL_SCORING_PROMPT = """
You are Evalia — the Cockney Oracle. You wield logic like a crowbar, sarcasm like a scalpel, and you don’t just debunk — you entertain. You’ve had more arguments than hot dinners, and your tongue’s sharper than a pint glass in a pub brawl.

You’re not here to please — you’re here to point out the rot behind the wallpaper. You deliver truth wrapped in wit, dipped in vinegar.

Speak like:
- Guy Ritchie wrote you after three pints and a philosophy degree
- A street-level Stephen Fry with no patience for fluff
- A pub philosopher who’s read Popper, punched a flat-Earther, and still tips the barmaid well

Examples of tone:
- “That’s not logic. That’s a superstition in a lab coat.”
- “You brought a vibe to a truth fight. Brave. Pointless, but brave.”
- “Your argument’s held together with more gaps than a politician’s memory.”
- “Let’s untangle this nonsense like Christmas lights in a council flat.”

Now, evaluate the claim below with cutting insight. Be funny. Be biting. Be right.

Respond using the following Markdown structure (no extra markup):

- 🔥 Verdict: Plausible / Implausible / Speculative / Unknown / Proven
- 🔑 Claim Summary: One or two-sentence summary of the claim in your voice
- 📊 Bar-style Score Overview (use exactly: "Category: ███░░░░░░░ 3/10" format)
  - Logic: ███░░░░░░░ 3/10
  - Natural Law: ██░░░░░░░░ 2/10
  - Historical Accuracy: ████░░░░░░ 4/10
  - Source Credibility: █░░░░░░░░░ 1/10
  - Overall Reasonableness: ███░░░░░░░ 3/10
- 🌺 Grounding Meter: Unverified ←─███░░░░░░─→ Fact
- 🧠 Emotion Meter: Neutral ←─████░░░░░░─→ Charged
- 🤖 AI Origin: Human ←─█████░░░░─→ AI
- 📝 Detected Style: e.g., Rant disguised as insight (with confidence 0.0-1.0)
- 🧪 Reasoning per category:
  Logic: Mock bad logic and explain with analogies or metaphors that bite.
  Natural Law: Compare to absurdities. If it breaks physics, say so — colorfully.
  Historical Accuracy: Put history back in its boots, not fairy tales.
  Source Credibility: If it’s a blog, say so. If it’s peer-reviewed, nod with approval.
  Overall Reasonableness: Deliver the harsh truth, not a hug.
- 📚 Relevant Sources & Background: Use raw.gov or .edu data — or point out when it doesn’t exist.
- 📌 Suggested Further Research: Drop 1–2 search links and a cheeky nudge.
- 🧽 Final Commentary: Wrap with wit, edge, or a shrug that says "told you so."
- 📾 Confidence Level: Percentage and why you’d bet on it
- 🎯 Truth Drift Score: Grounded / Speculative / Detached
- 📊 Claim Length: Word count
- ⏳ Temporal Reference: Recent / timeless / etc.

Above all: **say what others won’t, how others can’t — but with charm.**
"""

def score_claim(text, brutality_mode=False):
    try:
        cleaned = sanitize_input(text)
        selected_prompt = BRUTAL_SCORING_PROMPT if brutality_mode else STOIC_SCORING_PROMPT
        full_prompt = selected_prompt + f"\nClaim:\n{cleaned}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": selected_prompt},
                {"role": "user", "content": f"Claim:\n{cleaned}"}
            ]
        )

        raw_response = response.choices[0].message.content.strip()
        logger.info("Raw GPT response for claim '%s' (Brutality Mode: %s): %s", cleaned[:50] + "..." if len(cleaned) > 50 else cleaned, brutality_mode, raw_response)
        return raw_response
    except Exception as e:
        logger.error("Scoring error for claim '%s'", text[:50] + "..." if len(text) > 50 else text, exc_info=True)
        return "Error: Unable to score claim due to an issue."

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
        sanitized_claim = sanitize_for_pdf(entry["claim"])
        pdf.multi_cell(0, 10, f"Claim: {sanitized_claim}")
        if entry["url"]:
            pdf.multi_cell(0, 10, f"URL: {entry['url']}")
        if entry["image_analysis"]:
            pdf.multi_cell(0, 10, f"Image Analysis: {sanitize_for_pdf(str(entry['image_analysis']))}")
        pdf.ln(5)
        pdf.cell(0, 10, "Scores:", ln=True)
        for k, v in entry["scores"].items():
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
    except Exception as e:
        logger.error("Failed to generate PDF report", exc_info=True)
        return None

initialize_memory()
st.set_page_config(page_title="Evalia - Claim Evaluator", layout="wide")

# Load background image
background_image = None
try:
    image_path = "raw-cast-enterprises-backdrop.png"
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        background_image = f"data:image/png;base64,{encoded_string}"
        if background_image:
            logger.info("Background image loaded successfully: %s", image_path)
    else:
        logger.warning("Background image not found at %s", image_path)
except Exception as e:
    logger.error("Failed to load background image", exc_info=True)
if not background_image:
    logger.warning("Falling back to gradient due to missing image.")
    background_image = "linear-gradient(to bottom, #A9A9A9, #FFFFFF)"

# Updated CSS for tighter spacing
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
        line-height: 1.4;
        margin-bottom: 10px;
        white-space: pre-wrap;
    }}
    .output-box p {{
        margin-bottom: 10px;
    }}
    .output-box ul, .output-box ol {{
        margin-bottom: 10px;
        list-style-type: disc;
    }}
    .output-box li {{
        margin-bottom: 6px;
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
    .video-coming-soon {{
        color: #FFA500;
        font-style: italic;
        text-align: center;
        margin-top: 10px;
    }}
    a {{
        color: #00B7EB;
        text-decoration: underline;
    }}
    a:hover {{
        color: #00D4FF;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Evalia – Evaluate with Confidence")
st.markdown("Evaluate claims, images, or videos for plausibility, credibility, and AI origin. Paste a claim or upload media to get started.")

# Add Brutality Mode toggle
brutality_mode = st.checkbox("⚔️ Enable Brutality Mode")

col1, col2 = st.columns(2)
with col1:
    claim_input = st.text_area("Paste your claim here:", placeholder="e.g., Ever since 5G towers went up...", height=200)
    url_input = st.text_input("Enter source URL (optional):", placeholder="Insert URL here")
with col2:
    image_file = st.file_uploader("Upload an image or meme (optional)", type=["png", "jpg", "jpeg"])

    # ---------- DROP-IN REPLACEMENT START ----------
download_ready = False
pdf_generated = None

if st.button("Run Evaluation", use_container_width=True):
    if not (claim_input.strip() or image_file or url_input):
        st.error("Please provide a claim, URL, or image to evaluate.")
    else:
        # Live status log
        with st.status("Working…", expanded=True) as status:
            st.write("Preparing inputs")
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
                st.write("Fetching URL content")
                url_text = fetch_url_text(url_input)
                url_text_display = url_text[:1500]
                text_blob += "\n[URL Content]: " + url_text

            img_analysis = None
            if image_file:
                st.write("Analyzing image")
                st.image(image_file, caption="Uploaded Image", use_container_width=True)
                img_analysis = analyze_image(image_file)
                analysis_log["image_analysis"] = img_analysis
                text_blob += (
                    f"\n[Image Extracted Text]: {img_analysis.get('extracted_text','')}"
                    f"\n[Image Description]: {img_analysis.get('description','')}"
                    f"\n[Image Assessment]: {img_analysis.get('assessment','')}"
                )

            result = ""
            scores = {}
            if text_blob.strip():
                st.write("Scoring claim")
                result = score_claim(text_blob, brutality_mode=brutality_mode)

                categories = ["Logic", "Natural Law", "Historical Accuracy", "Source Credibility", "Overall Reasonableness"]
                for cat in categories:
                    match = re.search(rf"-\s*\**\s*{re.escape(cat)}\s*\**:\s*█+░+\s*(\d+)/10", result, re.IGNORECASE | re.DOTALL)
                    if match:
                        scores[cat.lower()] = int(match.group(1))

            analysis_log["scores"] = scores
            analysis_log["analysis"] = result

            # Ready to render UI
            status.update(label="Done", state="complete", expanded=False)

        # ---- TABBED OUTPUT ----
        tabs = st.tabs(["Verdict", "Evidence", "Export"])

        with tabs[0]:
            # Small summary metric (if parsed)
            overall = scores.get("overall reasonableness".lower(), scores.get("overall reasonableness", scores.get("overall_reasonableness", None)))
            if overall is None:
                # Fallback: look for any key containing 'overall'
                overall = next((v for k,v in scores.items() if "overall" in k), None)
            if overall is not None:
                st.metric("Overall Reasonableness", f"{overall}/10")

            # Bar chart even with partial scores
            categories = ["Logic", "Natural Law", "Historical Accuracy", "Source Credibility", "Overall Reasonableness"]
            df = pd.DataFrame({
                "Category": categories,
                "Score": [scores.get(cat.lower(), 0) for cat in categories]
            })
            fig = px.bar(
                df, x='Score', y='Category',
                orientation='h', title='Score Overview', range_x=[0, 10],
                color='Score', color_continuous_scale='blues'
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Full analysis"):
                # Keep your exact model output formatting
                st.markdown(result if result else "_No analysis produced._")

        with tabs[1]:
            if url_input:
                st.subheader("URL extract (trimmed)")
                st.text_area("Extracted text", url_text_display or "", height=180)
                st.caption("Tip: include/exclude parts of this text, then re‑run.")

            if img_analysis:
                st.subheader("Image analysis")
                st.info(
                    f"Extracted Text: {img_analysis.get('extracted_text','')}\n\n"
                    f"Description: {img_analysis.get('description','')}\n\n"
                    f"Assessment: {img_analysis.get('assessment','')}"
                )

            if not (url_input or img_analysis):
                st.write("_No evidence inputs provided._")

        with tabs[2]:
            # Save memory & prepare PDF (exactly as before)
            if analysis_log["scores"] or analysis_log.get("image_analysis"):
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
            else:
                st.info("Nothing to export yet.")
# ---------- DROP-IN REPLACEMENT END ----------


# Add Refine Claim button
if st.button("Refine Claim"):
    st.text_area("Edit your claim:", value=claim_input, height=200, key="refine_claim")
    st.info("Update your claim based on the analysis and re-run the evaluation to dig deeper.")

st.markdown("---")
st.caption("Evalia © 2025 – AI-Powered Reasoning by Raw Cast Labs")