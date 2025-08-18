import streamlit as st
from datetime import datetime, timezone
from core.logging_config import configure_evalia_logger
from core.api_config import initialize_memory, OPENAI_API_KEY
from core.analysis import score_claim
from core.fetchers import fetch_url_text, analyze_image
from rendering.seal import render_evalia_seal
from core.claim_output.pdf_report import generate_pdf_report
from core.ui.ui_components import spicy_tldr, set_custom_css, display_verdict_tab, display_evidence_tab, display_export_tab

# Initialize
logger = configure_evalia_logger()
initialize_memory()
st.set_page_config(page_title="Evalia - Claim Evaluator", layout="wide")
set_custom_css()

# UI
st.title("⚡ Evalia - Evaluate with Confidence")
st.caption("A rite-of-passage style evaluation. Hook → Gates → Artifacts → Missing Piece → Seal.")
brutality_mode = st.checkbox("🔥 Enable Brutality Mode", help="Unleash ruthless analysis")
col1, col2 = st.columns(2)
with col1:
    claim_input = st.text_area("Paste your claim here:", placeholder="e.g., Ever since 5G towers went up...", height=180, help="Enter the claim you want to evaluate")
    url_input = st.text_input("Enter source URL (optional):", placeholder="https://...", help="Add a URL for additional context")
with col2:
    image_file = st.file_uploader("Upload an image or meme (optional)", type=["png", "jpg", "jpeg"], help="Upload visual content to analyze")

if st.button("⚡ Cross the Threshold (Run Evaluation)", key="eval_button", use_container_width=True):
    if not (claim_input.strip() or image_file or url_input):
        st.error("❌ Please provide a claim, URL, or image to evaluate.")
    else:
        with st.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🔍 Gathering artifacts...")
            progress_bar.progress(0.2)

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
                status_text.text("🌐 Fetching URL content...")
                progress_bar.progress(0.4)
                url_text_display = fetch_url_text(url_input)
                text_blob += f"\n[URL Content]: {url_text_display}"

            if image_file:
                st.image(image_file, caption="📷 Uploaded Image", use_column_width=True)
                status_text.text("🖼️ Analyzing image...")
                progress_bar.progress(0.6)
                analysis_log["image_analysis"] = analyze_image(image_file)
                text_blob += (
                    f"\n[Image Extracted Text]: {analysis_log['image_analysis'].get('extracted_text','')}"
                    f"\n[Image Description]: {analysis_log['image_analysis'].get('description','')}"
                    f"\n[Image Assessment]: {analysis_log['image_analysis'].get('assessment','')}"
                )

            if text_blob.strip():
                status_text.text("🧠 Processing through AI analysis...")
                progress_bar.progress(0.8)
                result = score_claim(text_blob, brutality_mode)
                analysis_log["scores"] = result.get("scores", {})
                analysis_log["analysis"] = result

            status_text.text("✅ Analysis complete!")
            progress_bar.progress(1.0)

            import time
            time.sleep(1)
            progress_container = st.container()
            progress_container.empty()

            tabs = st.tabs(["🏆 Verdict (Quest)", "📋 Evidence", "📤 Export"])
            with tabs[0]:
                display_verdict_tab(result, analysis_log, url_text_display, brutality_mode)
            with tabs[1]:
                display_evidence_tab(url_text_display, analysis_log.get("image_analysis"))
            with tabs[2]:
                display_export_tab(analysis_log, generate_pdf_report, logger)

if st.button("Refine Claim", key="refine_button"):
    st.text_area("Edit your claim:", value=claim_input, height=180, key="refine_claim")
    st.info("Update your claim and cross the threshold again.")

st.markdown("---")
st.caption("Evalia © 2025 – Raw Cast Enterprises | Guided evaluation: initiate → test → reveal → seal")