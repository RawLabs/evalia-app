#---Ui Components ---
import streamlit as st
import os  # Added import for os
from rendering.seal import render_evalia_seal
from core.analysis import save_to_memory

def spicy_tldr(analysis: dict) -> str:
    verdict = analysis.get("verdict", "Result")
    summary = analysis.get("claim_summary", "No summary available")
    return (f"{verdict}: {summary}")[:280]

def set_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(to bottom right, #4B4B4B 0%, #4B4B4B 66%, #2D2D2D 67%, #600000 100%);
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: #EAEAEA;
        }
        .stButton>button {
            border-radius: 12px;
            background: linear-gradient(135deg, #8B0000, #A11212);
            color: #FFFFFF;
            border: none;
            padding: 0.6rem 1rem;
            box-shadow: 0 8px 20px rgba(139,0,0,0.3);
            transition: all 0.2s ease;
            font-weight: 500;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #A11212, #C41E3A);
            box-shadow: 0 12px 28px rgba(139,0,0,0.5);
            transform: translateY(-2px);
        }
        div[data-testid="stTextInput"] > div,
        div[data-testid="stTextInput"] > div > div,
        div[data-testid="stTextArea"] > div,
        div[data-testid="stTextArea"] > div > div {
            background: rgba(45, 45, 45, 0.9) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(126,200,255,0.2) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
            transition: all 0.2s ease;
        }
        div[data-testid="stTextInput"] > div:hover,
        div[data-testid="stTextArea"] > div:hover {
            border-color: rgba(126,200,255,0.4) !important;
            transform: translateY(-1px);
        }
        .css-1emrehy .stProgress > div > div {
            background: linear-gradient(90deg, #4FC3F7, #039BE5) !important;
        }
        h1 {
            font-family: "Impact", "Arial Black", sans-serif;
            font-weight: 900;
            font-size: clamp(32px, 5vw, 60px);
            letter-spacing: -0.03em;
            text-align: center;
            background: linear-gradient(90deg, #B0BEC5 0%, #B71C1C 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            margin: 1rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(45, 45, 45, 0.8);
            border-radius: 10px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(126,200,255,0.1);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #8B0000, #A11212) !important;
        }
        .streamlit-expander {
            border: 1px solid rgba(126,200,255,0.2);
            border-radius: 8px;
            background: rgba(45, 45, 45, 0.9);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def display_verdict_tab(result, analysis_log, url_text_display, brutality_mode):
    if result and result.get("scores"):
        scores = result["scores"]
        avg_reasonableness = int(sum(scores.values()) / len(scores))
        confidence = avg_reasonableness * 10
        st.markdown(f"### Persona: {'Brutal' if brutality_mode else 'Stoic'}")
        st.markdown(f"**Overall Reasonableness:** {avg_reasonableness}/10")
        st.markdown(f"**Confidence Level:** {confidence}%")
        st.progress(float(scores.get("overall_reasonableness", 0)) / 10.0)
        st.info(f"**📋 Summary:** {spicy_tldr(result)}")

        st.markdown("### 🚪 Gates of Reason")
        gates = [
            ("🧠 Logic", "logic"),
            ("⚖️ Natural Law", "natural_law"),
            ("📚 Historical Accuracy", "historical_accuracy"),
            ("🔍 Source Credibility", "source_credibility"),
        ]
        for label, key in gates:
            val = scores.get(key, 0)
            color = "🟢" if val >= 7 else "🟡" if val >= 4 else "🔴"
            with st.expander(f"{color} {label} — {val}/10", expanded=True):  # Expanded by default for thoroughness
                st.markdown(result.get("reasoning", {}).get(key, "_No detailed analysis available._"))

        # New section for additional metrics
        st.markdown("### 📊 Additional Metrics")
        with st.expander("Grounding Meter", expanded=True):
            st.markdown(result.get("grounding_meter", "_No grounding available._"))
        with st.expander("Emotion Meter", expanded=True):
            st.markdown(result.get("emotion_meter", "_No emotion analysis available._"))
        with st.expander("AI Origin Likelihood", expanded=True):
            st.markdown(result.get("ai_origin", "_No AI origin analysis available._"))
        with st.expander("Detected Style", expanded=True):
            st.markdown(result.get("detected_style", "_No style detection available._"))

        st.markdown("### Trial of Evidence")
        if url_text_display:
            st.subheader("Artifact: Extracted URL Text")
            st.text_area("", url_text_display, height=150)
        if analysis_log.get("image_analysis"):
            st.subheader("Artifact: Image Analysis")
            st.info(
                f"Extracted Text: {analysis_log['image_analysis'].get('extracted_text','')}\n\n"
                f"Description: {analysis_log['image_analysis'].get('description','')}\n\n"
                f"Assessment: {analysis_log['image_analysis'].get('assessment','')}"
            )

        st.markdown("### The Missing Piece: Suggested Research")
        for point in result.get("suggested_research", []):
            st.markdown(f"- {point}")

        # New section for sources
        st.markdown("### 🔗 Relevant Sources")
        for source in result.get("relevant_sources", []):
            st.markdown(f"- [{source.get('annotation', 'No description')}]({source.get('url', '#')})")

        st.markdown("### Final Commentary")
        st.markdown(result.get("final_commentary", "_No commentary available._"))

        st.markdown("## Seal of Passage")
        verdict_line = spicy_tldr(result)
        seal_png = render_evalia_seal(verdict_line, brutality_mode, "static/Evalia Logo Silver.png")
        st.image(seal_png, caption="Evalia Seal of Passage", width=300)

        # Removed the full JSON expander to avoid redundancy
    else:
        st.warning("⚠️ Analysis failed or no valid scores generated. Please try a clearer claim or check logs for details.")

def display_evidence_tab(url_text_display, img_analysis):
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

def display_export_tab(analysis_log, generate_pdf_report, logger):
    if analysis_log.get("scores") or analysis_log.get("image_analysis"):
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