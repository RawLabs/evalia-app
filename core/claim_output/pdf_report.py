"""PDF report generation."""
from fpdf import FPDF
import re
import unicodedata
from core.logging_config import configure_evalia_logger

logger = configure_evalia_logger()

def sanitize_for_pdf(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1 - \2", text)
    text = (text.replace('—', '-').replace('–', '-')
                .replace('“', '"').replace('”', '"').replace('’', "'"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = unicodedata.normalize('NFKD', text).encode('latin1', 'replace').decode('latin1')
    return text.replace('?', '')

def _pdf_h1(pdf, text):
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, txt=text, ln=True)
    pdf.ln(2)

def _pdf_h2(pdf, text):
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, txt=text, ln=True)

def _pdf_p(pdf, text):
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, txt=text)
    pdf.ln(1)

def _pdf_kv(pdf, k, v):
    pdf.set_font("Arial", "B", 11)
    pdf.cell(45, 6, txt=f"{k}:", ln=0)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, txt=str(v))

def _pdf_divider(pdf):
    left = getattr(pdf, "l_margin", 10)
    right = getattr(pdf, "r_margin", 10)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.2)
    y = pdf.get_y()
    pdf.line(left, y, pdf.w - right, y)
    pdf.ln(3)

def _pdf_scores_one_col(pdf, items):
    pdf.set_font("Arial", "", 11)
    for k, v in items:
        pdf.cell(0, 6, txt=f"{k}: {v}/10", ln=True)
    pdf.ln(2)

def generate_pdf_report(entry):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.set_margins(10, 10, 10)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, txt="Evalia - Claim Analysis Report", ln=True, align='C')
        _pdf_divider(pdf)
        _pdf_h2(pdf, "Meta")
        _pdf_kv(pdf, "Timestamp", sanitize_for_pdf(entry.get("timestamp", "")))
        claim_clean = sanitize_for_pdf(entry.get("claim", ""))
        if claim_clean:
            _pdf_kv(pdf, "Claim", claim_clean)
        url = entry.get("url")
        if url:
            _pdf_kv(pdf, "URL", sanitize_for_pdf(url))
        img = entry.get("image_analysis") or {}
        if any(img.values()):
            _pdf_h2(pdf, "Image Summary")
            parts = [
                f"Description: {sanitize_for_pdf(img.get('description', ''))}" if img.get("description") else None,
                f"Assessment: {sanitize_for_pdf(img.get('assessment', ''))}" if img.get("assessment") else None,
                f"Extracted Text (snippet): {sanitize_for_pdf(img.get('extracted_text', '')[:400].rstrip() + '...' if len(img.get('extracted_text', '')) > 400 else img.get('extracted_text', ''))}" if img.get("extracted_text") else None
            ]
            parts = [p for p in parts if p]
            if parts:
                _pdf_p(pdf, "\n".join(parts))
                _pdf_divider(pdf)
        scores = entry.get("scores") or {}
        if scores:
            _pdf_h2(pdf, "Scores")
            items = [(k.replace("_", " ").title(), v) for k, v in scores.items()]
            _pdf_scores_one_col(pdf, items)
        analysis = entry.get("analysis", {})
        if analysis:
            _pdf_divider(pdf)
            _pdf_h1(pdf, "Analysis")
            if analysis.get("error"):
                _pdf_p(pdf, sanitize_for_pdf(analysis.get("error", "Analysis failed")))
            else:
                _pdf_kv(pdf, "Verdict", sanitize_for_pdf(analysis.get("verdict", "")))
                _pdf_kv(pdf, "Claim Summary", sanitize_for_pdf(analysis.get("claim_summary", "")))
                for category, reasoning in analysis.get("reasoning", {}).items():
                    _pdf_h2(pdf, category.replace("_", " ").title())
                    _pdf_p(pdf, sanitize_for_pdf(reasoning))
                _pdf_kv(pdf, "Grounding Meter", sanitize_for_pdf(analysis.get("grounding_meter", "")))
                _pdf_kv(pdf, "Emotion Meter", sanitize_for_pdf(analysis.get("emotion_meter", "")))
                _pdf_kv(pdf, "AI Origin", sanitize_for_pdf(analysis.get("ai_origin", "")))
                _pdf_kv(pdf, "Detected Style", sanitize_for_pdf(analysis.get("detected_style", "")))
                _pdf_h2(pdf, "Relevant Sources")
                for source in analysis.get("relevant_sources", []):
                    _pdf_p(pdf, f"{sanitize_for_pdf(source.get('annotation', ''))} - {sanitize_for_pdf(source.get('url', ''))}")
                _pdf_h2(pdf, "Suggested Research")
                for point in analysis.get("suggested_research", []):
                    _pdf_p(pdf, f"- {sanitize_for_pdf(point)}")
                _pdf_kv(pdf, "Final Commentary", sanitize_for_pdf(analysis.get("final_commentary", "")))
                _pdf_kv(pdf, "Confidence Level", str(analysis.get("confidence_level", 0)))
                _pdf_kv(pdf, "Truth Drift Score", str(analysis.get("truth_drift_score", 0)))
                _pdf_kv(pdf, "Claim Length", str(analysis.get("claim_length", 0)))
                _pdf_kv(pdf, "Temporal Reference", sanitize_for_pdf(analysis.get("temporal_reference", "")))

        pdf_file = f"evalia_report_{entry.get('timestamp', '').replace(':', '_')}.pdf"
        pdf.output(pdf_file)
        logger.info("Generated PDF report: %s", pdf_file)
        return pdf_file
    except Exception:
        logger.error("Failed to generate PDF report", exc_info=True)
        return None
