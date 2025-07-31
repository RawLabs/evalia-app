import openai
import os
import json
import logging
from datetime import datetime, timezone
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from scipy import fftpack
from statistics import stdev, mean
import re
import tempfile  # Added for safe temp file handling to prevent permission issues
import base64  # Added for vision fallback
import io  # Added for BytesIO
import requests  # Added for source fetching
import pandas as pd  # Added for dataframes in charting
import plotly.express as px  # Added for interactive charts

try:
    from tavily import TavilyClient  # Added for fact-checking integration
except ImportError:
    TavilyClient = None

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment Setup
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set. Use 'export OPENAI_API_KEY=your_key' before running.")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
except EnvironmentError as e:
    st.error(str(e))
    logger.error(f"Environment setup error: {str(e)} with context: {os.environ.get('PATH')}")
    st.stop()

MEMORY_FILE = "evalia_memory.json"

# Source credibility dictionary
source_cred = {
    'associated press': 9,
    'ap': 9,
    'reuters': 9,
    'npr': 8,
    'bbc': 9,
    'pbs': 9,
    'bloomberg': 8,
    'washington post': 8,
    'new york times': 8,
    'abc': 8,
    'cbs': 8,
    'nbc': 8,
    'c-span': 9,
    'abc australia': 8,
    # Lower credibility examples
    'fox': 5,
    'newsmax': 3,
    # Add more as needed
}

# Create memory file if it doesn't exist
def initialize_memory():
    try:
        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'w') as f:
                json.dump([], f)
        return True
    except (IOError, PermissionError) as e:
        st.error(f"Failed to initialize memory file: {e}")
        logger.error(f"Memory initialization error: {str(e)}")
        return False

# Save analysis to memory
def save_to_memory(metadata):
    try:
        with open(MEMORY_FILE, 'r+') as f:
            data = json.load(f)
            data.append(metadata)
            f.seek(0)
            json.dump(data, f, indent=2)
        return True
    except (IOError, json.JSONDecodeError, PermissionError) as e:
        st.error(f"Failed to save to memory: {e}")
        logger.error(f"Memory save error: {str(e)}")
        return False

# Sanitize user input
def sanitize_input(claim):
    if len(claim) > 10000:
        raise ValueError("Claim is too long. Please keep it under 10,000 characters.")
    # Remove emojis and special characters
    claim = re.sub(r'[\U00010000-\U0010ffff]', '', claim)  # Remove emojis
    claim = re.sub(r'[^\w\s\.,!?]', '', claim)  # Remove other special chars except basic punctuation
    return " ".join(claim.strip().split())

# Visual helpers (kept for meters; bars now use Plotly)
def grounding_meter(score):
    if score >= 40:
        position = "█" * 8 + "░░"
        label = "Leaning Truth"
    elif score >= 25:
        position = "█" * 5 + "░" * 5
        label = "Borderline"
    else:
        position = "█" * 2 + "░" * 8
        label = "Unverified"
    return f"`Unverified ←─{position}─→ Fact`  (Grounding Meter: 'Fact' means based on evidence, 'Unverified' means source is unavailable for verification) ({label})"

def emotion_meter(score, emotional_phrases):
    if score >= 7:
        level = "🔥🔥🔥"
        label = f"Highly Charged (detected phrases: {', '.join(emotional_phrases)})"
        position = "█" * 8 + "░░"
    elif score >= 4:
        level = "🔥🔥"
        label = f"Emotionally Tinted (detected phrases: {', '.join(emotional_phrases)})"
        position = "█" * 5 + "░" * 5
    elif score >= 1:
        level = "🔥"
        label = f"Low Emotion (detected phrases: {', '.join(emotional_phrases)})"
        position = "█" * 2 + "░" * 8
    else:
        level = ""
        label = "Neutral (no emotional phrases detected)"
        position = "░░" * 10
    return f"`Neutral ←─{position}─→ Charged`  🧠 Emotion Meter: {level}  ({label})"

def ai_origin_meter(ai_score):
    if ai_score >= 0.8:
        position = "█" * 8 + "░░"
        label = "Likely AI-Generated"
    elif ai_score >= 0.5:
        position = "█" * 5 + "░" * 5
        label = "Possibly AI-Generated"
    else:
        position = "█" * 2 + "░" * 8
        label = "Likely Human-Written"
    return f"`Human ←─{position}─→ AI`  ({label})"

# GPT-based Style detection
def detect_style_gpt(claim):
    try:
        styles_list = ["Symbolic/metaphysical", "Conspiratorial", "Academic", "Emotional/personal", "Basic/blunt", "Exploratory"]
        prompt = (
            f"Classify the style of the following claim into one of these categories: {', '.join(styles_list)}. "
            "Consider keywords and tone. "
            "Return ONLY in this exact format: 'Style: X, Confidence: Y' where X is the style and Y is a number from 0.0 to 1.0."
            f"\n\nClaim: {claim}"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"Style:\s*(.+?)\s*,\s*Confidence:\s*(\d*\.?\d+)", content, re.IGNORECASE)
        if match:
            style = match.group(1).strip()
            confidence = float(match.group(2))
            if style not in styles_list:
                style = "Exploratory"
                confidence = 0.1
            return style, confidence
        else:
            logger.warning(f"Style detection parsing issue: {content}")
            return "Exploratory", 0.1
    except Exception as e:
        logger.error(f"Style detection error: {str(e)}")
        return "Exploratory", 0.1

# GPT-based Emotion detection
def detect_emotion_gpt(claim):
    try:
        prompt = (
            "Detect the emotional intensity in the following claim on a scale of 0-10 (10 highly charged). "
            "List any emotional phrases detected. "
            "Return ONLY in this exact format: 'Score: Z, Phrases: phrase1,phrase2,...' where Z is the score and phrases are comma-separated (or 'None' if none)."
            f"\n\nClaim: {claim}"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"Score:\s*(\d*\.?\d+)\s*,\s*Phrases:\s*(.+)", content, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            phrases = [p.strip() for p in match.group(2).split(',') if p.strip() != 'None']
            return min(score, 10), phrases
        else:
            logger.warning(f"Emotion detection parsing issue: {content}")
            return 0, []
    except Exception as e:
        logger.error(f"Emotion detection error: {str(e)}")
        return 0, []

# Feedback style instructions
def feedback_style_instruction(style):
    return {
        "Symbolic/metaphysical": "Respect metaphorical framing; gently guide toward evidence without dismissing symbolic meaning.",
        "Conspiratorial": "Be clear and firm on facts, but maintain a respectful and curious tone. Avoid ridicule.",
        "Academic": "Use precise language and address theoretical constructs with clarity.",
        "Emotional/personal": "Be compassionate. Frame critique as support for personal growth.",
        "Basic/blunt": "Keep it simple, direct, and helpful without jargon.",
        "Exploratory": "Encourage inquiry and deeper exploration in a welcoming, curious tone."
    }.get(style, "Encourage inquiry and deeper exploration in a welcoming, curious tone.")

# Check source credibility
def check_source_credibility(source_url, potential_sources):
    base_score, base_comment = 1, "No source provided; credibility cannot be assessed."
    if source_url:
        if any(platform in source_url for platform in ["instagram.com", "twitter.com", "rumble.com", "x.com", "t.me"]):
            base_score, base_comment = 2, "Social media source detected. Credibility is low due to lack of peer review or editorial standards."
        else:
            base_score, base_comment = 5, "Source provided but not fully analyzed."
    
    if potential_sources:
        scores = [source_cred.get(src.lower(), 5) for src in potential_sources]
        avg_score = mean(scores) if scores else base_score
        comment = f"Mentioned sources: {', '.join(potential_sources)}. Average credibility: {avg_score:.1f}/10. {base_comment}"
        return avg_score, comment
    return base_score, base_comment

# Aggregate AI scores (for multiple texts/logs)
def aggregate_ai_scores(scores):
    if not scores:
        return 0.5
    return sum(scores) / len(scores)

# Improved AI Detection Function with retry
def detect_ai_origin(text, retry_count=0):
    try:
        sentences = [s for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.5, "Insufficient text for analysis."
        
        lengths = [len(s.split()) for s in sentences]
        length_std = stdev(lengths) / mean(lengths) if len(lengths) > 1 and mean(lengths) > 0 else 0
        uniformity_score = 1 - min(length_std, 1)
        
        words = text.lower().split()
        word_freq = {w: words.count(w) / len(words) for w in set(words)}
        perplexity_proxy = 1 / max(word_freq.values()) if word_freq else 1
        perplexity_score = min(perplexity_proxy / 10, 1)
        
        prompt = (
            "Analyze the following text to determine if it was generated by an AI (e.g., GPT, LLaMA) or written by a human. "
            "Consider linguistic patterns like repetitive phrasing, uniform sentence structure, or lack of personal nuance. "
            "Return ONLY in this exact format: 'Probability: X, Explanation: Y' where X is a number from 0.0 to 1.0 (1.0 means definitely AI-generated)."
            f"\n\nText: {text}"
        )
        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to gpt-4o for better performance
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
        ai_prob = 0.5
        ai_explanation = "Unable to determine AI probability due to parsing issue."
        match = re.search(r"probability:\s*(\d*\.?\d+).*explanation:\s*(.+)", content, re.DOTALL | re.IGNORECASE)
        if match:
            ai_prob = float(match.group(1))
            ai_explanation = match.group(2).strip()
        else:
            if retry_count < 1:  # Retry once
                reformat_prompt = f"Reformat this response to exactly: 'Probability: X, Explanation: Y': {content}"
                retry_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": reformat_prompt}]
                )
                retry_content = retry_response.choices[0].message.content.strip()
                return detect_ai_origin(text, retry_count=1)  # Recursive retry with content
            st.warning(f"AI detection parsing issue: Expected 'Probability: X, Explanation: Y' format, got '{content}'")
            logger.warning(f"AI detection parsing issue: {content}")
        
        combined_score = 0.6 * ai_prob + 0.2 * uniformity_score + 0.2 * perplexity_score
        return min(max(combined_score, 0.0), 1.0), ai_explanation
    except Exception as e:
        st.error(f"AI detection error: {str(e)}")
        logger.error(f"AI detection error: {str(e)} with context: {str(locals())}")
        return 0.5, f"AI detection failed: {str(e)}"

# Image forensic analysis with AI detection
def analyze_image(image_file):
    try:
        if not image_file:
            return "Error: No image file uploaded.", None, None, 0
        logger.info(f"Processing image: {image_file.name}")
        image_bytes = image_file.getvalue()  # Read once to avoid buffer issues
        img = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess for OCR: Enhance contrast, resize, threshold
        preprocessed_img = img.convert('L')  # Grayscale
        preprocessed_img = preprocessed_img.resize((preprocessed_img.width * 2, preprocessed_img.height * 2))  # Resize up
        preprocessed_img = preprocessed_img.point(lambda x: 0 if x < 128 else 255)  # Threshold
        
        try:
            extracted_text = pytesseract.image_to_string(preprocessed_img)
            extracted_text = sanitize_input(extracted_text) if extracted_text else "No text detected in image."
            logger.info(f"Extracted text (first 100 chars): {extracted_text[:100]}...")  # Enhanced logging
        except Exception as e:
            extracted_text = f"OCR failed: {str(e)}. Ensure Tesseract OCR is installed and configured (e.g., 'tesseract --version')."
            logger.error(f"OCR error: {str(e)} with context: {str(locals())}")
        
        # Check OCR confidence: Simple heuristic for nonsense (e.g., >30% short words or low length)
        words = extracted_text.split()
        if extracted_text != "No text detected in image." and (len(words) < 5 or sum(1 for w in words if len(w) < 3) / len(words) > 0.3 if words else True):
            logger.warning("Low OCR confidence; falling back to vision for cross-verification.")
            try:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                vision_prompt = "Extract any text from this image accurately, including layout if relevant."
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a precise text extractor."},
                        {"role": "user", "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ]
                )
                vision_text = response.choices[0].message.content.strip()
                extracted_text = sanitize_input(vision_text) or extracted_text  # Use vision if better
                logger.info(f"Vision cross-verified text (first 100 chars): {extracted_text[:100]}...")
            except Exception as e:
                logger.error(f"Vision cross-verify error: {str(e)}")
        
        # Vision fallback if still no text
        if extracted_text == "No text detected in image." or not extracted_text.strip():
            try:
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                vision_prompt = "Describe the image in detail, including any text, scene, people, and implied message or humor if it's a meme."
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful image describer."},
                        {"role": "user", "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ]
                )
                extracted_text = response.choices[0].message.content.strip()
                extracted_text = sanitize_input(extracted_text)
                logger.info(f"Vision description (first 100 chars): {extracted_text[:100]}...")
            except Exception as e:
                extracted_text = f"Vision description failed: {str(e)}"
                logger.error(f"Vision error: {str(e)}")
        
        exif_data = {}
        try:
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
            exif_summary = f"EXIF data: {exif_data.get('DateTime', 'No date found')}, Device: {exif_data.get('Model', 'Unknown')}"
            logger.debug(f"EXIF summary: {exif_summary}")  # Enhanced logging
        except Exception as e:
            exif_summary = "No EXIF data available or image format unsupported."
            logger.warning(f"EXIF extraction failed: {str(e)} with context: {str(locals())}")
        
        tampering_score = 5
        tampering_comment = "No clear signs of tampering detected (basic analysis)."
        tampered_img = None
        mean_diff = 0
        try:
            # Direct PIL to OpenCV conversion
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if img_cv is None:
                raise ValueError("Invalid image data; could not convert.")
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                compressed = cv2.imread(tmp.name)
            if compressed is None:
                tampering_comment = "Image decoding failed; skipping advanced analysis."
                logger.warning("Compressed image is None.")
            else:
                diff = cv2.absdiff(img_cv, compressed)
                logger.debug(f"Diff shape: {diff.shape}, Mean diff: {np.mean(diff)}")  # Enhanced logging
                mean_diff = np.mean(diff)
                if mean_diff > 50:
                    tampering_score = 3
                    tampering_comment = "Possible tampering detected (high difference in compression artifacts)."
                tampered_img = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)  # Convert for display
                logger.debug(f"Tampered image generated: {tampered_img is not None}, Shape: {tampered_img.shape if tampered_img is not None else 'None'}")
                if tampered_img is not None and tampered_img.size == 0:
                    logger.warning("Tampered image is empty, skipping display.")
                    tampered_img = None
            os.unlink(tmp.name)  # Clean up temp file
        except Exception as e:
            tampering_comment = f"Tampering analysis failed: {str(e)}"
            logger.error(f"Tampering analysis error: {str(e)} with context: {str(locals())}")
        
        ai_score = 0.5
        ai_comment = "No clear AI-generated artifacts detected."
        try:
            img_cv_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)  # Direct grayscale
            freq = fftpack.fft2(img_cv_gray)
            freq_shift = fftpack.fftshift(freq)
            magnitude = np.abs(freq_shift)
            high_freq_energy = np.sum(magnitude[int(img_cv_gray.shape[0]/4):int(3*img_cv_gray.shape[0]/4), 
                                               int(img_cv_gray.shape[1]/4):int(3*img_cv_gray.shape[1]/4)])
            total_energy = np.sum(magnitude)
            freq_score = min(high_freq_energy / total_energy if total_energy > 0 else 0, 1)
            
            prompt = (
                "Analyze the following image description to determine if the image was likely generated by an AI (e.g., DALL·E, Stable Diffusion). "
                "Consider unnatural textures, inconsistent lighting, or overly polished visuals. "
                "Return a response in the format: 'Probability: X (0.0 to 1.0, where 1.0 is definitely AI-generated), Explanation: Y'."
                f"\n\nDescription: {extracted_text}"
            )
            response = client.chat.completions.create(
                model="gpt-4o",  # Updated to gpt-4o
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            ai_prob = 0.5
            ai_explanation = "Unable to determine AI probability due to parsing issue."
            match = re.search(r"Probability:\s*(\d*\.?\d+).*Explanation:\s*(.+)", content, re.DOTALL | re.IGNORECASE)
            if match:
                ai_prob = float(match.group(1))
                ai_explanation = match.group(2).strip()
            else:
                if retry_count < 1:
                    reformat_prompt = f"Reformat this response to exactly: 'Probability: X, Explanation: Y': {content}"
                    retry_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": reformat_prompt}]
                    )
                    retry_content = retry_response.choices[0].message.content.strip()
                    # Parse retry
                    retry_match = re.search(r"Probability:\s*(\d*\.?\d+).*Explanation:\s*(.+)", retry_content, re.DOTALL | re.IGNORECASE)
                    if retry_match:
                        ai_prob = float(retry_match.group(1))
                        ai_explanation = retry_match.group(2).strip()
                else:
                    st.warning(f"AI detection parsing issue: Expected 'Probability: X, Explanation: Y' format, got '{content}'")
                    logger.warning(f"AI detection parsing issue: {content}")
            
            ai_score = 0.5 * (1 - freq_score) + 0.5 * ai_prob
            ai_comment = f"Frequency Analysis: {freq_score:.2f}, GPT-4o: {ai_explanation}"
            logger.debug(f"AI image score: {ai_score}, Comment: {ai_comment}")  # Enhanced logging
        except Exception as e:
            ai_comment = f"AI detection failed: {str(e)}"
            logger.error(f"AI detection error: {str(e)} with context: {str(locals())}")
        
        text_ai_score, text_ai_comment = detect_ai_origin(extracted_text) if extracted_text != "No text detected in image." else (0.5, "No text to analyze.")
        aggregated_ai = aggregate_ai_scores([ai_score, text_ai_score])  # Aggregate image and text AI scores
        
        logic_score = 5 if extracted_text != "No text detected in image." else 3
        natural_law_score = 5
        historical_accuracy_score = 5
        source_credibility_score = tampering_score
        reasonableness_score = 4 if tampering_score < 5 else 5
        weighted_score = (logic_score + natural_law_score * 1.5 + historical_accuracy_score +
                         source_credibility_score * 1.5 + reasonableness_score)
        
        analysis_summary = (
            f"Image Analysis Results:\n"
            f"- Extracted Text: {extracted_text}\n"
            f"- EXIF Metadata: {exif_summary}\n"
            f"- Tampering Analysis: {tampering_comment} (Score: {tampering_score}/10)\n"
            f"- AI Origin (Image): {ai_comment} (Score: {ai_score:.2f})\n"
            f"- AI Origin (Text): {text_ai_comment} (Score: {text_ai_score:.2f})\n"
            f"- Aggregated AI Score: {aggregated_ai:.2f}\n"
            f"Preliminary Scores (subject to LLM refinement):\n"
            f"- Logic: {logic_score}/10\n"
            f"- Natural Law: {natural_law_score}/10\n"
            f"- Historical Accuracy: {historical_accuracy_score}/10\n"
            f"- Source Credibility: {source_credibility_score}/10\n"
            f"- Overall Reasonableness: {reasonableness_score}/10\n"
            f"Weighted Score: {weighted_score}/50"
        )
        
        style, style_confidence = detect_style_gpt(extracted_text) if extracted_text != "No text detected in image." else ("Exploratory", 0.1)
        style_guidance = feedback_style_instruction(style)
        
        # Fact-check integration for image text
        tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY")) if TavilyClient and os.getenv("TAVILY_API_KEY") else None
        fact_check_summary = ""
        if tavily_client:
            try:
                search_results = tavily_client.search(query=f"fact check: {extracted_text}", max_results=5)
                fact_check_summary = "\n".join([f"{r['title']}: {r['content']}" for r in search_results.get('results', [])])
            except Exception as e:
                logger.error(f"Fact-check error: {str(e)}")
                fact_check_summary = "Fact-check failed."
        else:
            fact_check_summary = "No external fact-check available. Set TAVILY_API_KEY for integration."
        
        system_prompt = (
            "You are Evalia, an advanced reasoning engine designed to evaluate claims for misinformation, including those derived from images. "
            "You have been provided with forensic analysis of an image, including extracted text, EXIF metadata, tampering indicators, and AI origin detection. "
            "Evaluate the claim (extracted text or implied message) based on the following criteria:\n"
            "1. Logic (10 pts): Is the claim or visual narrative internally consistent and free of logical fallacies?\n"
            "2. Natural Law (10 pts ×1.5): Does the claim or image align with established scientific principles?\n"
            "3. Historical Accuracy (10 pts): Is the claim or image consistent with verified historical records?\n"
            "4. Source Credibility (10 pts ×1.5): Is the image authentic (based on tampering and AI analysis)?\n"
            "5. Overall Reasonableness (10 pts): Does the claim or image make sense in a broader context?\n"
            f"Image Analysis Summary:\n{analysis_summary}\n"
            f"External fact-check results:\n{fact_check_summary}\n"
            f"Detected claim style: {style} (confidence: {style_confidence:.2f}). "
            f"Adjust feedback style: {style_guidance}\n"
            "For conspiratorial styles, prioritize empathy (e.g., 'It's totally valid to question powerful figures and industries—many feel betrayed by events like Warp Speed. Let's unpack this together without judgment.'), Socratic questioning (e.g., 'What specific evidence convinced you of gene editing in the air? If we found studies showing otherwise, would that shift your view?'), balanced counter-evidence (layer facts gradually, start with neutral sources like independent labs, include visuals or unedited clips), addressing conspiracy roots (gently debunk origins, e.g., 'These ideas often stem from real concerns like overpopulation, but evidence points elsewhere.'), empowering self-verification (point to tools like 'Try searching FDA databases yourself for vaccine ingredients, or watch unedited clips. Here's a start: CDC reports.'), and lighter tones (conversational, e.g., 'We've all been down rabbit holes—let's climb out with facts.').\n"
            "Return a weighted score (out of 50) and a confidence estimate percentage, with commentary. "
            "Include an 'AI Origin' section assessing whether the image or text is AI-generated or human-written, using provided AI detection scores. "
            "Final commentary must sound human, lightly wise, persuasive without being preachy, and digestible to the average person—especially those skeptical or overwhelmed. "
            "Use phrases like 'It’s understandable to suspect hidden motives' to connect with skeptical audiences. "
            "Include 1–2 specific, reputable sources relevant to the claim’s topic. "
            "Encourage users to verify information themselves with a positive, empowering tone.\n"
            "For the Emotion Meter, infer emotional intensity from the extracted text or visual tone.\n"
            "Present results using simple score format without unicode bars (e.g., Logic: X/10).\n"
            "Format your entire response as well-structured Markdown. Use ## for headings, - for bullet lists, **bold** for emphasis, and ensure each section is on a new line with proper spacing.\n"
            "Include:\n"
            f"- 🔥 Verdict: Plausible / Implausible / Speculative / Unknown / Proven\n"
            f"- 📊 Bar-style Score Overview (list scores as Category: X/10)\n"
            f"- 🌺 Grounding Meter (Unverified ←─██░░░░░░░░─→ Fact)\n"
            f"- 🧠 Emotion Meter (Neutral ←─██░░░░░░░░─→ Charged, based on emotional intensity)\n"
            f"- 🤖 AI Origin (Human ←─██░░░░░░░░─→ AI)\n"
            f"- 📝 Detected Style: {style}\n"
            f"- 🧪 Reasoning per category\n"
            f"- 📚 Relevant sources or background\n"
            f"- 📌 Suggested further research or similar studies\n"
            f"- 🧽 Final commentary (aimed at skeptical and confused minds)\n"
            f"- 📾 Confidence Level\n"
            f"- 🎯 Truth Drift Score (Grounded / Speculative / Detached)\n"
            f"- 📊 Claim Length (in words, for extracted text)\n"
            f"- ⏳ Temporal Reference (recent/timeless/historical/future-focused)"
        )

        user_prompt = f"Image Claim:\n{extracted_text}\n\nForensic Analysis:\n{analysis_summary}"

        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to gpt-4o
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        content = response.choices[0].message.content
        if not content:
            logger.error("No analysis result from AI with context: {str(locals())}")
            return "Error: No analysis result generated by AI. Check API status or simplify prompt.", None, None, 0
        
        # Parse scores from content for Plotly chart
        categories = ['Logic', 'Natural Law', 'Historical Accuracy', 'Source Credibility', 'Overall Reasonableness']
        scores = []
        for cat in categories:
            match = re.search(rf"{cat}:\s*(\d+)/10", content)
            scores.append(int(match.group(1)) if match else 0)
        df = pd.DataFrame({'Category': categories, 'Score': scores})
        fig = px.bar(df, x='Score', y='Category', orientation='h', 
                     title='Score Overview', range_x=[0, 10],
                     color='Score', color_continuous_scale='blues')
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claim_length": len(extracted_text.split()) if extracted_text != "No text detected in image." else 0,
            "style": style,
            "style_confidence": style_confidence,
            "image_analysis": analysis_summary,
            "ai_origin_image": ai_score,
            "ai_origin_text": text_ai_score,
            "raw_feedback": content
        }
        
        if save_to_memory(metadata):
            logger.info("Metadata saved to memory.")
        return content, fig, tampered_img, mean_diff
    
    except Exception as e:
        st.error(f"Image analysis error: {str(e)}")
        logger.error(f"Image analysis error: {str(e)} with context: {str(locals())}")
        return f"Error: Could not analyze image - {str(e)}", None, None, 0

# Evaluate text claim with AI detection
def evaluate_claim(claim, source_url=None):
    try:
        claim = sanitize_input(claim)
        if not initialize_memory():
            return "Error: Could not initialize memory.", None
        
        style, style_confidence = detect_style_gpt(claim)
        style_guidance = feedback_style_instruction(style)
        
        emotion_score, emotional_phrases = detect_emotion_gpt(claim)
        
        # Extract potential sources
        potential_sources = re.findall(r'(associated press|ap|reuters|npr|bbc|pbs|bloomberg|washington post|new york times|abc|cbs|nbc|c-span|abc australia|fox|newsmax)', claim.lower())
        source_score, source_comment = check_source_credibility(source_url, potential_sources)
        
        ai_score, ai_comment = detect_ai_origin(claim)
        
        # Fact-check integration
        tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY")) if TavilyClient and os.getenv("TAVILY_API_KEY") else None
        fact_check_summary = ""
        if tavily_client:
            try:
                search_results = tavily_client.search(query=f"fact check: {claim}", max_results=5)
                fact_check_summary = "\n".join([f"{r['title']}: {r['content']}" for r in search_results.get('results', [])])
            except Exception as e:
                logger.error(f"Fact-check error: {str(e)}")
                fact_check_summary = "Fact-check failed."
        else:
            fact_check_summary = "No external fact-check available. Set TAVILY_API_KEY for integration."
        
        # Fetch source content if URL provided
        source_summary = ""
        if source_url:
            try:
                response = requests.get(source_url, timeout=10)
                source_content = response.text[:2000]  # Truncate to avoid token limits
                summary_prompt = f"Summarize the key points from this source content relevant to the claim: {source_content[:1000]}"  # Further truncate for prompt
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                source_summary = resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Source fetch error: {str(e)}")
                source_summary = "Unable to fetch or summarize source content."
        
        system_prompt = (
            "You are Evalia, an advanced reasoning engine designed to evaluate claims for misinformation. "
            "Evaluate claims based on the following criteria:\n"
            "1. Logic (10 pts): Is the claim internally consistent and free of logical fallacies?\n"
            "2. Natural Law (10 pts ×1.5): Does the claim align with established scientific principles?\n"
            "3. Historical Accuracy (10 pts): Is the claim consistent with verified historical records?\n"
            "4. Source Credibility (10 pts ×1.5): Are the sources reliable and authoritative? " + source_comment + "\n"
            f"Source summary (if available): {source_summary}\n"
            "5. Overall Reasonableness (10 pts): Does the claim make sense in a broader context?\n"
            f"External fact-check results:\n{fact_check_summary}\n"
            f"Detected claim style: {style} (confidence: {style_confidence:.2f}). "
            f"Adjust feedback style: {style_guidance}\n"
            "For conspiratorial styles, prioritize empathy (e.g., 'It's totally valid to question powerful figures and industries—many feel betrayed by events like Warp Speed. Let's unpack this together without judgment.'), Socratic questioning (e.g., 'What specific evidence convinced you of gene editing in the air? If we found studies showing otherwise, would that shift your view?'), balanced counter-evidence (layer facts gradually, start with neutral sources like independent labs, include visuals or unedited clips), addressing conspiracy roots (gently debunk origins, e.g., 'These ideas often stem from real concerns like overpopulation, but evidence points elsewhere.'), empowering self-verification (point to tools like 'Try searching FDA databases yourself for vaccine ingredients, or watch unedited clips. Here's a start: CDC reports.'), and lighter tones (conversational, e.g., 'We've all been down rabbit holes—let's climb out with facts.').\n"
            f"AI Origin Analysis: Probability of AI generation: {ai_score:.2f}. {ai_comment}\n"
            "Return a weighted score (out of 50) and a confidence estimate percentage, with commentary. "
            "Include an 'AI Origin' section assessing whether the claim is AI-generated or human-written, using the provided AI detection score. "
            "Final commentary must sound human, lightly wise, persuasive without being preachy, and digestible to the average person—especially those skeptical or overwhelmed. "
            "Use phrases like 'It’s understandable to suspect hidden motives' to connect with skeptical audiences. "
            "Include 1–2 specific, reputable sources relevant to the claim’s topic. "
            "Encourage users to verify information themselves with a positive, empowering tone.\n"
            "For the Emotion Meter, use the provided emotional phrases: " + (", ".join(emotional_phrases) if emotional_phrases else "None") + ".\n"
            "Present results using simple score format without unicode bars (e.g., Logic: X/10).\n"
            "Format your entire response as well-structured Markdown. Use ## for headings, - for bullet lists, **bold** for emphasis, and ensure each section is on a new line with proper spacing.\n"
            "Include:\n"
            f"- 🔥 Verdict: Plausible / Implausible / Speculative / Unknown / Proven\n"
            f"- 📊 Bar-style Score Overview (list scores as Category: X/10)\n"
            f"- 🌺 Grounding Meter (Unverified ←─██░░░░░░░░─→ Fact)\n"
            f"- 🧠 Emotion Meter (Neutral ←─██░░░░░░░░─→ Charged, based on emotional intensity, using provided phrases)\n"
            f"- 🤖 AI Origin (Human ←─██░░░░░░░░─→ AI)\n"
            f"- 📝 Detected Style: {style}\n"
            f"- 🧪 Reasoning per category\n"
            f"- 📚 Relevant sources or background\n"
            f"- 📌 Suggested further research or similar studies\n"
            f"- 🧽 Final commentary (aimed at skeptical and confused minds)\n"
            f"- 📾 Confidence Level\n"
            f"- 🎯 Truth Drift Score (Grounded / Speculative / Detached)\n"
            f"- 📊 Claim Length (in words)\n"
            f"- ⏳ Temporal Reference (recent/timeless/historical/future-focused)"
        )

        user_prompt = f"Claim:\n{claim}"

        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to gpt-4o
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        content = response.choices[0].message.content
        
        # Parse scores from content for Plotly chart
        categories = ['Logic', 'Natural Law', 'Historical Accuracy', 'Source Credibility', 'Overall Reasonableness']
        scores = []
        for cat in categories:
            match = re.search(rf"{cat}:\s*(\d+)/10", content)
            scores.append(int(match.group(1)) if match else 0)
        df = pd.DataFrame({'Category': categories, 'Score': scores})
        fig = px.bar(df, x='Score', y='Category', orientation='h', 
                     title='Score Overview', range_x=[0, 10],
                     color='Score', color_continuous_scale='blues')
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claim_length": len(claim.split()),
            "style": style,
            "style_confidence": style_confidence,
            "emotional_phrases": emotional_phrases,
            "ai_origin": ai_score,
            "ai_comment": ai_comment,
            "raw_feedback": content
        }
        
        if save_to_memory(metadata):
            logger.info("Text claim metadata saved.")
        return content, fig
    
    except ValueError as e:
        st.error(str(e))
        logger.error(f"Claim input error: {str(e)}")
        return "Error: Invalid claim input.", None
    except openai.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        logger.error(f"OpenAI API error: {str(e)}")
        return "Error: Could not evaluate claim due to API issues.", None

# Placeholder for video analysis
def analyze_video(video_file):
    st.warning("Video analysis not yet implemented.")
    logger.warning("Video analysis not implemented.")
    return None

# Function to derive claim from URL
def derive_claim_from_url(source_url):
    try:
        response = requests.get(source_url, timeout=10)
        if response.status_code != 200:
            raise Exception("Failed to fetch URL")
        content_type = response.headers.get('Content-Type', '').lower()
        if 'image' in content_type:
            # Download image to BytesIO for analysis
            image_bytes = io.BytesIO(response.content)
            image_bytes.name = "url_image.jpg"  # Fake name for file_uploader simulation
            return None, image_bytes, None  # Return as image_file
        elif 'video' in content_type:
            # For video, warn as not implemented
            st.warning("Video URLs not yet supported for direct evaluation.")
            return None, None, None
        else:
            # Text/HTML: Summarize as claim
            source_content = response.text[:2000]
            summary_prompt = "Extract the main claim or key message from this webpage content. If it's a social post, include the post text."
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"{summary_prompt}\n\nContent: {source_content}"}]
            )
            derived_claim = resp.choices[0].message.content.strip()
            return derived_claim, None, None  # Return as claim
    except Exception as e:
        logger.error(f"URL derivation error: {str(e)}")
        st.error(f"Could not derive content from URL: {str(e)}")
        return None, None, None

# Streamlit App with AI test prompt
initialize_memory()
st.set_page_config(
    page_title="Evalia",
    layout="wide",
    page_icon=":mag:",
    initial_sidebar_state="collapsed"
)

# Logo options
st.logo('evalia logo 3d.jpg', size='large')  # Primary: Try large size
# Fallback if logo not showing (uncomment to use)
# st.image('evalia logo 3d.jpg', use_column_width='auto', caption='Evalia Logo')

# Load and encode background image
try:
    with open("68887836-metal-background-texture-of-titanium-sheet-of-metal-surface-steel.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    background_image = f"data:image/jpeg;base64,{encoded_string}"
except FileNotFoundError:
    st.warning("Background image '68887836-metal-background-texture-of-titanium-sheet-of-metal-surface-steel.jpg' not found. Falling back to gradient.")
    logger.warning("Background image not found.")
    background_image = "linear-gradient(to bottom, #A9A9A9, #FFFFFF)"  # Fallback gradient

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #003087;
    }}
    .stTitle {{
        color: #003087;
        font-family: 'Roboto', sans-serif
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);  /* Subtle shadow */
    }}
    .stTextArea, .stFileUploader, .stTextInput {{
        background-color: #282828;
        color: #E0E0E0;
        border: 1px solid #003087;
        border-radius: 8px;  /* Softer corners */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);  /* Light shadow for depth */
    }}
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {{
        color: #A0A0A0;
    }}
    .stButton>button {{
        background-color: #003087;
        color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: background-color 0.3s;  /* Hover animation */
    }}
    .stButton>button:hover {{
        background-color: #001F5F;
    }}
    .output-box {{
        background-color: #282828;
        color: #E0E0E0;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .footer {{
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #666;
        font-size: 12px;
        background-color: rgba(255,255,255,0.8);  /* Semi-transparent for overlay */
    }}
    </style>
    <div class="stTitle"><h1>🔍 Evalia</h1><p>Evaluate with Confidence</p></div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
Evaluate claims, images, videos for plausibility, credibility, and AI origin. Paste a claim here or upload media to get started.
""")

# Improved layout: Use columns for inputs
col1, col2 = st.columns(2)
with col1:
    claim = st.text_area("Paste your claim here:", placeholder="Place claim here...", height=200)
    source_url = st.text_input("Enter source URL (optional):", placeholder="Insert URL here")
with col2:
    image_file = st.file_uploader("Upload an image or meme (optional)", type=["png", "jpg", "jpeg"])
    video_file = st.file_uploader("Upload a video (optional)", type=["mp4", "mov"])

if st.button("Evaluate Claim"):
    derived_claim = None
    derived_image = None
    derived_video = None
    if source_url and not (claim or image_file or video_file):
        with st.spinner("Deriving content from URL..."):
            derived_claim, derived_image, derived_video = derive_claim_from_url(source_url)
    if claim or image_file or video_file or derived_claim or derived_image or derived_video:
        effective_claim = claim or derived_claim or ""
        effective_image = image_file or derived_image
        effective_video = video_file or derived_video
        if effective_claim:
            with st.spinner("Evaluating text claim..."):
                result, fig = evaluate_claim(effective_claim, source_url)
            if derived_claim:
                st.write("Derived Claim from URL:", effective_claim)
            st.markdown("---")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown(f'<div class="output-box">{result}</div>', unsafe_allow_html=True)
        if effective_image:
            st.image(effective_image, caption="Image from URL Preview", use_container_width=True)
            with st.spinner("Analyzing image..."):
                image_result, image_fig, tampered_img, mean_diff = analyze_image(effective_image)
            if not image_result.startswith("Error:"):
                st.markdown("---")
                if image_fig:
                    st.plotly_chart(image_fig, use_container_width=True)
                if tampered_img is not None:
                    if mean_diff < 10:
                        st.write("No significant tampering detected; difference image is uniform.")
                    else:
                        st.image(tampered_img, caption="Tampered Image Preview (Difference Analysis)", use_container_width=True)
                st.markdown(f'<div class="output-box">{image_result}</div>', unsafe_allow_html=True)
                st.success("Analysis completed. Scroll up to review results.")
            else:
                st.error(image_result)
        if effective_video:
            with st.spinner("Analyzing video..."):
                video_result = analyze_video(effective_video)
                if video_result:
                    st.markdown(video_result)
    else:
        st.warning("Please provide a claim, image, video, or URL to evaluate.")

st.markdown('<div class="footer">Powered by Evalia 🔍</div>', unsafe_allow_html=True)

# Dependencies Note (for Pop!_OS/Ubuntu-based systems):
# System: sudo apt install tesseract-ocr tesseract-ocr-eng python3-opencv libopencv-dev
# Python (pip): openai streamlit pillow pytesseract opencv-python scipy statistics numpy re tempfile base64 io requests pandas plotly
# Test: tesseract --version; python -c "import cv2, pytesseract"