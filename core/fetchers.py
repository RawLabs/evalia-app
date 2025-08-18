"""External data fetchers (HTTP, files, etc.)."""
import requests
import base64
import io
from core.logging_config import configure_evalia_logger
from core.api_config import OPENAI_API_KEY
import openai

logger = configure_evalia_logger()
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def fetch_url_text(url):
    try:
        r = requests.get(url, timeout=10)
        return r.text[:3000]
    except Exception as e:
        logger.error("URL fetch error for %s", url, exc_info=True)
        return f"Error fetching URL: {str(e)}"

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
                {"role": "system", "content": "You are a precise image analysis assistant."},
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
                "extracted_text": raw_content,
                "description": "Error parsing description.",
                "assessment": "Error in assessment."
            }
    except Exception:
        logger.error("Image analysis error", exc_info=True)
        return {"extracted_text": "Error extracting text.", "description": "", "assessment": ""}
