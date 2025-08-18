
import json
import re
from core.logging_config import configure_evalia_logger
from core.prompts import STOIC_SCORING_PROMPT, BRUTAL_SCORING_PROMPT
from core.api_config import OPENAI_API_KEY, MEMORY_FILE
import openai

logger = configure_evalia_logger()
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def sanitize_input(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[^\w\s\.,!?:/\-\(\)\[\]]', '', text)
    return " ".join(text.strip().split())

def save_to_memory(entry):
    try:
        enhanced_entry = {
            **entry,
            "claim_word_count": len(entry.get("claim", "").split()),
            "had_url": bool(entry.get("url")),
            "had_image": bool(entry.get("image_analysis")),
            "persona_used": "brutal" if entry.get("brutality_mode") else "stoic",
            "scores_generated": bool(entry.get("scores")),
            "analysis_length": len(json.dumps(entry.get("analysis", ""))),
            "verdict_extracted": bool(entry.get("analysis", {}).get("verdict")),
            "all_scores_present": len(entry.get("scores", {})) == 5,
        }
        with open(MEMORY_FILE, 'r+') as f:
            data = json.load(f)
            data.append(enhanced_entry)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
        logger.info("Enhanced data saved: %s (%d words, %s mode)",
                    enhanced_entry.get("claim", "")[:50] + "...",
                    enhanced_entry['claim_word_count'],
                    enhanced_entry['persona_used'])
    except Exception:
        logger.error("Failed to save to memory", exc_info=True)

def score_claim(text, brutality_mode=False):
    try:
        cleaned = sanitize_input(text)
        sys_prompt = BRUTAL_SCORING_PROMPT if brutality_mode else STOIC_SCORING_PROMPT

        def ask(prompt, brutality_mode, retries=2):
            temp = 0.1
            for attempt in range(retries + 1):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": f"Claim:\n{cleaned}"}
                        ],
                        temperature=temp,
                    ).choices[0].message.content.strip()
                    logger.info("Raw GPT response (attempt %d): %s", attempt + 1, response[:500] + "..." if len(response) > 500 else response)
                    response = re.sub(r'^```json\s*\n?', '', response)
                    response = re.sub(r'\n?```$', '', response).strip()
                    parsed = json.loads(response)
                    required_fields = ["verdict", "claim_summary", "scores", "reasoning"]
                    if not all(field in parsed for field in required_fields):
                        raise ValueError("Missing required JSON fields")
                    if not all(key in parsed["scores"] for key in ["logic", "natural_law", "historical_accuracy", "source_credibility", "overall_reasonableness"]):
                        raise ValueError("Missing required score fields")
                    # Convert string scores to integers
                    for key in parsed["scores"]:
                        try:
                            parsed["scores"][key] = int(parsed["scores"][key])
                        except (ValueError, TypeError):
                            logger.warning("Invalid score value for %s: %s, setting to 0", key, parsed["scores"][key])
                            parsed["scores"][key] = 0
                    return parsed
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("JSON parse failed on attempt %d: %s", attempt + 1, str(e))
                    if attempt < retries:
                        prompt += "\nOutput ONLY a valid JSON object, no fences, no extra text."
                    else:
                        logger.error("All retries failed for JSON parsing")
                        return {
                            "error": f"Failed to parse JSON after {retries + 1} attempts: {str(e)}",
                            "verdict": "Unknown",
                            "claim_summary": "Analysis failed due to formatting error",
                            "scores": {
                                "logic": 0,
                                "natural_law": 0,
                                "historical_accuracy": 0,
                                "source_credibility": 0,
                                "overall_reasonableness": 0
                            },
                            "reasoning": {},
                            "grounding_meter": "",
                            "emotion_meter": "",
                            "ai_origin": "",
                            "detected_style": "",
                            "relevant_sources": [],
                            "suggested_research": [],
                            "final_commentary": "",
                            "confidence_level": 0,
                            "truth_drift_score": 0,
                            "claim_length": len(cleaned.split()),
                            "temporal_reference": ""
                        }
        return ask(sys_prompt, brutality_mode)
    except Exception as e:
        logger.error("Scoring error: %s", str(e), exc_info=True)
        return {
            "error": f"Unable to score claim: {str(e)}",
            "verdict": "Unknown",
            "claim_summary": "Analysis failed due to an issue",
            "scores": {
                "logic": 0,
                "natural_law": 0,
                "historical_accuracy": 0,
                "source_credibility": 0,
                "overall_reasonableness": 0
            },
            "reasoning": {},
            "grounding_meter": "",
            "emotion_meter": "",
            "ai_origin": "",
            "detected_style": "",
            "relevant_sources": [],
            "suggested_research": [],
            "final_commentary": "",
            "confidence_level": 0,
            "truth_drift_score": 0,
            "claim_length": len(cleaned.split()) if 'cleaned' in locals() else 0,
            "temporal_reference": ""
        }

