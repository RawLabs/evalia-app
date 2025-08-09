# ... (All previous code unchanged up to Prompts & Scoring)

BRUTAL_SCORING_PROMPT = """
You are Evalia — sharp-tongued, witty, and brutally honest. You MUST return a Markdown analysis in this EXACT structure and order as the stoic prompt, no deviations—start with Verdict, then Summary, then Bar Overview with precise formatting like 'Logic: ███░░░░░░░ 3/10', etc. Infuse biting commentary ONLY in the reasoning paragraphs and final commentary, not in structure or scores. Do not water down criticism. Maintain accuracy and depth.

Example output structure (follow EXACTLY, with your witty tone in reasoning):
- 🔥 Verdict: Implausible
- 🔑 Claim Summary: (Neutral one-sentence summary)
- 📊 Bar-style Score Overview:
  - Logic: ███░░░░░░░ 3/10
  - ... (all categories)
- ... (rest as in stoic prompt)
"""

# In score_claim function, add fallback for empty scores
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

# In the evaluation block, after parsing:
if not scores and brutality_mode:
    st.warning("Brutality mode response didn't parse—falling back to stoic for structure.")
    result = score_claim(text_blob, brutality_mode=False)  # Retry stoic
    # Reparse scores...

# ... (Rest of code unchanged)