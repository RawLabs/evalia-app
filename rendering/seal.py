"""Seal rendering utilities."""
from PIL import Image, ImageDraw, ImageFont
import io

def render_evalia_seal(verdict_text: str, brutality_mode: bool, logo_path: str = None) -> bytes:
    W, H = 400, 300
    bg_color = (139, 0, 0) if brutality_mode else (75, 75, 75)
    text_color = (234, 234, 234)
    accent_text = (126, 200, 255)
    img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        logo = Image.open(logo_path).convert("RGBA").resize((70, 70))
        lx, ly = W - 80, H - 80
        img.alpha_composite(logo, (lx, ly))
    except Exception:
        pass

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 25)
        verdict_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        title_font = verdict_font = small_font = ImageFont.load_default()

    draw.text((W // 2, 37), "Seal of Passage", font=title_font, fill=text_color, anchor="mm")
    max_text_w = int(W * 0.72)
    lines = []
    words = verdict_text.split()
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=verdict_font)
        if bbox[2] - bbox[0] <= max_text_w:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    wrapped = '\n'.join(lines)
    draw.multiline_text((W // 2, 195), wrapped, font=verdict_font, fill=text_color, anchor="mm", align="center", spacing=4)
    draw.text((W // 2, H - 5), "Evalia", font=small_font, fill=accent_text, anchor="mm")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
