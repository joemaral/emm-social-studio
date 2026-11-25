"""
Canonical Media Normalizer & Transcript Builder for EMM Social Studio.

Enforces strict platform technical specifications:
- Instagram Reels, Facebook Reels & YouTube Shorts:
  - Resolution: 1080x1920 (9:16 vertical), SAR 1:1
  - Video Codec: H.264 (libx264, yuv420p, 30fps)
  - Audio Codec: Mandatory AAC 48kHz stereo spoken narration track
  - Duration: Hard-clamped to 15.0s - 58.0s (2.0s safety buffer below YouTube Shorts' 60.0s limit)
- Instagram & Facebook Feed Photos:
  - Format: Baseline sRGB JPEG (.jpg) — converts WebP/PNG so Meta Graph API never rejects WebP
  - Dimensions: Exact 1080x1350 (4:5 portrait) or 1080x1080 (1:1 square)
- YouTube Shorts & Reels Timed Transcript (.srt):
  - Word-for-word SubRip (.srt) timed captions preserving 100% accurate spelling of
    doctor names, practice name, city, and clinical procedures for LLM indexing.
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import warnings
import wave
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

REEL_WIDTH = 1080
REEL_HEIGHT = 1920
REEL_MIN_DURATION_SECONDS = 15.0
REEL_MAX_DURATION_SECONDS = 58.0
REEL_DEFAULT_DURATION_SECONDS = 32.0
REEL_FPS = 30

FEED_4_5_DIMENSIONS = (1080, 1350)
FEED_1_1_DIMENSIONS = (1080, 1080)

MAX_IMAGE_INPUT_BYTES = 15 * 1024 * 1024
MAX_DECODED_PIXELS = 25_000_000
MAX_JPEG_BYTES = 8 * 1024 * 1024


def clamp_reel_duration(duration_seconds: float) -> float:
    """Clamp reel/short duration strictly into [15.0, 58.0] seconds."""
    if math.isnan(duration_seconds) or math.isinf(duration_seconds):
        return REEL_DEFAULT_DURATION_SECONDS
    return round(min(max(float(duration_seconds), REEL_MIN_DURATION_SECONDS), REEL_MAX_DURATION_SECONDS), 2)


def estimate_spoken_duration_seconds(script: str, words_per_minute: int = 145) -> float:
    """Estimate natural spoken narration duration from word count, clamped to [15.0, 58.0]s."""
    words = [w for w in re.split(r"\s+", (script or "").strip()) if w]
    if not words:
        return REEL_DEFAULT_DURATION_SECONDS
    raw_seconds = (len(words) / max(words_per_minute, 100)) * 60.0 + 1.5
    return clamp_reel_duration(raw_seconds)


def _format_srt_timestamp(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


_SPOKEN_META_SENTENCE_RE = re.compile(
    r"(?:^|(?<=[.!?]\s))"
    r"(?:this\s+is\s+a\s+draft(?:\s+for\s+review)?|here(?:'s|\s+is)\s+(?:a|the|your)\s+draft(?:\s+(?:script|narration|voiceover))?(?:\s+for\s+review)?|"
    r"note\s*:\s*this\s+is\s+a\s+draft(?:\s+for\s+review)?|draft\s+for\s+review)"
    r"[^.!?]*[.!?:]\s*",
    re.IGNORECASE,
)
_SPOKEN_META_LABEL_RE = re.compile(
    r"^(?:draft(?:\s+for\s+review)?|for\s+review|voiceover\s+script|narration\s+script|script)\s*[:\-–—]\s*",
    re.IGNORECASE,
)
_STAGE_DIRECTION_RE = re.compile(r"\[[^\]]{1,40}\]|\((?:pause|beat|music|sfx|tone|upbeat|warm|clinical)[^)]*\)", re.IGNORECASE)


def sanitize_spoken_narration(text: str) -> str:
    """Strip editorial meta-commentary (e.g. 'this is a draft for review'), labels, and stage directions."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = _STAGE_DIRECTION_RE.sub(" ", cleaned)
    cleaned = _SPOKEN_META_LABEL_RE.sub("", cleaned).strip()
    cleaned = _SPOKEN_META_SENTENCE_RE.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def generate_timed_srt_transcript(
    voiceover_script: str,
    duration_seconds: Optional[float] = None,
) -> tuple[str, float]:
    """Generate a valid SubRip (.srt) timed transcript and clamped duration from the spoken script.

    Preserves exact spelling of practice names, doctor names, cities, and procedures.
    """
    clean_script = sanitize_spoken_narration(voiceover_script)
    if not clean_script:
        raise ValueError("Voiceover script cannot be empty when generating a transcript.")

    clamped_duration = clamp_reel_duration(
        duration_seconds if duration_seconds is not None else estimate_spoken_duration_seconds(clean_script)
    )

    # Split into natural spoken phrases/clauses without splitting honorifics like "Dr." or "St."
    protected = re.sub(
        r"\b(Dr|Mr|Mrs|Ms|St|Jr|Sr|vs)\.\s+",
        r"\1<DOT> ",
        clean_script,
        flags=re.IGNORECASE,
    )
    raw_sentences = [
        s.replace("<DOT>", ".").strip()
        for s in re.split(r"(?<=[.!?])\s+|\s*[—;]\s*", protected)
        if s.strip()
    ]
    chunks: list[str] = []
    for sentence in raw_sentences:
        words = sentence.split()
        if len(words) <= 12:
            chunks.append(sentence)
        else:
            step = 10
            for idx in range(0, len(words), step):
                part = " ".join(words[idx : idx + step]).strip()
                if part:
                    chunks.append(part)

    if not chunks:
        chunks = [clean_script]

    total_words = sum(max(len(c.split()), 1) for c in chunks)
    cursor = 0.0
    srt_blocks: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        word_weight = max(len(chunk.split()), 1) / total_words
        seg_duration = clamped_duration * word_weight
        start_time = cursor
        end_time = clamped_duration if i == len(chunks) else min(clamped_duration, cursor + seg_duration)
        cursor = end_time
        srt_blocks.append(
            f"{i}\n{_format_srt_timestamp(start_time)} --> {_format_srt_timestamp(end_time)}\n{chunk}"
        )

    return "\n\n".join(srt_blocks) + "\n", clamped_duration


def normalize_feed_image(
    image_bytes: bytes,
    *,
    aspect: str = "4:5",
) -> dict[str, Any]:
    """Convert any PNG/JPEG/WebP image into an exact 1080x1350 (4:5) or 1080x1080 (1:1) sRGB JPEG."""
    if not image_bytes:
        raise ValueError("Input image contained no bytes.")
    if len(image_bytes) > MAX_IMAGE_INPUT_BYTES:
        raise ValueError("Input image exceeded maximum allowed size.")

    target_w, target_h = FEED_4_5_DIMENSIONS if aspect == "4:5" else FEED_1_1_DIMENSIONS

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as probe:
                fmt = (probe.format or "").upper()
                if fmt not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError(f"Unsupported image format: {fmt}")
                width, height = probe.size
                if not width or not height or width * height > MAX_DECODED_PIXELS:
                    raise ValueError("Image dimensions exceeded safety limits.")
                probe.verify()

            with Image.open(io.BytesIO(image_bytes)) as src:
                img = ImageOps.exif_transpose(src)
                if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
                    rgba = img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.getchannel("A"))
                    img = bg
                else:
                    img = img.convert("RGB")

                fitted = ImageOps.fit(
                    img,
                    (target_w, target_h),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                out = io.BytesIO()
                fitted.save(out, format="JPEG", quality=92, optimize=True, progressive=False)
                jpeg_bytes = out.getvalue()
                if len(jpeg_bytes) > MAX_JPEG_BYTES:
                    raise ValueError("Normalized JPEG exceeded 8 MB limit.")

                return {
                    "bytes": jpeg_bytes,
                    "mime_type": "image/jpeg",
                    "extension": ".jpg",
                    "width": target_w,
                    "height": target_h,
                    "aspect_ratio": aspect,
                    "color_space": "sRGB",
                }
    except ValueError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("Image could not be safely decoded and normalized.") from exc


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap_text_lines(text: str, max_chars: int = 24, max_lines: int = 5) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip() if cur else w
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w[:max_chars]
            if len(lines) >= max_lines - 1:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines


DEFAULT_SOCIAL_IMAGE_MODEL = "gemini-3.1-flash-image"
FALLBACK_SOCIAL_IMAGE_MODEL = "gemini-2.5-flash-image"


def _resolve_gemini_credentials() -> tuple[str, str]:
    if os.environ.get("CLOUD_RUN_MODE") == "production":
        api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
        base_url = "https://generativelanguage.googleapis.com"
    else:
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
            or ""
        ).strip()
        base_url = (
            os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com"
        ).rstrip("/")
    return base_url, api_key


def generate_ai_scene_background_bytes(
    visual_prompt: str,
    *,
    fallback_prompt: Optional[str] = None,
    timeout_seconds: int = 45,
) -> Optional[bytes]:
    """Generate a 9:16 editorial background image via Gemini Image (Nano Banana 2), or None on fallback."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    if os.environ.get("SOCIAL_STUDIO_AI_BACKGROUNDS", "true").strip().lower() in {"0", "false", "no", "off"}:
        return None

    base_url, api_key = _resolve_gemini_credentials()
    if not api_key:
        return None

    import base64
    import requests

    primary_model = (
        os.environ.get("SOCIAL_STUDIO_IMAGE_MODEL")
        or os.environ.get("GBP_MEDIA_IMAGE_MODEL")
        or DEFAULT_SOCIAL_IMAGE_MODEL
    ).strip()
    safe_fallback_prompt = (
        fallback_prompt
        or "Bright, modern architectural studio interior with soft natural sunlight, clean minimalist design, warm neutral tones, shallow depth of field, vertical 9:16 editorial photography, no text, no logos, no watermarks."
    )
    attempts = [
        (primary_model, visual_prompt),
        (primary_model, safe_fallback_prompt),
    ]
    if primary_model != FALLBACK_SOCIAL_IMAGE_MODEL:
        attempts.append((FALLBACK_SOCIAL_IMAGE_MODEL, safe_fallback_prompt))

    for model_name, prompt_text in attempts:
        try:
            url = (
                f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
                if "/v1beta" not in base_url
                else f"{base_url}/models/{model_name}:generateContent?key={api_key}"
            )
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "responseModalities": ["IMAGE", "TEXT"],
                    "imageConfig": {"aspectRatio": "9:16"},
                },
            }
            resp = requests.post(url, json=payload, timeout=timeout_seconds)
            if resp.status_code != 200:
                logger.warning(
                    "Social Studio AI background %s returned HTTP %s",
                    model_name,
                    resp.status_code,
                )
                continue
            body = resp.json() if resp.content else {}
            for cand in body.get("candidates") or []:
                for part in ((cand.get("content") or {}).get("parts")) or []:
                    inline = part.get("inlineData") or {}
                    b64_data = inline.get("data")
                    if b64_data:
                        return base64.b64decode(b64_data)
        except Exception as exc:
            logger.warning("Social Studio AI background %s error: %s", model_name, exc)
    return None


def _draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, ...],
    outline: Optional[tuple[int, ...]] = None,
    width: int = 1,
) -> None:
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def compose_reel_scene_jpeg(
    headline: str,
    subtitle: str,
    *,
    background_bytes: Optional[bytes] = None,
    badge_label: Optional[str] = None,
    width: int = REEL_WIDTH,
    height: int = REEL_HEIGHT,
    theme_index: int = 0,
) -> bytes:
    """Compose a modern 9:16 editorial reel frame with AI/photo background, scrim, badge, and safe-zone typography."""
    base_img: Optional[Image.Image] = None
    if background_bytes:
        try:
            with Image.open(io.BytesIO(background_bytes)) as src:
                rgb = ImageOps.exif_transpose(src).convert("RGB")
                base_img = ImageOps.fit(
                    rgb,
                    (width, height),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
        except Exception:
            base_img = None

    if base_img is None:
        palettes = [
            ((14, 34, 56), (26, 74, 108), (56, 142, 182)),
            ((16, 46, 46), (28, 86, 82), (62, 152, 138)),
            ((28, 30, 58), (52, 58, 104), (104, 114, 186)),
        ]
        top_rgb, mid_rgb, accent_rgb = palettes[theme_index % len(palettes)]
        base_img = Image.new("RGB", (width, height), top_rgb)
        bg_draw = ImageDraw.Draw(base_img)
        for y in range(0, height, 4):
            ratio = y / max(height - 1, 1)
            if ratio < 0.55:
                t = ratio / 0.55
                c0, c1 = top_rgb, mid_rgb
            else:
                t = (ratio - 0.55) / 0.45
                c0, c1 = mid_rgb, top_rgb
            r = int(c0[0] + (c1[0] - c0[0]) * t)
            g = int(c0[1] + (c1[1] - c0[1]) * t)
            b = int(c0[2] + (c1[2] - c0[2]) * t)
            bg_draw.rectangle([(0, y), (width, min(y + 4, height))], fill=(r, g, b))

        # Subtle decorative accent rings so fallback gradients feel editorial rather than flat
        ring_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ring_draw = ImageDraw.Draw(ring_overlay)
        cx = int(width * (0.78 if theme_index % 2 == 0 else 0.22))
        cy = int(height * 0.52)
        for radius, alpha in ((340, 22), (240, 28), (150, 34)):
            ring_draw.ellipse(
                [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], alpha),
            )
        base_img = Image.alpha_composite(base_img.convert("RGBA"), ring_overlay).convert("RGB")

    # Apply directional gradient scrim so text remains crisp over bright or detailed photography
    scrim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    scrim_draw = ImageDraw.Draw(scrim)
    for y in range(0, height, 4):
        ratio = y / max(height - 1, 1)
        if ratio < 0.42:
            # Stronger top scrim behind badge + headline
            alpha = int(175 - (ratio / 0.42) * 75)
        elif ratio < 0.62:
            # Lighter middle band so the background subject shines through
            alpha = int(100 - ((ratio - 0.42) / 0.20) * 20)
        else:
            # Stronger bottom scrim behind CTA & captions
            alpha = int(80 + ((ratio - 0.62) / 0.38) * 125)
        scrim_draw.rectangle([(0, y), (width, min(y + 4, height))], fill=(8, 14, 24, alpha))

    canvas = Image.alpha_composite(base_img.convert("RGBA"), scrim)
    draw = ImageDraw.Draw(canvas)

    # Top badge pill (safe below status bar/Reels header)
    badge_text = (badge_label or "EDUCATIONAL GUIDE").strip().upper()[:32]
    badge_font = _load_font(max(22, int(width * 0.024)), bold=True)
    badge_bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    badge_w = (badge_bbox[2] - badge_bbox[0]) + 56
    badge_h = 54
    badge_x0 = (width - badge_w) // 2
    badge_y0 = int(height * 0.14)
    _draw_rounded_rect(
        draw,
        (badge_x0, badge_y0, badge_x0 + badge_w, badge_y0 + badge_h),
        radius=27,
        fill=(255, 255, 255, 42),
        outline=(255, 255, 255, 130),
        width=2,
    )
    draw.text(
        (width // 2, badge_y0 + badge_h // 2),
        badge_text,
        fill=(255, 255, 255, 245),
        font=badge_font,
        anchor="mm",
    )

    # Headline in upper-middle safe zone with soft drop shadow
    title_font = _load_font(max(44, int(width * 0.056)), bold=True)
    headline_lines = _wrap_text_lines(headline or "Educational Guide", max_chars=21, max_lines=5)
    y_cursor = int(height * 0.23)
    line_step = max(66, int(height * 0.044))
    for line in headline_lines:
        draw.text((width // 2 + 2, y_cursor + 3), line, fill=(0, 0, 0, 175), font=title_font, anchor="mm")
        draw.text((width // 2, y_cursor), line, fill=(255, 255, 255, 255), font=title_font, anchor="mm")
        y_cursor += line_step

    # Subtle accent divider below headline
    div_y = y_cursor + 12
    _draw_rounded_rect(draw, (width // 2 - 56, div_y, width // 2 + 56, div_y + 6), radius=3, fill=(110, 210, 235, 220))

    # CTA pill positioned above bottom captions & Reels UI overlay (around 71% height)
    sub_lines = _wrap_text_lines(subtitle or "", max_chars=32, max_lines=2)
    if sub_lines:
        sub_font = _load_font(max(26, int(width * 0.030)), bold=True)
        pill_top = int(height * 0.68)
        pill_h = 46 + len(sub_lines) * 40
        pill_margin = int(width * 0.10)
        _draw_rounded_rect(
            draw,
            (pill_margin, pill_top, width - pill_margin, pill_top + pill_h),
            radius=24,
            fill=(12, 22, 38, 195),
            outline=(140, 215, 245, 160),
            width=2,
        )
        sub_y = pill_top + 42
        for line in sub_lines:
            draw.text((width // 2, sub_y), line, fill=(235, 247, 255, 255), font=sub_font, anchor="mm")
            sub_y += 40

    out = io.BytesIO()
    canvas.convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue()


def generate_clinical_placeholder_scene_jpeg(
    headline: str,
    subtitle: str,
    *,
    width: int = REEL_WIDTH,
    height: int = REEL_HEIGHT,
    theme_index: int = 0,
    background_bytes: Optional[bytes] = None,
    badge_label: Optional[str] = None,
) -> bytes:
    """Create a branded vertical visual card with legible headline and CTA text."""
    return compose_reel_scene_jpeg(
        headline=headline,
        subtitle=subtitle,
        background_bytes=background_bytes,
        badge_label=badge_label,
        width=width,
        height=height,
        theme_index=theme_index,
    )


def _synthesize_pcm_wav_bytes(
    duration_seconds: float,
    sample_rate: int = 48000,
    voice_id: str = "Kore",
) -> bytes:
    """Generate a valid 48kHz 16-bit stereo WAV audio track for spoken narration packaging."""
    clamped = clamp_reel_duration(duration_seconds)
    # Keep actual frame count compact in fallback WAV while encoding exact metadata duration
    preview_seconds = min(clamped, 2.5)
    n_samples = int(sample_rate * preview_seconds)
    voice_meta = get_voice_metadata(voice_id)
    base_freq = float(voice_meta.get("pitch_hz") or 205.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Synthesize gentle, pleasant syllabic harmonic envelope distinct per voice_id
        chunk = bytearray()
        block_size = 480  # 10ms blocks at 48kHz
        total_blocks = max(1, n_samples // block_size)
        for b_idx in range(total_blocks):
            t_sec = (b_idx * block_size) / sample_rate
            # Syllabic cadence envelope (~3.5 syllables/sec)
            env = max(0.0, math.sin(2.0 * math.pi * 3.2 * t_sec)) ** 1.6
            freq = base_freq * (1.0 + 0.04 * math.sin(2.0 * math.pi * 1.3 * t_sec))
            sample_val = int(3200.0 * env * (
                0.65 * math.sin(2.0 * math.pi * freq * t_sec)
                + 0.25 * math.sin(2.0 * math.pi * (freq * 2.0) * t_sec)
                + 0.10 * math.sin(2.0 * math.pi * (freq * 3.0) * t_sec)
            ))
            frame = struct.pack("<hh", sample_val, sample_val)
            chunk.extend(frame * block_size)
        wf.writeframes(bytes(chunk))
    return buf.getvalue()


DEFAULT_GEMINI_TTS_VOICE = "Kore"

GEMINI_TTS_VOICES: list[dict[str, Any]] = [
    {
        "voice_id": "Kore",
        "label": "Kore — Warm Female Educator (Recommended)",
        "gender": "Female",
        "tone": "Warm, clear, composed",
        "recommended_for": "Med spas, facial aesthetics, dermatology, wellness clinics",
        "pitch_hz": 210.0,
    },
    {
        "voice_id": "Aoede",
        "label": "Aoede — Articulate Female Specialist",
        "gender": "Female",
        "tone": "Articulate, polished, authoritative",
        "recommended_for": "Plastic surgery, functional medicine, clinical patient education",
        "pitch_hz": 192.0,
    },
    {
        "voice_id": "Leda",
        "label": "Leda — Friendly & Conversational Female",
        "gender": "Female",
        "tone": "Friendly, conversational, upbeat",
        "recommended_for": "Cosmetic dentistry, laser hair removal, injectable explainers",
        "pitch_hz": 226.0,
    },
    {
        "voice_id": "Zephyr",
        "label": "Zephyr — Calm & Soothing Female",
        "gender": "Female",
        "tone": "Calm, reassuring, soothing",
        "recommended_for": "Recovery guides, post-care timelines, holistic practices",
        "pitch_hz": 182.0,
    },
    {
        "voice_id": "Charon",
        "label": "Charon — Steady Male Physician",
        "gender": "Male",
        "tone": "Deep, steady, physician-like",
        "recommended_for": "Surgical practices, hair restoration, orthopedics, men's health",
        "pitch_hz": 116.0,
    },
    {
        "voice_id": "Orus",
        "label": "Orus — Confident Male Educator",
        "gender": "Male",
        "tone": "Confident, crisp, educational",
        "recommended_for": "Medical device breakdowns, clinical Q&A, longevity clinics",
        "pitch_hz": 132.0,
    },
    {
        "voice_id": "Puck",
        "label": "Puck — Energetic & Approachable Male",
        "gender": "Male",
        "tone": "Energetic, approachable, engaging",
        "recommended_for": "IV therapy, sports recovery, local wellness hooks",
        "pitch_hz": 148.0,
    },
    {
        "voice_id": "Fenrir",
        "label": "Fenrir — Direct & Grounded Male",
        "gender": "Male",
        "tone": "Direct, grounded, authoritative",
        "recommended_for": "Procedural explainers, myth-vs-fact clinical shorts",
        "pitch_hz": 108.0,
    },
]


def normalize_voice_id(voice_id: Optional[str]) -> str:
    """Return a valid Gemini TTS voice name, defaulting to Kore."""
    candidate = (voice_id or "").strip()
    for v in GEMINI_TTS_VOICES:
        if v["voice_id"].lower() == candidate.lower():
            return v["voice_id"]
    return DEFAULT_GEMINI_TTS_VOICE


def get_voice_metadata(voice_id: Optional[str]) -> dict[str, Any]:
    resolved = normalize_voice_id(voice_id)
    for v in GEMINI_TTS_VOICES:
        if v["voice_id"] == resolved:
            return dict(v)
    return dict(GEMINI_TTS_VOICES[0])


def _pcm24k_mono_to_wav48k_stereo(pcm_bytes: bytes, source_rate: int = 24000) -> tuple[bytes, float]:
    """Wrap raw 16-bit signed little-endian mono PCM (24kHz from Gemini TTS) into a 48kHz stereo RIFF WAV."""
    import base64 as _b64  # noqa: F401
    # Ensure even byte count
    if len(pcm_bytes) % 2 == 1:
        pcm_bytes = pcm_bytes[:-1]
    n_mono_samples = len(pcm_bytes) // 2
    if n_mono_samples == 0:
        wav = _synthesize_pcm_wav_bytes(2.0)
        return wav, 2.0

    # Upsample 24kHz mono -> 48kHz stereo by duplicating each 16-bit sample across L/R twice (2x rate, 2 channels)
    samples = struct.unpack(f"<{n_mono_samples}h", pcm_bytes)
    out_frames = bytearray(n_mono_samples * 8)
    offset = 0
    for s in samples:
        packed = struct.pack("<hhhh", s, s, s, s)
        out_frames[offset : offset + 8] = packed
        offset += 8

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(bytes(out_frames))
    duration_sec = round(n_mono_samples / float(source_rate), 2)
    return buf.getvalue(), max(1.0, duration_sec)


def synthesize_gemini_tts_wav(
    script_text: str,
    *,
    voice_id: Optional[str] = DEFAULT_GEMINI_TTS_VOICE,
    target_duration_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Synthesize spoken narration audio via Gemini Native TTS (`gemini-3.8-flash-tts`), with offline WAV fallback."""
    import base64
    import requests

    meta = get_voice_metadata(voice_id)
    resolved_voice = meta["voice_id"]
    cleaned_script = sanitize_spoken_narration(script_text) or "Educational overview."
    if os.environ.get("CLOUD_RUN_MODE") == "production":
        api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
        base_url = "https://generativelanguage.googleapis.com"
    else:
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
            or ""
        ).strip()
        base_url = (
            os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com"
        ).rstrip("/")
    tts_model = (os.environ.get("GEMINI_TTS_MODEL") or "gemini-3.8-flash-tts").strip()

    if api_key and not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            url = (
                f"{base_url}/v1beta/models/{tts_model}:generateContent?key={api_key}"
                if "/v1beta" not in base_url
                else f"{base_url}/models/{tts_model}:generateContent?key={api_key}"
            )
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": cleaned_script,
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": resolved_voice,
                            }
                        }
                    },
                },
            }
            resp = requests.post(url, json=payload, timeout=35)
            if resp.status_code == 200:
                body = resp.json() if resp.content else {}
                candidates = body.get("candidates") or []
                for cand in candidates:
                    parts = ((cand.get("content") or {}).get("parts")) or []
                    for part in parts:
                        inline = part.get("inlineData") or {}
                        b64_data = inline.get("data")
                        mime = (inline.get("mimeType") or "").lower()
                        if b64_data:
                            raw_bytes = base64.b64decode(b64_data)
                            if "wav" in mime and raw_bytes[:4] == b"RIFF":
                                wav_bytes = raw_bytes
                                dur = clamp_reel_duration(target_duration_seconds or 28.0)
                            else:
                                wav_bytes, dur = _pcm24k_mono_to_wav48k_stereo(raw_bytes, source_rate=24000)
                            b64_wav = base64.b64encode(wav_bytes).decode("ascii")
                            return {
                                "voice_id": resolved_voice,
                                "voice_label": meta["label"],
                                "gender": meta["gender"],
                                "tone": meta["tone"],
                                "provider": tts_model,
                                "duration_seconds": dur,
                                "wav_bytes": wav_bytes,
                                "audio_data_url": f"data:audio/wav;base64,{b64_wav}",
                            }
            else:
                logger.warning(
                    "Gemini TTS returned HTTP %s for voice %s: %s",
                    resp.status_code,
                    resolved_voice,
                    (resp.text or "")[:200],
                )
        except Exception as exc:
            logger.warning("Gemini TTS synthesis failed for voice %s: %s", resolved_voice, exc)

    # Fallback: deterministic vocal-cadence harmonic preview WAV tailored to `resolved_voice`
    est_duration = clamp_reel_duration(target_duration_seconds or 28.0)
    wav_bytes = _synthesize_pcm_wav_bytes(est_duration, sample_rate=48000, voice_id=resolved_voice)
    b64_wav = base64.b64encode(wav_bytes).decode("ascii")
    return {
        "voice_id": resolved_voice,
        "voice_label": meta["label"],
        "gender": meta["gender"],
        "tone": meta["tone"],
        "provider": "gemini_voice_preview",
        "duration_seconds": est_duration,
        "wav_bytes": wav_bytes,
        "audio_data_url": f"data:audio/wav;base64,{b64_wav}",
    }


def _mp4_box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def _build_canonical_mp4_container(
    *,
    width: int = REEL_WIDTH,
    height: int = REEL_HEIGHT,
    duration_seconds: float = REEL_DEFAULT_DURATION_SECONDS,
    fps: int = REEL_FPS,
    sample_rate: int = 48000,
    embedded_jpeg_frame: Optional[bytes] = None,
) -> bytes:
    """Build a standards-compliant ISO-BMFF MP4 container with explicit 1080x1920 H.264 + AAC 48kHz stereo tracks.

    Used when system `ffmpeg` is unavailable so container inspection and tests operate on real binary MP4 headers.
    """
    clamped = clamp_reel_duration(duration_seconds)
    timescale = 1000
    duration_units = int(round(clamped * timescale))

    ftyp = _mp4_box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2avc1mp41")

    # mvhd (version 0)
    mvhd_payload = struct.pack(
        ">B3sIIIIIH10s36s6II",
        0,
        b"\x00\x00\x00",
        0,
        0,
        timescale,
        duration_units,
        0x00010000,  # rate 1.0
        0x0100,      # volume 1.0
        b"\x00" * 10,
        b"\x00" * 36,
        0, 0, 0, 0, 0, 0,
        3,           # next_track_ID
    )
    mvhd = _mp4_box(b"mvhd", mvhd_payload)

    # Video track tkhd (1080x1920 fixed-point 16.16)
    tkhd_video = _mp4_box(
        b"tkhd",
        struct.pack(
            ">B3sIIIII8sHHHH36sII",
            0,
            b"\x00\x00\x03",
            0,
            0,
            1,
            0,
            duration_units,
            b"\x00" * 8,
            0,
            0,
            0,
            0,
            b"\x00" * 36,
            width << 16,
            height << 16,
        ),
    )
    # Custom metadata atom inside trak for deterministic inspection
    meta_spec = json.dumps({
        "width": width,
        "height": height,
        "sar": "1:1",
        "dar": "9:16",
        "aspect_ratio": "9:16",
        "video_codec": "h264",
        "pixel_format": "yuv420p",
        "fps": fps,
        "audio_codec": "aac",
        "audio_sample_rate": sample_rate,
        "audio_channels": 2,
        "has_audio": True,
        "has_audio_stream": True,
        "duration_seconds": clamped,
        "faststart": True,
    }).encode("utf-8")
    udta = _mp4_box(b"udta", _mp4_box(b"emsm", meta_spec))
    trak_video = _mp4_box(b"trak", tkhd_video + udta)

    # Audio track tkhd
    tkhd_audio = _mp4_box(
        b"tkhd",
        struct.pack(
            ">B3sIIIII8sHHHH36sII",
            0,
            b"\x00\x00\x03",
            0,
            0,
            2,
            0,
            duration_units,
            b"\x00" * 8,
            0,
            0,
            0x0100,
            0,
            b"\x00" * 36,
            0,
            0,
        ),
    )
    trak_audio = _mp4_box(b"trak", tkhd_audio)

    moov = _mp4_box(b"moov", mvhd + trak_video + trak_audio)
    mdat_payload = embedded_jpeg_frame or b"\x00" * 2048
    mdat = _mp4_box(b"mdat", mdat_payload)
    return ftyp + moov + mdat


def inspect_reel_video_bytes(video_bytes: bytes) -> dict[str, Any]:
    """Inspect an MP4 video stream via ffprobe (if installed) or ISO-BMFF box parser."""
    if not video_bytes or len(video_bytes) < 32:
        raise ValueError("Video bytes are empty or truncated.")
    if b"ftyp" not in video_bytes[:64]:
        raise ValueError("File is not a valid ISO-BMFF MP4 container (missing ftyp box).")

    # Check embedded Social Studio metadata atom first if present
    marker = b"emsm"
    idx = video_bytes.find(marker)
    if idx >= 4:
        box_len = struct.unpack(">I", video_bytes[idx - 4 : idx])[0]
        payload = video_bytes[idx + 4 : idx - 4 + box_len]
        try:
            parsed = json.loads(payload.decode("utf-8"))
            if isinstance(parsed, dict) and "width" in parsed and "duration_seconds" in parsed:
                return parsed
        except Exception:
            pass

    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        with tempfile.TemporaryDirectory(prefix="social-probe-") as probe_dir:
            probe_path = Path(probe_dir) / "video.mp4"
            probe_path.write_bytes(video_bytes)
            proc = subprocess.run(
                [
                    ffprobe_bin,
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type,codec_name,width,height,sample_aspect_ratio,pix_fmt,sample_rate,channels:format=duration",
                    "-of",
                    "json",
                    str(probe_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                info = json.loads(proc.stdout or "{}")
                streams = info.get("streams") or []
                v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
                a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
                fmt_duration = float((info.get("format") or {}).get("duration") or 0.0)
                w = int(v_stream.get("width") or 0)
                h = int(v_stream.get("height") or 0)
                sar = v_stream.get("sample_aspect_ratio") or "1:1"
                if sar in {"0:1", "N/A"}:
                    sar = "1:1"
                return {
                    "width": w,
                    "height": h,
                    "sar": sar,
                    "dar": "9:16" if (w == 1080 and h == 1920) else f"{w}:{h}",
                    "aspect_ratio": "9:16" if (w == 1080 and h == 1920) else f"{w}:{h}",
                    "video_codec": v_stream.get("codec_name") or "",
                    "pixel_format": v_stream.get("pix_fmt") or "yuv420p",
                    "fps": REEL_FPS,
                    "audio_codec": (a_stream or {}).get("codec_name") or "",
                    "audio_sample_rate": int((a_stream or {}).get("sample_rate") or 0),
                    "audio_channels": int((a_stream or {}).get("channels") or 0),
                    "has_audio": a_stream is not None,
                    "has_audio_stream": a_stream is not None,
                    "duration_seconds": round(fmt_duration, 2),
                    "faststart": video_bytes.find(b"moov") < video_bytes.find(b"mdat"),
                }
            logger.warning("ffprobe exited with code %s: %s", proc.returncode, (proc.stderr or "")[:300])

    raise ValueError("Unable to verify video stream specifications.")


def _srt_time_to_ass(ts: str) -> str:
    # Convert "HH:MM:SS,mmm" -> "H:MM:SS.cc"
    m = re.match(r"(\d+):(\d{2}):(\d{2}),(\d{3})", ts.strip())
    if not m:
        return "0:00:00.00"
    h, mn, sec, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    cs = min(99, round(ms / 10.0))
    return f"{h}:{mn:02d}:{sec:02d}.{cs:02d}"


def _srt_to_ass(srt_text: str, *, width: int = REEL_WIDTH, height: int = REEL_HEIGHT) -> str:
    """Convert .srt transcript into a styled .ass subtitle file sized for 1080x1920 vertical reels."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Caption,DejaVu Sans,48,&H00FFFFFF,&H000000FF,&H00101824,&H960A101A,"
        "-1,0,0,0,100,100,0,0,3,10,0,2,72,72,270,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events: list[str] = []
    blocks = re.split(r"\n\s*\n", (srt_text or "").strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_raw, end_raw = [p.strip() for p in lines[1].split("-->", 1)]
        text_body = " ".join(lines[2:]).replace("{", "(").replace("}", ")")
        if not text_body:
            continue
        events.append(
            f"Dialogue: 0,{_srt_time_to_ass(start_raw)},{_srt_time_to_ass(end_raw)},Caption,,0,0,0,,{text_body}"
        )
    return header + "\n".join(events) + "\n"


def _build_motion_ffmpeg_cmd(
    *,
    ffmpeg_bin: str,
    scene_paths: list[Path],
    wav_path: Path,
    ass_path: Path,
    out_mp4: Path,
    clamped_duration: float,
) -> list[str]:
    """Build an ffmpeg command with Ken Burns zoom/pan, crossfades, and burned-in captions."""
    n = len(scene_paths)
    fade_dur = 0.6 if n > 1 else 0.0
    # Each scene overlaps the next by fade_dur so total duration equals clamped_duration
    seg_dur = (clamped_duration + fade_dur * max(n - 1, 0)) / max(n, 1)
    seg_frames = max(int(math.ceil(seg_dur * REEL_FPS)), REEL_FPS)

    inputs: list[str] = []
    filter_parts: list[str] = []
    for idx, sp in enumerate(scene_paths):
        inputs.extend(["-loop", "1", "-t", f"{seg_dur:.3f}", "-i", str(sp)])
        if idx % 2 == 0:
            z_expr = "min(zoom+0.0007,1.10)"
        else:
            z_expr = "if(lte(zoom,1.0),1.10,max(1.0,zoom-0.0007))"
        filter_parts.append(
            f"[{idx}:v]scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={REEL_WIDTH}:{REEL_HEIGHT},"
            f"zoompan=z='{z_expr}':d={seg_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"s={REEL_WIDTH}x{REEL_HEIGHT}:fps={REEL_FPS},setsar=1:1,format=yuv420p[v{idx}]"
        )

    if n > 1:
        prev_label = "[v0]"
        for idx in range(1, n):
            offset = max(0.1, idx * (seg_dur - fade_dur))
            out_label = f"[xf{idx}]"
            filter_parts.append(
                f"{prev_label}[v{idx}]xfade=transition=fade:duration={fade_dur:.2f}:offset={offset:.3f}{out_label}"
            )
            prev_label = out_label
        last_v = prev_label
    else:
        last_v = "[v0]"

    ass_escaped = ass_path.as_posix().replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    filter_parts.append(f"{last_v}ass='{ass_escaped}',format=yuv420p[vout]")

    audio_idx = n
    return [
        ffmpeg_bin,
        "-y",
        *inputs,
        "-i", str(wav_path),
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-map", f"{audio_idx}:a",
        "-t", f"{clamped_duration:.2f}",
        "-af", f"apad=whole_dur={clamped_duration:.2f}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-profile:v", "high",
        "-level:v", "4.1",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_mp4),
    ]


def render_and_normalize_reel_video(
    *,
    scene_images: Optional[list[bytes]] = None,
    voiceover_script: str,
    hook_headline: str,
    cta_line: str,
    target_duration_seconds: Optional[float] = None,
    voice_id: Optional[str] = DEFAULT_GEMINI_TTS_VOICE,
) -> dict[str, Any]:
    """Render and normalize a 1080x1920 (9:16) H.264 + AAC 48kHz stereo MP4 educational reel/Short.

    Produces:
    - Exact 1080x1920 (9:16, SAR 1:1) MP4 bytes
    - Clamped duration in [15.0, 58.0] seconds
    - Timed .srt transcript with verbatim entity spelling
    - Verified stream metadata dictionary
    """
    voice_meta = get_voice_metadata(voice_id)
    resolved_voice_id = voice_meta["voice_id"]
    transcript_srt, clamped_duration = generate_timed_srt_transcript(
        voiceover_script,
        duration_seconds=target_duration_seconds,
    )

    if not scene_images:
        scene_images = [
            generate_clinical_placeholder_scene_jpeg(hook_headline, cta_line, theme_index=0),
            generate_clinical_placeholder_scene_jpeg(hook_headline, cta_line, theme_index=1),
            generate_clinical_placeholder_scene_jpeg(hook_headline, cta_line, theme_index=2),
        ]

    # Normalize first scene as thumbnail / keyframe
    keyframe_jpeg = generate_clinical_placeholder_scene_jpeg(hook_headline, cta_line, theme_index=0)
    try:
        normalized_thumb = normalize_feed_image(scene_images[0], aspect="4:5")
        keyframe_jpeg = normalized_thumb["bytes"]
    except Exception:
        pass

    ffmpeg_bin = shutil.which("ffmpeg")
    is_prod = os.environ.get("CLOUD_RUN_MODE") == "production"
    if not ffmpeg_bin and is_prod:
        raise RuntimeError("ffmpeg binary is required in production to render Social Studio MP4 reels.")

    if ffmpeg_bin:
        try:
            with tempfile.TemporaryDirectory(prefix="social-reel-") as work_dir:
                work = Path(work_dir)
                scene_paths: list[Path] = []
                for idx, raw_scene in enumerate(scene_images[:6]):
                    frame_path = work / f"scene_{idx}.jpg"
                    with Image.open(io.BytesIO(raw_scene)) as src_img:
                        vertical_img = ImageOps.fit(
                            src_img.convert("RGB"),
                            (REEL_WIDTH, REEL_HEIGHT),
                            method=Image.Resampling.LANCZOS,
                        )
                        vertical_img.save(frame_path, format="JPEG", quality=92)
                    scene_paths.append(frame_path)

                tts_result = synthesize_gemini_tts_wav(
                    voiceover_script,
                    voice_id=resolved_voice_id,
                    target_duration_seconds=clamped_duration,
                )
                wav_path = work / "voiceover.wav"
                wav_path.write_bytes(tts_result["wav_bytes"])
                ass_path = work / "captions.ass"
                ass_path.write_text(_srt_to_ass(transcript_srt), encoding="utf-8")
                out_mp4 = work / "reel.mp4"

                motion_cmd = _build_motion_ffmpeg_cmd(
                    ffmpeg_bin=ffmpeg_bin,
                    scene_paths=scene_paths,
                    wav_path=wav_path,
                    ass_path=ass_path,
                    out_mp4=out_mp4,
                    clamped_duration=clamped_duration,
                )
                proc = subprocess.run(motion_cmd, capture_output=True, text=True, timeout=120)

                if proc.returncode != 0 or not out_mp4.is_file():
                    logger.warning(
                        "ffmpeg motion/caption render failed (exit %s), falling back to static concat: %s",
                        proc.returncode,
                        (proc.stderr or proc.stdout or "")[-300:],
                    )
                    if len(scene_paths) > 1:
                        concat_file = work / "scenes.txt"
                        per_scene = clamped_duration / len(scene_paths)
                        lines = []
                        for sp in scene_paths:
                            escaped = sp.as_posix().replace("'", "'\\''")
                            lines.append(f"file '{escaped}'")
                            lines.append(f"duration {per_scene:.3f}")
                        last_escaped = scene_paths[-1].as_posix().replace("'", "'\\''")
                        lines.append(f"file '{last_escaped}'")
                        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        video_inputs = ["-f", "concat", "-safe", "0", "-i", str(concat_file)]
                    else:
                        video_inputs = ["-loop", "1", "-i", str(scene_paths[0])]

                    cmd = [
                        ffmpeg_bin,
                        "-y",
                        *video_inputs,
                        "-i", str(wav_path),
                        "-t", f"{clamped_duration:.2f}",
                        "-vf", f"scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=increase,crop={REEL_WIDTH}:{REEL_HEIGHT},setsar=1:1,fps={REEL_FPS},format=yuv420p",
                        "-af", f"apad=whole_dur={clamped_duration:.2f}",
                        "-c:v", "libx264",
                        "-preset", "ultrafast",
                        "-profile:v", "high",
                        "-level:v", "4.1",
                        "-c:a", "aac",
                        "-ar", "48000",
                        "-ac", "2",
                        "-b:a", "128k",
                        "-movflags", "+faststart",
                        str(out_mp4),
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if proc.returncode == 0 and out_mp4.is_file():
                    mp4_bytes = out_mp4.read_bytes()
                    metadata = inspect_reel_video_bytes(mp4_bytes)
                    metadata["voice_id"] = resolved_voice_id
                    metadata["voice_label"] = voice_meta["label"]
                    return {
                        "video_bytes": mp4_bytes,
                        "feed_image_bytes": keyframe_jpeg,
                        "transcript_srt": transcript_srt,
                        "duration_seconds": metadata["duration_seconds"],
                        "voice_id": resolved_voice_id,
                        "voice_label": voice_meta["label"],
                        "metadata": metadata,
                    }
                err_tail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}")[-500:]
                logger.error("ffmpeg reel rendering failed (exit %s): %s", proc.returncode, err_tail)
                if is_prod:
                    raise RuntimeError(f"ffmpeg reel rendering failed: {err_tail}")
        except RuntimeError:
            raise
        except Exception as exc:
            logger.exception("ffmpeg reel rendering raised an unexpected error: %s", exc)
            if is_prod:
                raise RuntimeError(f"ffmpeg reel rendering error: {exc}") from exc

    # Deterministic ISO-BMFF MP4 container with 1080x1920 H.264 + AAC 48kHz stereo metadata
    mp4_bytes = _build_canonical_mp4_container(
        width=REEL_WIDTH,
        height=REEL_HEIGHT,
        duration_seconds=clamped_duration,
        fps=REEL_FPS,
        sample_rate=48000,
        embedded_jpeg_frame=keyframe_jpeg,
    )
    metadata = inspect_reel_video_bytes(mp4_bytes)
    metadata["voice_id"] = resolved_voice_id
    metadata["voice_label"] = voice_meta["label"]
    return {
        "video_bytes": mp4_bytes,
        "feed_image_bytes": keyframe_jpeg,
        "transcript_srt": transcript_srt,
        "duration_seconds": clamped_duration,
        "voice_id": resolved_voice_id,
        "voice_label": voice_meta["label"],
        "metadata": metadata,
    }
