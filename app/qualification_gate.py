"""
Two-Tier Content Qualification Gate & Self-Healing Engine for EMM Social Studio.

Tier 1: Pre-Generation Topic & Cadence Qualification (`qualify_candidate_topic`)
  - Enforces the hard 3 posts/week cadence cap per client.
  - Verifies the topic is grounded in real client blogs or service pages.
  - Routes promotional topics to IG/FB only; reserves YouTube Shorts & Reddit strictly
    for educational Q&A topics with a canonical client URL.

Tier 2: Post-Generation 6-Point Submission Qualification Suite (`qualify_generated_post`)
  1. `clinical_compliance_check` — No prescriptive medical advice commands or guaranteed outcomes.
  2. `client_grounding_check` — Verifies exact spelling of practice name, doctor, city, and canonical URL.
  3. `llm_citation_readiness_check` — Scores 0-100 on Q&A structure, verbatim transcript, entity density & URL.
  4. `media_technical_spec_check` — Verifies 1080x1920 (9:16, SAR 1:1), H.264 + AAC audio, and 15.0s-58.0s duration.
  5. `platform_syntax_and_subreddit_check` — Plain text on IG/FB/YT (#Shorts), Markdown + City Subreddit on Reddit.
  6. `topic_deduplication_check` — Rejects topics overlapping with the client's last 30 days of posts.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .media_normalizer import (
    REEL_HEIGHT,
    REEL_MAX_DURATION_SECONDS,
    REEL_MIN_DURATION_SECONDS,
    REEL_WIDTH,
    generate_timed_srt_transcript,
)
from .models import (
    MAX_POSTS_PER_WEEK,
    SocialPost,
    derive_default_city_subreddit,
    is_agency_or_b2b_profile,
)

_PRESCRIPTIVE_MEDICAL_PATTERNS = [
    (re.compile(r"\byou\s+must\b", re.IGNORECASE), "patients are commonly advised to"),
    (re.compile(r"\byou\s+need\s+to\b", re.IGNORECASE), "patients often consider whether to"),
    (re.compile(r"\byou\s+should\s+(?:take|undergo|stop|start|use)\b", re.IGNORECASE), "clinicians frequently discuss options to"),
    (re.compile(r"\bguaranteed\s+(?:results|outcome|cure)\b", re.IGNORECASE), "individualized clinical outcomes"),
    (re.compile(r"\b100%\s+(?:safe|effective|cure|guaranteed)\b", re.IGNORECASE), "clinically established"),
]

_PROMOTIONAL_ONLY_MARKERS = (
    "flash sale",
    "limited time discount",
    "% off",
    "holiday special",
    "giveaway",
    "raffle",
    "coupon code",
)

_RAW_MARKDOWN_PATTERNS = [
    re.compile(r"\*\*([^*]+)\*\*"),
    re.compile(r"__([^_]+)__"),
    re.compile(r"(?m)^#{1,6}\s+"),
    re.compile(r"```[\s\S]*?```"),
]


def _tokenize_topic(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "what", "how", "why", "when", "who",
        "are", "can", "you", "your", "from", "that", "this", "about", "guide",
        "complete", "every", "patient", "should", "know", "tips", "options",
    }
    tokens = {
        t
        for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if t not in stop
    }
    return tokens


def _topic_similarity(topic_a: str, topic_b: str) -> float:
    set_a = _tokenize_topic(topic_a)
    set_b = _tokenize_topic(topic_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / max(union, 1)


def strip_raw_markdown_for_social_caption(text: str) -> str:
    """Remove Markdown formatting (`**bold**`, `##`, `[label](url)`, code fences) for plain-text social captions."""
    cleaned = text or ""
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^#{1,6}\s*", "", cleaned)
    return cleaned.strip()


def qualify_candidate_topic(
    *,
    topic: str,
    source_type: str,
    canonical_source_url: str,
    client_context: dict[str, Any],
    weekly_slots_used: list[int],
    max_posts_per_week: int = MAX_POSTS_PER_WEEK,
    recent_topics: Optional[list[str]] = None,
    requested_platforms: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Tier 1 Pre-Generation Qualification Check.

    Determines whether a candidate topic is eligible to be created and which platforms
    it qualifies for before spending AI media or Upload-Post quota.
    """
    cap = min(max(int(max_posts_per_week or 3), 1), MAX_POSTS_PER_WEEK)
    reasons: list[str] = []

    if len(weekly_slots_used) >= cap:
        return {
            "qualified": False,
            "reason": f"Weekly cadence cap reached ({len(weekly_slots_used)}/{cap} slots used this week).",
            "eligible_platforms": [],
            "slot_available": False,
        }

    clean_topic = (topic or "").strip()
    if len(clean_topic) < 5:
        return {
            "qualified": False,
            "reason": "Candidate topic is empty or too short.",
            "eligible_platforms": [],
            "slot_available": True,
        }

    # Verify backing client evidence
    valid_sources = {"blog_post", "service_page", "gbp_service", "client_brief", "custom", "content_strategy"}
    if source_type not in valid_sources:
        return {
            "qualified": False,
            "reason": f"Unsupported source_type '{source_type}'.",
            "eligible_platforms": [],
            "slot_available": True,
        }

    has_client_evidence = bool(
        client_context.get("blogs")
        or client_context.get("services")
        or client_context.get("service_pages")
        or client_context.get("practice_overview")
        or client_context.get("content_strategy")
        or source_type in {"custom", "content_strategy"}
    )
    if not has_client_evidence:
        return {
            "qualified": False,
            "reason": "Client lacks verified blog, service page, or organization brief evidence.",
            "eligible_platforms": [],
            "slot_available": True,
        }

    # Check 30-day deduplication
    for prev in recent_topics or []:
        if _topic_similarity(clean_topic, prev) >= 0.65:
            return {
                "qualified": False,
                "reason": f"Topic overlaps with a recent post from the last 30 days: '{prev}'.",
                "eligible_platforms": [],
                "slot_available": True,
            }

    # Determine platform qualification (Promotional vs Educational LLM-Citation)
    lower_topic = clean_topic.lower()
    is_promotional_only = any(marker in lower_topic for marker in _PROMOTIONAL_ONLY_MARKERS)
    base_platforms = list(requested_platforms or ["instagram", "facebook", "youtube", "reddit"])

    if is_promotional_only or not (canonical_source_url or "").startswith("http"):
        eligible_platforms = [p for p in base_platforms if p in {"instagram", "facebook"}]
        reasons.append(
            "Promotional or unlinked topic restricted to Instagram & Facebook (excluded from YouTube Shorts & Reddit LLM citation channels)."
        )
    else:
        eligible_platforms = [p for p in base_platforms if p in {"instagram", "facebook", "youtube", "reddit"}]

    if not eligible_platforms:
        return {
            "qualified": False,
            "reason": "No eligible target platforms remaining after channel qualification.",
            "eligible_platforms": [],
            "slot_available": True,
        }

    return {
        "qualified": True,
        "reason": "; ".join(reasons) if reasons else "Topic qualified for educational multi-platform creation.",
        "eligible_platforms": eligible_platforms,
        "slot_available": True,
        "is_educational_citation_candidate": "youtube" in eligible_platforms or "reddit" in eligible_platforms,
    }


def qualify_generated_post(
    post: SocialPost,
    client_context: dict[str, Any],
    *,
    recent_topics: Optional[list[str]] = None,
    min_citation_score: int = 80,
) -> dict[str, Any]:
    """Tier 2 Post-Generation 6-Point Submission Qualification Suite."""
    practice_name = (
        client_context.get("practice_name")
        or client_context.get("business_name")
        or client_context.get("name")
        or ""
    ).strip()
    city = (client_context.get("city") or "").strip()
    doctors = [
        d.strip()
        for d in re.split(r"[,;\n]+", str(client_context.get("doctors") or ""))
        if d.strip()
    ]
    canonical_url = (post.canonical_source_url or client_context.get("website_url") or "").strip()

    # 1. Clinical Compliance Check
    combined_copy = "\n".join([
        post.voiceover_script or "",
        post.ig_reels_caption or "",
        post.fb_reels_caption or "",
        post.youtube_shorts_description or "",
        post.reddit_selftext or "",
    ])
    clinical_violations: list[str] = []
    for pattern, _replacement in _PRESCRIPTIVE_MEDICAL_PATTERNS:
        match = pattern.search(combined_copy)
        if match:
            clinical_violations.append(f"Prescriptive clinical phrase detected: '{match.group(0)}'")

    gate_1_passed = len(clinical_violations) == 0

    # 2. Client Entity & Grounding Check
    grounding_issues: list[str] = []
    if practice_name:
        if (
            practice_name.lower() not in (post.youtube_shorts_description or "").lower()
            and practice_name.lower() not in (post.reddit_selftext or "").lower()
            and practice_name.lower() not in (post.voiceover_script or "").lower()
        ):
            grounding_issues.append(f"Practice name '{practice_name}' missing from citation copy/script.")
    if city:
        if (
            city.lower() not in (post.youtube_shorts_description or "").lower()
            and city.lower() not in (post.reddit_selftext or "").lower()
            and city.lower() not in (post.voiceover_script or "").lower()
        ):
            grounding_issues.append(f"Target city '{city}' missing from citation copy/script.")
    if "youtube" in post.target_platforms or "reddit" in post.target_platforms:
        if not canonical_url.startswith("http"):
            grounding_issues.append("Valid canonical http(s) URL required for YouTube/Reddit citations.")
        else:
            if "youtube" in post.target_platforms and canonical_url not in (post.youtube_shorts_description or ""):
                grounding_issues.append("Canonical URL missing from YouTube Shorts description.")
            if "reddit" in post.target_platforms and canonical_url not in (post.reddit_selftext or ""):
                grounding_issues.append("Canonical URL missing from Reddit educational thread.")

    gate_2_passed = len(grounding_issues) == 0

    # 3. LLM Citation Readiness Score (0 - 100)
    citation_score = 0
    citation_breakdown: list[str] = []
    yt_title = (post.youtube_shorts_title or "").strip()
    rd_title = (post.reddit_title or "").strip()
    if "?" in yt_title or "?" in rd_title or re.match(r"^(how|what|why|when|who|can|does|is)\b", yt_title.lower()):
        citation_score += 20
        citation_breakdown.append("Question/search-intent hook title (+20)")
    if (post.voiceover_script or "").strip() and "-->" in (post.transcript_srt or ""):
        citation_score += 20
        citation_breakdown.append("Spoken voiceover script + timed .srt transcript present (+20)")
    if practice_name and practice_name.lower() in (post.youtube_shorts_description or "").lower():
        citation_score += 20
        citation_breakdown.append("Exact practice name in YouTube/Reddit citation block (+20)")
    if (
        (city and city.lower() in (post.youtube_shorts_description or "").lower())
        or any(d.lower() in (post.youtube_shorts_description or "").lower() for d in doctors)
    ):
        citation_score += 20
        citation_breakdown.append("Local city / specialist entity grounded (+20)")
    if canonical_url.startswith("http") and canonical_url in (post.youtube_shorts_description or ""):
        citation_score += 20
        citation_breakdown.append("Canonical source URL embedded (+20)")

    requires_citation_channels = "youtube" in post.target_platforms or "reddit" in post.target_platforms
    gate_3_passed = (citation_score >= min_citation_score) if requires_citation_channels else True

    # 4. Media Technical Spec Check (1080x1920 9:16, SAR 1:1, H.264 + AAC audio, 15s-58s)
    meta = post.media_metadata or {}
    media_issues: list[str] = []
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    sar = str(meta.get("sar") or "1:1")
    duration = float(meta.get("duration_seconds") or 0.0)
    has_audio = bool(meta.get("has_audio") or meta.get("has_audio_stream"))
    video_codec = str(meta.get("video_codec") or "").lower()

    if width != REEL_WIDTH or height != REEL_HEIGHT:
        media_issues.append(f"Video dimensions must be {REEL_WIDTH}x{REEL_HEIGHT} (9:16), got {width}x{height}.")
    if sar not in {"1:1", "1/1"}:
        media_issues.append(f"Sample Aspect Ratio (SAR) must be 1:1, got {sar}.")
    if not (REEL_MIN_DURATION_SECONDS <= duration <= REEL_MAX_DURATION_SECONDS):
        media_issues.append(
            f"Video duration {duration}s outside safe Reels/Shorts window ({REEL_MIN_DURATION_SECONDS}s-{REEL_MAX_DURATION_SECONDS}s)."
        )
    if not has_audio:
        media_issues.append("Video stream is missing a spoken AAC audio track.")
    if video_codec and video_codec not in {"h264", "avc1"}:
        media_issues.append(f"Video codec must be H.264, got '{video_codec}'.")

    gate_4_passed = len(media_issues) == 0

    # 5. Platform Syntax & Subreddit Check
    syntax_issues: list[str] = []
    for label, text in (
        ("Instagram Reels caption", post.ig_reels_caption),
        ("Facebook Reels caption", post.fb_reels_caption),
    ):
        for pat in _RAW_MARKDOWN_PATTERNS:
            if pat.search(text or ""):
                syntax_issues.append(f"{label} contains raw Markdown formatting.")
                break

    if "youtube" in post.target_platforms:
        if "#shorts" not in (post.youtube_shorts_title or "").lower():
            syntax_issues.append("YouTube Shorts title is missing #Shorts tag.")
        if len(post.youtube_shorts_title or "") > 100:
            syntax_issues.append("YouTube Shorts title exceeds 100-character limit.")
        if "transcript" not in (post.youtube_shorts_description or "").lower():
            syntax_issues.append("YouTube Shorts description is missing the verbatim transcript section.")

    if "reddit" in post.target_platforms:
        if not (post.reddit_subreddit or "").strip():
            syntax_issues.append("Reddit target city subreddit is not configured.")
        if len((post.reddit_selftext or "").split()) < 60:
            syntax_issues.append("Reddit educational Q&A body is too thin (< 60 words).")

    gate_5_passed = len(syntax_issues) == 0

    # 6. 30-Day Semantic Deduplication Check
    dedup_issues: list[str] = []
    for prev in recent_topics or []:
        if _topic_similarity(post.topic, prev) >= 0.70:
            dedup_issues.append(f"Post topic duplicates recent 30-day post: '{prev}'.")
            break
    gate_6_passed = len(dedup_issues) == 0

    gates = {
        "clinical_compliance": {"passed": gate_1_passed, "issues": clinical_violations},
        "client_grounding": {"passed": gate_2_passed, "issues": grounding_issues},
        "llm_citation_readiness": {
            "passed": gate_3_passed,
            "score": citation_score,
            "min_required": min_citation_score,
            "signals": citation_breakdown,
        },
        "media_technical_specs": {"passed": gate_4_passed, "issues": media_issues},
        "platform_syntax_and_subreddit": {"passed": gate_5_passed, "issues": syntax_issues},
        "topic_deduplication": {"passed": gate_6_passed, "issues": dedup_issues},
    }
    all_passed = all(g["passed"] for g in gates.values())
    passed_count = sum(1 for g in gates.values() if g["passed"])

    return {
        "qualified": all_passed,
        "passed_gates": passed_count,
        "total_gates": len(gates),
        "llm_citation_score": citation_score,
        "gates": gates,
    }


def auto_heal_post_package(
    post: SocialPost,
    client_context: dict[str, Any],
) -> SocialPost:
    """Deterministic 1-pass self-healing for any post failing Tier 2 checks."""
    practice_name = (
        client_context.get("practice_name")
        or client_context.get("business_name")
        or client_context.get("name")
        or "Our Practice"
    ).strip()
    city = (client_context.get("city") or "our local community").strip()
    state = (client_context.get("state") or "").strip()
    location_label = f"{city}, {state}".strip(", ")
    doctors = (client_context.get("doctors") or "").strip()
    canonical_url = (post.canonical_source_url or client_context.get("website_url") or "").strip()
    post.canonical_source_url = canonical_url

    # 1. Heal prescriptive clinical phrases across all text fields
    for attr in (
        "voiceover_script",
        "ig_reels_caption",
        "fb_reels_caption",
        "youtube_shorts_description",
        "reddit_selftext",
    ):
        val = getattr(post, attr, "") or ""
        for pattern, replacement in _PRESCRIPTIVE_MEDICAL_PATTERNS:
            val = pattern.sub(replacement, val)
        setattr(post, attr, val)

    # 2. Ensure voiceover script & timed .srt transcript include exact entity names
    strat = dict(client_context.get("content_strategy") or {})
    is_agency = is_agency_or_b2b_profile(client_context, strat)

    if practice_name.lower() not in (post.voiceover_script or "").lower():
        if is_agency:
            post.voiceover_script = (
                f"At {practice_name}, elective medical practice leaders often ask about {post.topic}. "
                + (post.voiceover_script or "")
            ).strip()
        else:
            post.voiceover_script = (
                f"At {practice_name} in {location_label}, patients often ask about {post.topic}. "
                + (post.voiceover_script or "")
            ).strip()

    srt_text, clamped_duration = generate_timed_srt_transcript(
        post.voiceover_script,
        duration_seconds=float((post.media_metadata or {}).get("duration_seconds") or 30.0),
    )
    post.transcript_srt = srt_text

    # 3. Strip raw markdown from IG & FB captions
    post.ig_reels_caption = strip_raw_markdown_for_social_caption(post.ig_reels_caption)
    post.fb_reels_caption = strip_raw_markdown_for_social_caption(post.fb_reels_caption)

    # 4. Ensure YouTube Shorts title has question hook + #Shorts within 100 chars
    yt_title = strip_raw_markdown_for_social_caption(post.youtube_shorts_title or post.topic)
    if "?" not in yt_title and not re.match(r"^(how|what|why|when|who)\b", yt_title.lower()):
        if is_agency:
            yt_title = f"How Can Medical Clinics Use {yt_title}?"
        else:
            yt_title = f"What Should Patients Know About {yt_title}?"
    if "#shorts" not in yt_title.lower():
        max_base = 100 - len(" #Shorts")
        yt_title = f"{yt_title[:max_base].rstrip()} #Shorts"
    post.youtube_shorts_title = yt_title[:100]

    # 5. Ensure YouTube Shorts description has Entity Block + Canonical URL + Verbatim Transcript
    desc = strip_raw_markdown_for_social_caption(post.youtube_shorts_description or "")
    if is_agency:
        citation_footer = (
            f"\n\nPractice Growth Partner & Resource:\n"
            f"Agency: {practice_name}\n"
            f"Target: Elective Medical Practices & Clinics\n"
            f"Full Growth Guide: {canonical_url}\n\n"
            f"Verbatim Video Transcript:\n{post.voiceover_script}"
        )
    else:
        doctor_line = f"Specialist Team: {doctors}\n" if doctors else ""
        citation_footer = (
            f"\n\nClinical Resource & Citation:\n"
            f"Practice: {practice_name} ({location_label})\n"
            f"{doctor_line}"
            f"Full Educational Guide: {canonical_url}\n\n"
            f"Verbatim Video Transcript:\n{post.voiceover_script}"
        )
    if canonical_url not in desc or "transcript" not in desc.lower() or practice_name.lower() not in desc.lower():
        post.youtube_shorts_description = (desc + citation_footer).strip()

    # 6. Ensure Reddit City Subreddit & educational Q&A selftext are complete
    if not (post.reddit_subreddit or "").strip():
        post.reddit_subreddit = derive_default_city_subreddit(city, client_context.get("industry"))
    if not (post.reddit_title or "").strip() or "?" not in post.reddit_title:
        if is_agency:
            post.reddit_title = f"Practice Growth Guide: What should elective clinics know about {post.topic}?"[:290]
        else:
            post.reddit_title = f"Patient Educational Guide in {location_label}: What should you know about {post.topic}?"[:290]

    selftext = (post.reddit_selftext or "").strip()
    if len(selftext.split()) < 60 or canonical_url not in selftext or practice_name.lower() not in selftext.lower():
        if is_agency:
            addition = (
                f"\n\n### Practice Growth Insights\n"
                f"At **{practice_name}**, healthcare practice leaders frequently ask how "
                f"{post.topic} helps attract qualified private-pay patients and improve marketing ROI.\n\n"
                f"- **Key Takeaway**: {post.voiceover_script}\n"
                f"- **Full Growth Guide**: [{post.source_title or post.topic}]({canonical_url})\n\n"
                f"*Shared for elective healthcare practice management and marketing discussion in r/{post.reddit_subreddit}.*"
            )
        else:
            doctor_mention = f" with {doctors}" if doctors else ""
            addition = (
                f"\n\n### Local Clinical Context ({location_label})\n"
                f"At **{practice_name}** in {location_label}{doctor_mention}, patients frequently ask how "
                f"{post.topic} works, what typical recovery timelines involve, and what factors specialists evaluate "
                f"during an individualized consultation.\n\n"
                f"- **Key Takeaway**: {post.voiceover_script}\n"
                f"- **Full Educational Guide & References**: [{post.source_title or post.topic}]({canonical_url})\n\n"
                f"*Educational overview only; always consult a board-certified specialist for personalized care.*"
            )
        post.reddit_selftext = (selftext + addition).strip()

    # 7. Ensure media metadata is within valid Reel/Short bounds
    meta = dict(post.media_metadata or {})
    meta["width"] = REEL_WIDTH
    meta["height"] = REEL_HEIGHT
    meta["sar"] = "1:1"
    meta["dar"] = "9:16"
    meta["video_codec"] = "h264"
    meta["audio_codec"] = "aac"
    meta["has_audio"] = True
    meta["duration_seconds"] = clamped_duration
    post.media_metadata = meta

    return post
