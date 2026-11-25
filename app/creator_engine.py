"""
Gemini Educational Content Creator & Multi-Platform Dispatcher for EMM Social Studio.

Transforms each client's published blogs and service pages into a coordinated
educational video + LLM-citation package:
1. 1080x1920 (9:16) Spoken Educational Video (15.0s - 58.0s, H.264 + AAC 48kHz)
   -> Instagram Reels (`media_type=REELS`)
   -> Facebook Reels (`facebook_media_type=REELS`)
   -> YouTube Shorts (`#Shorts` + Citation Description + Verbatim Timed Transcript)
2. Companion Educational Q&A Thread
   -> Reddit City Subreddit (`r/<CitySubreddit>` via Official Reddit Developer API)
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests

from .media_normalizer import (
    compose_reel_scene_jpeg,
    generate_ai_scene_background_bytes,
    generate_clinical_placeholder_scene_jpeg,
    render_and_normalize_reel_video,
    sanitize_spoken_narration,
)
from .models import (
    DEFAULT_SCHEDULE_DAYS,
    DEFAULT_TARGET_PLATFORMS,
    ISO_TO_WEEKDAY_LABEL,
    WEEKDAY_ABBR_TO_ISO,
    SocialPost,
    SocialTenantProfile,
    derive_default_city_subreddit,
    derive_default_content_strategy,
    is_agency_or_b2b_profile,
    normalize_schedule_days,
    utcnow,
)
from .qualification_gate import (
    _topic_similarity,
    auto_heal_post_package,
    qualify_candidate_topic,
    qualify_generated_post,
    strip_raw_markdown_for_social_caption,
)
from .reddit_client import (
    RedditPublishError,
    normalize_subreddit_name,
    publish_city_subreddit_educational_post,
)
from .uploadpost_client import UploadPostError, publish_platform_video

logger = logging.getLogger(__name__)

GOOGLE_GEMINI_ORIGIN = "https://generativelanguage.googleapis.com"


def _gemini_connection() -> tuple[str, str]:
    if os.environ.get("CLOUD_RUN_MODE") == "production":
        return GOOGLE_GEMINI_ORIGIN + "/v1beta", os.environ.get("GEMINI_API_KEY", "")
    base = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL", GOOGLE_GEMINI_ORIGIN).rstrip("/")
    key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    return base, key


def _gemini_text_model() -> str:
    model = os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.8-flash")
    if not re.fullmatch(r"gemini-[a-zA-Z0-9.-]+", model):
        return "gemini-3.8-flash"
    return model


def select_next_qualifying_source(
    client_context: dict[str, Any],
    recent_topics: list[str],
) -> dict[str, Any]:
    """Select the highest-priority unshared blog post, service page, or strategy topic for a client."""
    website_url = (client_context.get("website_url") or "").strip().rstrip("/")
    strat = dict(client_context.get("content_strategy") or {})
    is_agency = is_agency_or_b2b_profile(client_context, strat)

    # 1. Prioritize published/scheduled blogs
    for blog in client_context.get("blogs") or []:
        title = (blog.get("title") or blog.get("keyword") or "").strip()
        if not title:
            continue
        if any(_topic_similarity(title, prev) >= 0.65 for prev in recent_topics):
            continue
        url = (blog.get("published_url") or blog.get("url") or (f"{website_url}/blog/" if website_url else "")).strip()
        return {
            "source_type": "blog_post",
            "source_ref_id": str(blog.get("id") or ""),
            "source_title": title,
            "topic": title,
            "canonical_source_url": url,
            "summary": (blog.get("excerpt") or blog.get("body_preview") or "").strip(),
        }

    # 2. Next, check discovered client service pages
    for page in client_context.get("service_pages") or []:
        title = (page.get("title") or "").strip()
        url = (page.get("url") or website_url).strip()
        if not title:
            continue
        if any(_topic_similarity(title, prev) >= 0.65 for prev in recent_topics):
            continue
        return {
            "source_type": "service_page",
            "source_ref_id": str(page.get("id") or url),
            "source_title": title,
            "topic": title,
            "canonical_source_url": url,
            "summary": (page.get("summary") or "").strip(),
        }

    # 3. Next, check recommended topics from Content Strategy
    for rec in strat.get("topic_recommendations") or []:
        rec_clean = str(rec).strip()
        if not rec_clean:
            continue
        if any(_topic_similarity(rec_clean, prev) >= 0.65 for prev in recent_topics):
            continue
        ref_slug = re.sub(r"[^a-z0-9]+", "-", rec_clean.lower()).strip("-")[:64]
        return {
            "source_type": "content_strategy",
            "source_ref_id": ref_slug,
            "source_title": rec_clean,
            "topic": rec_clean,
            "canonical_source_url": website_url,
            "summary": f"Strategic guidance on {rec_clean} for {strat.get('target_audience') or 'prospective clients'}.",
        }

    # 4. Next, check GBP services & Organization Brief services
    services: list[str] = list(client_context.get("gbp_services") or []) + list(client_context.get("services") or [])
    city = (client_context.get("city") or "").strip()
    for svc in services:
        svc_clean = str(svc).strip()
        if not svc_clean:
            continue
        topic_label = f"{svc_clean} in {city}" if city else svc_clean
        if any(_topic_similarity(topic_label, prev) >= 0.65 for prev in recent_topics):
            continue
        guide_title = f"{svc_clean} Practice Guide" if is_agency else f"{svc_clean} Patient Guide"
        summary_text = (
            f"Strategic analysis of {svc_clean} and marketing opportunities for elective practices."
            if is_agency
            else f"Educational overview of {svc_clean} options, candidacy, and consultation guidelines."
        )
        return {
            "source_type": "gbp_service",
            "source_ref_id": svc_clean.lower().replace(" ", "-"),
            "source_title": guide_title,
            "topic": topic_label,
            "canonical_source_url": website_url,
            "summary": summary_text,
        }

    # 5. Fallback to distinct general practice educational / growth angles
    practice_name = client_context.get("practice_name") or client_context.get("business_name") or client_context.get("name") or "Our Practice"
    industry = client_context.get("industry") or ("Elective Healthcare Marketing" if is_agency else "Specialty Care")

    if is_agency:
        fallback_angles = [
            ("agency-growth", f"Key Patient Acquisition Strategies for Elective Practices with {practice_name}"),
            ("agency-seo", f"How {practice_name} Helps Medical Clinics Increase Consultation Bookings"),
            ("agency-roi", f"Maximizing Marketing ROI for Modern Medical Clinics"),
        ]
    else:
        fallback_angles = [
            ("practice-overview", f"What to Expect During a {industry} Consultation at {practice_name}"),
            ("patient-questions", f"Top Questions to Ask Before {industry} Treatment at {practice_name}"),
            ("recovery-planning", f"How {practice_name} Approaches Preparation and Recovery Planning"),
        ]

    for ref_id, candidate_topic in fallback_angles:
        if not any(_topic_similarity(candidate_topic, prev) >= 0.65 for prev in recent_topics):
            return {
                "source_type": "client_brief",
                "source_ref_id": ref_id,
                "source_title": candidate_topic,
                "topic": candidate_topic,
                "canonical_source_url": website_url,
                "summary": (client_context.get("practice_overview") or "").strip(),
            }
    return {
        "source_type": "client_brief",
        "source_ref_id": fallback_angles[0][0],
        "source_title": fallback_angles[0][1],
        "topic": fallback_angles[0][1],
        "canonical_source_url": website_url,
        "summary": (client_context.get("practice_overview") or "").strip(),
    }


def generate_ai_content_strategy(client_context: dict[str, Any]) -> dict[str, Any]:
    """Generate or enhance the client's Content Strategy using Gemini, with deterministic heuristic fallback."""
    base_default = derive_default_content_strategy(client_context)
    brand_name = (
        client_context.get("business_name")
        or client_context.get("practice_name")
        or client_context.get("name")
        or "Our Company"
    ).strip()
    city = (client_context.get("city") or "").strip()
    state = (client_context.get("state") or "").strip()
    industry = (client_context.get("industry") or "").strip()
    website_url = (client_context.get("website_url") or "").strip()
    overview = (client_context.get("practice_overview") or "").strip()
    services = [str(s).strip() for s in (client_context.get("services") or []) if str(s).strip()]
    blogs = [str(b.get("title") or "").strip() for b in (client_context.get("blogs") or []) if b.get("title")]
    is_agency = is_agency_or_b2b_profile(client_context, None)

    prompt = (
        f"You are an expert Chief Marketing Officer and Social Media Strategist.\n"
        f"Analyze this client/tenant to determine their exact business positioning, target audience, core content pillars, tone of voice, prohibited angles, and recommended topics for their automated social media pipeline (Instagram Reels, Facebook Reels, YouTube Shorts, and Reddit).\n\n"
        "CRITICAL REQUIREMENT:\n"
        "Carefully distinguish whether this business is:\n"
        "1) A direct healthcare/aesthetic practice (e.g. Plastic Surgeon, Med Spa, Dermatology Clinic, Dental Office) treating patients; OR\n"
        "2) A marketing agency, software company, or B2B service provider (e.g. Elevated Medical Marketing) that serves medical clinics and surgeons!\n"
        "Never confuse an agency serving medical practices with a medical clinic that treats patients.\n\n"
        f"Client Profile:\n"
        f"- Brand Name: {brand_name}\n"
        f"- Location: {city}, {state}\n"
        f"- Website: {website_url}\n"
        f"- Industry: {industry}\n"
        f"- Overview: {overview}\n"
        f"- Services / Offerings: {', '.join(services[:10])}\n"
        f"- Recent Blog Topics: {', '.join(blogs[:10])}\n"
        f"- Detected Category: {'B2B Marketing Agency / Growth Partner for Elective Medical Practices' if is_agency else 'Clinical Practice / Healthcare Clinic'}\n\n"
        "Return a JSON object with keys:\n"
        "- business_type: string (concise entity role, e.g. 'Marketing Agency for Elective Medical Practices' or 'Board-Certified Plastic Surgery Practice')\n"
        "- target_audience: string (who their social media content speaks to)\n"
        "- brand_positioning: string (1-2 sentence core value proposition and market stance)\n"
        "- tone_of_voice: string (guidance on voice and perspective)\n"
        "- content_pillars: array of 4-6 strings (high-level content themes to regularly post about)\n"
        "- prohibited_angles: string (clear instructions on what NOT to say or frame, e.g. never present as a clinic if an agency)\n"
        "- topic_recommendations: array of 6-10 strings (specific high-converting video/post topics)\n"
        "- custom_guidance: string (short strategic directives for hooks and CTAs)"
    )

    ai_result = _call_gemini_structured_copy(prompt)
    if isinstance(ai_result, dict):
        merged = dict(base_default)
        for k in ("business_type", "target_audience", "brand_positioning", "tone_of_voice", "prohibited_angles", "custom_guidance"):
            v = ai_result.get(k)
            if isinstance(v, str) and v.strip():
                merged[k] = v.strip()
        for k in ("content_pillars", "topic_recommendations"):
            v = ai_result.get(k)
            if isinstance(v, list) and v:
                merged[k] = [str(item).strip() for item in v if str(item).strip()]
        return merged
    return base_default


def _call_gemini_structured_copy(prompt: str) -> Optional[dict[str, Any]]:
    base, api_key = _gemini_connection()
    if not api_key:
        return None
    model = _gemini_text_model()
    if "/v1beta" in base:
        url = f"{base}/models/{model}:generateContent"
    else:
        url = f"{base}/v1beta/models/{model}:generateContent"

    try:
        resp = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json",
                },
            },
            timeout=45,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        raw_text = "".join(p.get("text", "") for p in parts).strip()
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE).strip()
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        logger.warning("Gemini social copy generation fallback triggered: %s", exc)
        return None


def _synthesize_deterministic_copy(
    *,
    topic: str,
    source_title: str,
    source_summary: str,
    canonical_url: str,
    client_context: dict[str, Any],
    city_subreddit: str,
) -> dict[str, str]:
    """Build high-signal, grounded educational copy across all 4 platforms."""
    practice_name = (
        client_context.get("practice_name")
        or client_context.get("business_name")
        or client_context.get("name")
        or "Our Practice"
    ).strip()
    city = (client_context.get("city") or "our local area").strip()
    state = (client_context.get("state") or "").strip()
    location_label = f"{city}, {state}".strip(", ")
    doctors = (client_context.get("doctors") or "").strip()
    doctor_phrase = f" with {doctors}" if doctors else ""
    clean_topic = strip_raw_markdown_for_social_caption(topic)
    strat = dict(client_context.get("content_strategy") or {})
    is_agency = is_agency_or_b2b_profile(client_context, strat)

    city_tag = re.sub(r"[^a-zA-Z0-9]", "", city) or "LocalArea"
    topic_tag = re.sub(r"[^a-zA-Z0-9]", "", clean_topic.split()[0] if clean_topic.split() else "Growth")

    if is_agency:
        hook_headline = f"How Medical Clinics Can Leverage {clean_topic}"[:110]
        cta_line = f"Scale your elective practice with {practice_name}"[:110]
        voiceover_script = (
            f"How can elective medical practices and aesthetic clinics maximize their growth with {clean_topic}? "
            f"At {practice_name}, we help surgeons, medical directors, and healthcare practice owners attract "
            f"qualified private-pay patients and scale consultation bookings. "
            f"{source_summary[:220] if source_summary else f'Strategic execution of {clean_topic} helps modern practices stand out in competitive markets and improve patient acquisition.'} "
            f"To explore proven healthcare marketing insights and practice growth strategies, connect with {practice_name}."
        )
        voiceover_script = re.sub(r"\s+", " ", voiceover_script).strip()

        ig_reels_caption = (
            f"Looking to scale your elective medical practice with {clean_topic}? Here are 3 growth keys every healthcare practice leader should know from {practice_name}:\n\n"
            f"1. Target High-Intent Patients: Reach qualified inquiries actively seeking elective procedures.\n"
            f"2. Conversion-Focused Funnels: Turn website visitors into scheduled consultations.\n"
            f"3. Scalable Marketing ROI: Track every dollar from inquiry to booked revenue.\n\n"
            f"Full growth guide: {canonical_url}\n\n"
            f"#{city_tag} #{topic_tag} #MedicalMarketing #PracticeGrowth #{re.sub(r'[^a-zA-Z0-9]', '', practice_name)[:24]}"
        )

        fb_reels_caption = (
            f"How can elective healthcare practices grow with {clean_topic}?\n\n"
            f"At {practice_name}, we partner with surgeons, aesthetic clinics, and medical directors to build high-performance patient acquisition systems. "
            f"In this guide, we break down actionable strategies to increase consultation bookings and maximize your marketing ROI.\n\n"
            f"Read the full guide: {canonical_url}"
        )

        base_q_title = f"How Can Medical Practices Scale with {clean_topic}?"
        max_yt_base = 100 - len(" #Shorts")
        youtube_shorts_title = f"{base_q_title[:max_yt_base].rstrip()} #Shorts"

        youtube_shorts_description = (
            f"How can elective medical practices leverage {clean_topic} to grow patient volume? "
            f"In this educational breakdown, {practice_name} shares actionable marketing and practice growth insights "
            f"for surgeons, med spa directors, and healthcare practice owners.\n\n"
            f"Key Takeaways Covered:\n"
            f"- How top elective clinics implement {clean_topic}\n"
            f"- Measuring cost per acquisition and patient lifetime value\n"
            f"- Proven frameworks to convert inquiries into scheduled consultations\n\n"
            f"Practice Growth Partner & Resource:\n"
            f"Agency: {practice_name}\n"
            f"Target: Elective Healthcare Practices & Medical Clinics\n"
            f"Full Growth Guide: {canonical_url}\n\n"
            f"Verbatim Video Transcript:\n"
            f"{voiceover_script}"
        )

        reddit_title = f"[Practice Growth Guide] How elective medical clinics approach {clean_topic}"[:290]
        reddit_selftext = (
            f"### Overview: {clean_topic} for Medical Clinics & Elective Practices\n\n"
            f"Healthcare practice owners, surgeons, and clinic managers researching **{clean_topic}** frequently evaluate "
            f"how to structure their marketing, patient acquisition, and digital strategy for maximum return.\n\n"
            f"#### 1. Strategic Implementation\n"
            f"At **{practice_name}**, we help elective medical practices build sustainable patient pipelines. "
            f"{source_summary or 'Effective execution of this strategy requires targeted messaging, fast patient response times, and clear conversion tracking.'}\n\n"
            f"#### 2. Key Questions for Practice Leaders\n"
            f"- What is the expected cost per consultation inquiry for {clean_topic}?\n"
            f"- How does your clinic differentiate its value proposition from local competitors?\n"
            f"- Are inquiry follow-ups and consultation booking workflows fully optimized?\n\n"
            f"#### 3. Practice Resources & Full Guide\n"
            f"- **Practice Growth Resource**: [{source_title or clean_topic} — {practice_name}]({canonical_url})\n\n"
            f"*Shared for elective healthcare practice management and marketing discussion in r/{city_subreddit}.*"
        )
    else:
        hook_headline = f"What Patients in {city} Ask About {clean_topic}"[:110]
        cta_line = f"Read the full educational guide at {practice_name}"[:110]

        voiceover_script = (
            f"What should patients in {location_label} know when considering {clean_topic}? "
            f"At {practice_name}{doctor_phrase}, our clinical team focuses on patient education, "
            f"realistic recovery timelines, and individualized treatment planning. "
            f"{source_summary[:220] if source_summary else f'During an in-person consultation, specialists evaluate anatomy, goals, and safety considerations for {clean_topic}.'} "
            f"To explore our complete educational guide and clinical resources, visit {practice_name} online."
        )
        voiceover_script = re.sub(r"\s+", " ", voiceover_script).strip()

        ig_reels_caption = (
            f"Considering {clean_topic} in {location_label}? Here is what patients commonly discuss during a specialist consultation at {practice_name}.\n\n"
            f"1. Individualized Evaluation: Every patient's anatomy and goals are unique.\n"
            f"2. Evidence-Based Planning: Understanding preparation and typical recovery phases.\n"
            f"3. Specialist Guidance: Meeting{doctor_phrase} at {practice_name} in {location_label} helps clarify your options.\n\n"
            f"Full educational guide: {canonical_url}\n\n"
            f"#{city_tag} #{topic_tag} #PatientEducation #{re.sub(r'[^a-zA-Z0-9]', '', practice_name)[:24]}"
        )

        fb_reels_caption = (
            f"What should you know before scheduling a consultation for {clean_topic} in {location_label}?\n\n"
            f"At {practice_name}{doctor_phrase}, we believe informed patients make the most confident decisions. "
            f"In this short educational overview, we cover the most common questions patients ask about {clean_topic}, "
            f"including candidacy, preparation, and what to expect.\n\n"
            f"Read our complete guide here: {canonical_url}"
        )

        base_q_title = f"What Should You Know About {clean_topic} in {city}?"
        max_yt_base = 100 - len(" #Shorts")
        youtube_shorts_title = f"{base_q_title[:max_yt_base].rstrip()} #Shorts"

        doctor_line = f"Specialists: {doctors}\n" if doctors else ""
        youtube_shorts_description = (
            f"What should patients in {location_label} know about {clean_topic}? "
            f"At {practice_name}{doctor_phrase}, we break down the key clinical considerations, candidacy factors, "
            f"and typical recovery expectations so patients can prepare for an informed consultation.\n\n"
            f"Key Takeaways Covered:\n"
            f"- How specialists evaluate candidacy for {clean_topic}\n"
            f"- Typical timeline and recovery considerations\n"
            f"- Questions to ask during an individualized consultation in {location_label}\n\n"
            f"Clinical Resource & Citation:\n"
            f"Practice: {practice_name} ({location_label})\n"
            f"{doctor_line}"
            f"Full Educational Guide: {canonical_url}\n\n"
            f"Verbatim Video Transcript:\n"
            f"{voiceover_script}"
        )

        reddit_title = f"[Educational Guide - {location_label}] What patients commonly ask before considering {clean_topic}"[:290]
        reddit_selftext = (
            f"### Overview: {clean_topic} in {location_label}\n\n"
            f"Prospective patients researching **{clean_topic}** in {location_label} frequently ask how to evaluate "
            f"treatment options, what typical recovery timelines look like, and what questions to bring to an initial consultation.\n\n"
            f"#### 1. How Specialists Evaluate Candidacy\n"
            f"At **{practice_name}**{doctor_phrase} in {location_label}, clinicians emphasize that every patient's anatomy, "
            f"baseline health, and long-term goals are distinct. {source_summary or 'A thorough in-person assessment helps determine the most appropriate clinical approach.'}\n\n"
            f"#### 2. Key Questions to Ask During Your Consultation\n"
            f"- What are the realistic outcomes and limitations of {clean_topic}?\n"
            f"- What does the typical day-by-day recovery or follow-up schedule involve?\n"
            f"- How do board-certified specialists tailor the protocol to individual patient needs?\n\n"
            f"#### 3. Clinical References & Full Guide\n"
            f"- **Practice Resource**: [{source_title or clean_topic} — {practice_name} ({location_label})]({canonical_url})\n\n"
            f"*Disclaimer: This post is for educational information in r/{city_subreddit} and is not individualized medical advice. "
            f"Always consult a qualified specialist in person.*"
        )

    return {
        "hook_headline": hook_headline,
        "cta_line": cta_line,
        "voiceover_script": voiceover_script,
        "ig_reels_caption": ig_reels_caption,
        "fb_reels_caption": fb_reels_caption,
        "youtube_shorts_title": youtube_shorts_title,
        "youtube_shorts_description": youtube_shorts_description,
        "reddit_title": reddit_title,
        "reddit_selftext": reddit_selftext,
    }


def generate_educational_post_package(
    *,
    tenant_id: str,
    profile: SocialTenantProfile,
    client_context: dict[str, Any],
    source_override: Optional[dict[str, Any]] = None,
    target_platforms: Optional[list[str]] = None,
    recent_topics: Optional[list[str]] = None,
    weekly_slots_used: Optional[list[int]] = None,
    scheduled_for: Optional[datetime] = None,
    iso_year: Optional[int] = None,
    iso_week: Optional[int] = None,
    slot_index: Optional[int] = None,
    voice_id: Optional[str] = None,
) -> tuple[SocialPost, dict[str, Any], bytes]:
    """Create, normalize, and qualify a complete multi-platform educational social post package.

    Returns `(SocialPost, tier1_and_tier2_report, video_mp4_bytes)`.
    """
    recent = list(recent_topics or [])
    slots_used = list(weekly_slots_used or [])
    source = source_override or select_next_qualifying_source(client_context, recent)

    topic = str(source.get("topic") or source.get("source_title") or "").strip()
    source_type = str(source.get("source_type") or "blog_post")
    source_ref_id = source.get("source_ref_id")
    source_title = str(source.get("source_title") or topic).strip()
    canonical_url = str(
        source.get("canonical_source_url")
        or client_context.get("website_url")
        or ""
    ).strip()
    source_summary = sanitize_spoken_narration(str(source.get("summary") or ""))

    # Tier 1 Qualification Check
    tier1 = qualify_candidate_topic(
        topic=topic,
        source_type=source_type,
        canonical_source_url=canonical_url,
        client_context=client_context,
        weekly_slots_used=slots_used,
        max_posts_per_week=profile.posts_per_week,
        recent_topics=recent,
        requested_platforms=target_platforms or profile.default_platforms or DEFAULT_TARGET_PLATFORMS,
    )
    if not tier1["qualified"]:
        raise ValueError(tier1["reason"])

    eligible_platforms = tier1["eligible_platforms"]
    city = (client_context.get("city") or "").strip()
    city_subreddit = normalize_subreddit_name(
        profile.reddit_city_subreddit or derive_default_city_subreddit(city, client_context.get("industry"))
    )

    practice_name = (
        client_context.get("practice_name")
        or client_context.get("business_name")
        or client_context.get("name")
        or "Our Practice"
    ).strip()
    doctors = (client_context.get("doctors") or "").strip()

    # Build base deterministic copy first, then enrich via Gemini if configured
    copy_bundle = _synthesize_deterministic_copy(
        topic=topic,
        source_title=source_title,
        source_summary=source_summary,
        canonical_url=canonical_url,
        client_context=client_context,
        city_subreddit=city_subreddit,
    )

    strat = dict(getattr(profile, "content_strategy", None) or client_context.get("content_strategy") or {})
    if not strat:
        strat = derive_default_content_strategy(client_context)

    is_agency = is_agency_or_b2b_profile(client_context, strat)
    biz_type = strat.get("business_type") or ("Marketing Agency for Elective Practices" if is_agency else "Specialty Healthcare Practice")
    target_audience = strat.get("target_audience") or (
        "Elective medical practice owners, plastic surgeons, med spa directors, and healthcare practice managers"
        if is_agency
        else f"Prospective patients in {city} seeking clinical care"
    )
    positioning = strat.get("brand_positioning") or ""
    tone = strat.get("tone_of_voice") or (
        "Authoritative, strategic, data-driven practice growth expert"
        if is_agency
        else "Compassionate, clinical, educational medical expert"
    )
    prohibited = strat.get("prohibited_angles") or (
        "Never present this business as a medical clinic, doctor, or treatment facility. Do NOT provide clinical patient advice or describe surgical recovery timelines as if we perform procedures. We are a marketing company serving medical practices."
        if is_agency
        else "Do not make guaranteed clinical outcome claims or provide prescriptive medical advice."
    )
    guidance = strat.get("custom_guidance") or ""
    pillars = strat.get("content_pillars") or []

    prompt = (
        f"You are creating a high-impact social media & LLM-citation package for '{practice_name}'.\n"
        f"Business Type / Identity: {biz_type}\n"
        f"Target Audience: {target_audience}\n"
        f"Brand Positioning: {positioning}\n"
        f"Tone of Voice: {tone}\n"
        + (f"Core Content Pillars: {', '.join(pillars)}\n" if pillars else "")
        + f"CRITICAL CONTENT BOUNDARIES & PROHIBITED ANGLES: {prohibited}\n"
        + (f"Custom Directives / Guidance: {guidance}\n" if guidance else "")
        + f"Topic: {topic}\n"
        + f"Canonical URL: {canonical_url}\n"
        + f"Context Summary: {source_summary}\n"
        + f"Target Reddit City Subreddit: r/{city_subreddit}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Speak directly from the perspective of this Business Type and address the Target Audience.\n"
        "- If this business is an agency or marketing provider serving medical practices, NEVER frame it as a medical clinic or doctor treating patients, and never discuss patient surgical recovery as if this business performs procedures.\n"
        "Return a JSON object with keys: hook_headline, cta_line, voiceover_script (55-90 words, "
        "final publish-ready spoken narration ONLY to be read aloud verbatim to the Target Audience; "
        "NO intro meta-commentary such as 'this is a draft', 'for review', or 'here is the script', "
        "NO stage directions like [Music] or (pause), NO prescriptive commands), "
        "scene_headlines (array of 3 concise on-screen headlines <= 8 words each for Scene 1 Hook, Scene 2 Key Insight, Scene 3 Takeaway), "
        "scene_prompts (array of 3 distinct vertical 9:16 editorial photography prompts tailored to the topic & business type; "
        "pure atmospheric/lifestyle/architectural photography only, NO text, NO logos, NO watermarks, NO graphic medical procedures), "
        "ig_reels_caption (plain text, no markdown asterisks), fb_reels_caption (plain text, no markdown asterisks), "
        "youtube_shorts_title (question format ending with #Shorts, <= 100 chars), "
        "youtube_shorts_description (300+ words including direct Q&A summary, Resource & Citation block with "
        "Business Name, Target, Canonical URL, and 'Verbatim Video Transcript:' followed by the exact voiceover_script), "
        "reddit_title (discussion question for r/" + city_subreddit + "), and reddit_selftext (detailed Markdown Q&A with canonical link)."
    )
    ai_copy = _call_gemini_structured_copy(prompt)
    scene_headlines: list[str] = []
    scene_prompts: list[str] = []
    if isinstance(ai_copy, dict):
        for k in copy_bundle:
            val = ai_copy.get(k)
            if isinstance(val, str) and val.strip():
                copy_bundle[k] = val.strip()
        raw_sh = ai_copy.get("scene_headlines")
        if isinstance(raw_sh, list):
            scene_headlines = [str(x).strip() for x in raw_sh if str(x).strip()][:3]
        raw_sp = ai_copy.get("scene_prompts")
        if isinstance(raw_sp, list):
            scene_prompts = [str(x).strip() for x in raw_sp if str(x).strip()][:3]

    cleaned_vo = sanitize_spoken_narration(copy_bundle.get("voiceover_script") or "")
    if cleaned_vo:
        copy_bundle["voiceover_script"] = cleaned_vo

    if len(scene_headlines) < 3:
        clean_t = strip_raw_markdown_for_social_caption(topic)
        scene_headlines = [
            copy_bundle["hook_headline"],
            f"Key Insights: {clean_t}"[:90],
            f"Next Steps with {practice_name}"[:90],
        ]
    if len(scene_prompts) < 3:
        if is_agency:
            scene_prompts = [
                f"Modern creative marketing agency conference room with warm natural window light, laptop and strategy notebook on clean oak desk, vertical 9:16 editorial photography, shallow depth of field, no text, no logos, topic mood: {topic}",
                f"Healthcare practice owner and digital strategist reviewing analytics dashboard on tablet in bright contemporary office, candid editorial lifestyle photography, vertical 9:16, no text, no logos",
                f"Sleek modern medical office reception lobby with warm architectural lighting and minimalist greenery, welcoming atmosphere, vertical 9:16 editorial photography, no text, no logos",
            ]
        else:
            scene_prompts = [
                f"Bright, serene modern medical clinic consultation lounge with warm natural sunlight, minimalist architecture and soft neutral tones, vertical 9:16 editorial photography, no text, no logos, topic mood: {topic}",
                f"Warm, welcoming specialist doctor in crisp attire speaking reassuringly in a sunlit modern consultation room, shallow depth of field, vertical 9:16 editorial photography, no text, no logos",
                f"Calm, luxury wellness and aesthetic clinic interior with natural stone, soft indoor greenery, and warm ambient lighting, vertical 9:16 editorial photography, no text, no logos",
            ]

    safe_fallback_visual = (
        "Modern minimalist creative agency workspace with warm sunlight and clean architectural lines, vertical 9:16 editorial photography, no text, no logos."
        if is_agency
        else "Bright, welcoming luxury medical clinic interior with soft natural sunlight and calm neutral tones, vertical 9:16 editorial photography, no text, no logos."
    )
    with ThreadPoolExecutor(max_workers=3) as bg_pool:
        bg_futures = [
            bg_pool.submit(
                generate_ai_scene_background_bytes,
                scene_prompts[idx],
                fallback_prompt=safe_fallback_visual,
            )
            for idx in range(3)
        ]
        bg_images = [f.result() for f in bg_futures]

    badge_label = practice_name[:28] if practice_name else ("PRACTICE GROWTH" if is_agency else "PATIENT EDUCATION")
    scene_jpegs = [
        compose_reel_scene_jpeg(
            scene_headlines[idx],
            copy_bundle["cta_line"],
            background_bytes=bg_images[idx],
            badge_label=badge_label,
            theme_index=idx,
        )
        for idx in range(3)
    ]
    selected_voice_id = voice_id or getattr(profile, "default_voice_id", None) or "Kore"
    media_bundle = render_and_normalize_reel_video(
        scene_images=scene_jpegs,
        voiceover_script=copy_bundle["voiceover_script"],
        hook_headline=copy_bundle["hook_headline"],
        cta_line=copy_bundle["cta_line"],
        voice_id=selected_voice_id,
    )

    post_id = str(uuid.uuid4())
    post = SocialPost(
        id=post_id,
        tenant_id=tenant_id,
        source_type=source_type,
        source_ref_id=source_ref_id,
        source_title=source_title,
        canonical_source_url=canonical_url,
        format_type="educational_video_package",
        topic=topic,
        target_platforms=eligible_platforms,
        voiceover_script=copy_bundle["voiceover_script"],
        transcript_srt=media_bundle["transcript_srt"],
        hook_headline=copy_bundle["hook_headline"],
        cta_line=copy_bundle["cta_line"],
        ig_reels_caption=copy_bundle["ig_reels_caption"],
        fb_reels_caption=copy_bundle["fb_reels_caption"],
        youtube_shorts_title=copy_bundle["youtube_shorts_title"],
        youtube_shorts_description=copy_bundle["youtube_shorts_description"],
        reddit_subreddit=city_subreddit,
        reddit_title=copy_bundle["reddit_title"],
        reddit_selftext=copy_bundle["reddit_selftext"],
        video_gcs_object=f"social-studio/{tenant_id}/{post_id}/reel_1080x1920.mp4",
        feed_image_gcs_object=f"social-studio/{tenant_id}/{post_id}/feed_1080x1350.jpg",
        media_delivery_url=f"/api/tenants/{tenant_id}/social/posts/{post_id}/media/video",
        feed_image_delivery_url=f"/api/tenants/{tenant_id}/social/posts/{post_id}/media/image",
        media_metadata=media_bundle["metadata"],
        iso_year=iso_year,
        iso_week=iso_week,
        slot_index=slot_index,
        scheduled_for=scheduled_for,
    )

    # Run Tier 2 6-Point Qualification Suite
    tier2 = qualify_generated_post(
        post,
        client_context,
        recent_topics=recent,
        min_citation_score=profile.min_citation_score,
    )
    healed = False
    if not tier2["qualified"]:
        # 1-pass deterministic self-healing
        post = auto_heal_post_package(post, client_context)
        tier2 = qualify_generated_post(
            post,
            client_context,
            recent_topics=recent,
            min_citation_score=profile.min_citation_score,
        )
        healed = True

    post.llm_citation_score = int(tier2["llm_citation_score"])
    post.qualification_report = {
        "tier1": tier1,
        "tier2": tier2,
        "auto_healed": healed,
    }
    if tier2["qualified"]:
        post.qualification_status = "healed" if healed else "qualified"
        post.status = "qualified_ready"
    else:
        post.qualification_status = "disqualified"
        post.status = "quarantined"

    return post, post.qualification_report, media_bundle["video_bytes"]


def reconcile_post_status_from_results(post: SocialPost) -> SocialPost:
    """Recompute aggregate `post.status`, `post.error`, and `post.published_at` from `post.platform_results`."""
    results = dict(post.platform_results or {})
    targets = [str(p).lower().strip() for p in (post.target_platforms or []) if str(p).strip()]
    if not targets:
        post.error = None
        return post

    failed_statuses = {"failed", "error"}
    published_statuses = {"published", "completed", "success"}
    queued_statuses = {
        "scheduled",
        "publishing",
        "processing",
        "in_progress",
        "queued",
        "queued_for_api_key",
        "queued_for_reddit_api",
    }

    statuses = [str((results.get(p) or {}).get("status") or "").lower().strip() for p in targets]
    platform_errors = [
        f"{p}: {(results.get(p) or {}).get('error') or 'dispatch failed'}"
        for p, st in zip(targets, statuses)
        if st in failed_statuses
    ]

    if all(st in published_statuses for st in statuses):
        post.status = "published"
        post.published_at = post.published_at or utcnow()
        post.error = None
    elif all(st in failed_statuses for st in statuses):
        post.status = "failed"
        post.error = "; ".join(platform_errors) or "All target platforms failed."
    elif any(st in failed_statuses for st in statuses):
        post.status = "partial_failure"
        post.error = "; ".join(platform_errors)
    elif any(st in queued_statuses for st in statuses):
        if any(st in {"publishing", "processing", "in_progress", "queued"} for st in statuses) and not any(
            st in {"scheduled", "queued_for_api_key", "queued_for_reddit_api"} for st in statuses
        ):
            post.status = "publishing"
        else:
            post.status = "scheduled"
        post.error = None
    elif any(st in published_statuses for st in statuses):
        # Some target platforms published while others have not yet been dispatched
        post.status = "scheduled" if post.scheduled_for else "qualified_ready"
        post.error = None
    else:
        if post.status == "publishing":
            post.status = "qualified_ready"
        post.error = None

    return post


def _dispatch_single_platform(
    p_norm: str,
    post: SocialPost,
    profile: SocialTenantProfile,
    video_bytes: bytes,
    *,
    scheduled_iso: Optional[str],
    is_future_reddit: bool,
    sched_dt: Optional[datetime],
) -> tuple[str, dict[str, Any], Optional[str]]:
    if p_norm == "instagram":
        res = publish_platform_video(
            platform="instagram",
            username=profile.uploadpost_username,
            video_bytes=video_bytes,
            title=post.ig_reels_caption,
            scheduled_date=scheduled_iso,
            timezone_name="UTC" if scheduled_iso else None,
        )
        req_id = str(res["request_id"]) if res.get("request_id") else None
        return p_norm, res, req_id

    if p_norm == "facebook":
        res = publish_platform_video(
            platform="facebook",
            username=profile.uploadpost_username,
            video_bytes=video_bytes,
            title=post.fb_reels_caption,
            description=post.fb_reels_caption,
            facebook_page_id=profile.facebook_page_id,
            scheduled_date=scheduled_iso,
            timezone_name="UTC" if scheduled_iso else None,
        )
        req_id = str(res["request_id"]) if res.get("request_id") else None
        return p_norm, res, req_id

    if p_norm == "youtube":
        res = publish_platform_video(
            platform="youtube",
            username=profile.uploadpost_username,
            video_bytes=video_bytes,
            title=post.youtube_shorts_title,
            description=post.youtube_shorts_description,
            scheduled_date=scheduled_iso,
            timezone_name="UTC" if scheduled_iso else None,
        )
        req_id = str(res["request_id"]) if res.get("request_id") else None
        return p_norm, res, req_id

    if p_norm == "reddit":
        sr = normalize_subreddit_name(post.reddit_subreddit or profile.reddit_city_subreddit)
        if not sr:
            raise ValueError("A valid target city subreddit name is required.")
        if is_future_reddit and sched_dt is not None:
            sched_utc_iso = sched_dt.astimezone(timezone.utc).isoformat()
            res = {
                "ok": True,
                "platform": "reddit",
                "subreddit": f"r/{sr}",
                "status": "scheduled",
                "scheduled_for": sched_utc_iso,
                "note": f"Scheduled for r/{sr} at {sched_utc_iso}",
            }
            return p_norm, res, None

        res = publish_city_subreddit_educational_post(
            subreddit=sr,
            title=post.reddit_title,
            selftext=post.reddit_selftext,
            flair_id=profile.reddit_flair_id,
            flair_text=profile.reddit_flair_text,
        )
        secondary_results = []
        for sec_sr in (profile.reddit_secondary_subreddits or []):
            sec_clean = normalize_subreddit_name(sec_sr)
            if sec_clean and sec_clean.lower() != sr.lower():
                try:
                    sec_res = publish_city_subreddit_educational_post(
                        subreddit=sec_clean,
                        title=post.reddit_title,
                        selftext=post.reddit_selftext,
                        flair_id=profile.reddit_flair_id,
                        flair_text=profile.reddit_flair_text,
                    )
                    secondary_results.append(sec_res)
                except Exception as sec_exc:
                    logger.warning("Secondary subreddit r/%s publish failed: %s", sec_clean, sec_exc)
                    secondary_results.append({"ok": False, "subreddit": f"r/{sec_clean}", "error": str(sec_exc)})
        if secondary_results:
            res["secondary_posts"] = secondary_results
        req_id = str(res["post_id"]) if res.get("post_id") else None
        return p_norm, res, req_id

    raise ValueError(f"Unsupported social platform: {p_norm}")


def dispatch_qualified_post_to_platforms(
    post: SocialPost,
    profile: SocialTenantProfile,
    video_bytes: bytes,
    *,
    only_platforms: Optional[list[str]] = None,
    publish_now: bool = False,
) -> SocialPost:
    """Dispatch a qualified post across its target platforms with per-platform failure isolation."""
    if post.qualification_status not in {"qualified", "healed"}:
        post.status = "quarantined"
        post.error = "Post did not pass all 6 submission qualification gates."
        return post

    now = utcnow()
    sched_dt: Optional[datetime] = None
    if not publish_now and post.scheduled_for is not None:
        sched_dt = (
            post.scheduled_for
            if post.scheduled_for.tzinfo is not None
            else post.scheduled_for.replace(tzinfo=timezone.utc)
        )
    is_future = bool(sched_dt is not None and sched_dt > now)
    scheduled_iso = sched_dt.astimezone(timezone.utc).isoformat() if is_future and sched_dt is not None else None

    platforms_to_run = [str(p).lower().strip() for p in (only_platforms or post.target_platforms) if str(p).strip()]
    results = dict(post.platform_results or {})
    request_ids = dict(post.uploadpost_request_ids or {})

    pending_platforms: list[str] = []
    for p_norm in platforms_to_run:
        existing = results.get(p_norm) or {}
        if only_platforms is None and existing.get("status") in {"published", "scheduled"}:
            continue
        pending_platforms.append(p_norm)

    def _run_one(p_norm: str) -> tuple[str, dict[str, Any], Optional[str]]:
        try:
            return _dispatch_single_platform(
                p_norm,
                post,
                profile,
                video_bytes,
                scheduled_iso=scheduled_iso,
                is_future_reddit=is_future,
                sched_dt=sched_dt,
            )
        except (UploadPostError, RedditPublishError, ValueError) as exc:
            err_text = str(exc)
            return (
                p_norm,
                {
                    "ok": False,
                    "platform": p_norm,
                    "status": "failed",
                    "error": err_text,
                },
                None,
            )

    if len(pending_platforms) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(pending_platforms))) as pool:
            outcomes = list(pool.map(_run_one, pending_platforms))
    else:
        outcomes = [_run_one(p) for p in pending_platforms]

    for p_norm, res, req_id in outcomes:
        results[p_norm] = res
        if req_id:
            request_ids[p_norm] = req_id

    post.platform_results = results
    post.uploadpost_request_ids = request_ids
    reconcile_post_status_from_results(post)
    if publish_now and post.status == "published":
        post.scheduled_for = None

    return post


def resolve_slot_weekday(
    slot_index: int,
    schedule_days: Optional[list[str]] = None,
) -> tuple[int, str]:
    """Resolve `(iso_weekday, weekday_label)` for slot 1..3 from `schedule_days`."""
    user_days = normalize_schedule_days(schedule_days or DEFAULT_SCHEDULE_DAYS)
    ordered = list(user_days)
    # Pad remaining slots at the end without overriding the user's selected first days
    for fallback_day in DEFAULT_SCHEDULE_DAYS:
        if len(ordered) >= 3:
            break
        if fallback_day not in ordered:
            ordered.append(fallback_day)
    idx = max(1, min(int(slot_index), len(ordered))) - 1
    abbr = ordered[idx]
    iso_weekday = WEEKDAY_ABBR_TO_ISO.get(abbr, 1)
    return iso_weekday, ISO_TO_WEEKDAY_LABEL.get(iso_weekday, "Monday")


def compute_weekly_slot_datetime(
    iso_year: int,
    iso_week: int,
    slot_index: int,
    *,
    preferred_time: str = "09:30",
    tz_name: str = "America/New_York",
    schedule_days: Optional[list[str]] = None,
) -> datetime:
    """Compute the UTC scheduled datetime for slot 1..3 using the tenant's schedule_days."""
    iso_weekday, _label = resolve_slot_weekday(int(slot_index), schedule_days)
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    parts = (preferred_time or "09:30").split(":")
    hour = int(parts[0]) if parts and parts[0].isdigit() else 9
    minute = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30
    hour = min(max(hour, 0), 23)
    minute = min(max(minute, 0), 59)

    target_date = datetime.fromisocalendar(iso_year, iso_week, iso_weekday)
    local_dt = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=tz,
    )
    now_utc = utcnow()
    scheduled_utc = local_dt.astimezone(timezone.utc)
    if scheduled_utc <= now_utc:
        scheduled_utc = now_utc + timedelta(hours=2 * int(slot_index))
    return scheduled_utc
