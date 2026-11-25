"""
Domain & Persistence Models for the EMM Social Studio Microservice.

Supports two persistence backends seamlessly:
1. Standalone Microservice Mode: Uses dedicated SQLAlchemy tables
   (`social_tenant_profiles`, `social_weekly_slots`, `social_posts`) when running
   against a dedicated microservice database.
2. Embedded / Shared Cloud SQL Mode: Uses `BastionAuditEvent` rows with dedicated
   `event_type` discriminators (`social_studio_profile`, `social_studio_slot`,
   `social_studio_post`) to preserve zero-DDL production schema compatibility
   (`check_database_schema`) and least-privilege `marketing_production_app` permissions.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    inspect,
)
from sqlalchemy.orm import DeclarativeBase, Session


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_dt(val: Optional[datetime | str]) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.isoformat()
    return str(val)


def _parse_dt(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str) and val.strip():
        try:
            dt = datetime.fromisoformat(val.strip().replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def derive_default_city_subreddit(city: Optional[str], industry: Optional[str] = None) -> str:
    """Derive a clean owned city subreddit slug from client city & industry."""
    raw_city = re.sub(r"[^a-zA-Z0-9]", "", (city or "").title())
    if not raw_city:
        raw_city = "Local"
    ind = (industry or "").lower()
    if any(k in ind for k in ("plastic", "cosmetic", "medspa", "aesthetic", "dermatol")):
        suffix = "Aesthetics"
    elif any(k in ind for k in ("dental", "dentist", "orthodon")):
        suffix = "DentalHealth"
    else:
        suffix = "HealthGuide"
    combined = f"{raw_city}{suffix}"
    # Reddit subreddit names have a 21-char max limit
    if len(combined) > 21:
        combined = f"{raw_city[:15]}Health"[:21]
    return combined


def derive_uploadpost_username(tenant_id: str, slug: Optional[str] = None) -> str:
    """Deterministic Upload-Post profile username per client tenant."""
    clean_slug = re.sub(r"[^a-z0-9_]", "_", (slug or "").lower()).strip("_")[:20]
    short_id = re.sub(r"[^a-z0-9]", "", tenant_id.lower())[:8]
    if clean_slug:
        return f"emm_{clean_slug}_{short_id}"
    return f"emm_client_{short_id}"


DEFAULT_TARGET_PLATFORMS = ["instagram", "facebook", "youtube", "reddit"]
DEFAULT_SCHEDULE_DAYS = ["mon", "wed", "fri"]
MAX_POSTS_PER_WEEK = 3

WEEKDAY_ABBR_TO_ISO = {
    "mon": 1, "monday": 1,
    "tue": 2, "tues": 2, "tuesday": 2,
    "wed": 3, "wednesday": 3,
    "thu": 4, "thur": 4, "thurs": 4, "thursday": 4,
    "fri": 5, "friday": 5,
    "sat": 6, "saturday": 6,
    "sun": 7, "sunday": 7,
}
ISO_TO_WEEKDAY_ABBR = {
    1: "mon",
    2: "tue",
    3: "wed",
    4: "thu",
    5: "fri",
    6: "sat",
    7: "sun",
}
ISO_TO_WEEKDAY_LABEL = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday",
}


def normalize_schedule_days(days: Optional[list[str]]) -> list[str]:
    """Normalize weekday tokens into sorted unique 3-letter abbreviations (e.g. ['mon', 'wed', 'fri'])."""
    iso_days: list[int] = []
    for raw in (days or []):
        iso = WEEKDAY_ABBR_TO_ISO.get(str(raw).strip().lower())
        if iso and iso not in iso_days:
            iso_days.append(iso)
    if not iso_days:
        return list(DEFAULT_SCHEDULE_DAYS)
    iso_days.sort()
    return [ISO_TO_WEEKDAY_ABBR[d] for d in iso_days]


def is_agency_or_b2b_profile(
    client_context: Optional[dict[str, Any]] = None,
    strategy: Optional[dict[str, Any]] = None,
) -> bool:
    """Determine whether the client/tenant is a marketing agency, SaaS, or B2B provider (like Elevated)."""
    strat = strategy or {}
    ctx = client_context or {}

    # 1. Explicit business_type or prohibited_angles in content strategy
    biz_type = str(strat.get("business_type") or "").lower()
    prohibited = str(strat.get("prohibited_angles") or "").lower()
    for needle in ("agency", "marketing company", "marketing agency", "b2b", "consulting", "software", "saas"):
        if needle in biz_type:
            return True
    if "not a medical clinic" in prohibited or "never present this company as a medical clinic" in prohibited or "never present this business as a medical clinic" in prohibited or "agency serving" in prohibited:
        return True

    # 2. Inspect tenant name, business name, practice name, and industry in context
    name = str(ctx.get("name") or "").lower()
    biz_name = str(ctx.get("business_name") or "").lower()
    practice_name = str(ctx.get("practice_name") or "").lower()
    industry = str(ctx.get("industry") or "").lower()

    combined_text = f"{name} {biz_name} {practice_name} {industry}"
    agency_keywords = ("marketing", "agency", "advertising", "consulting", "seo", "media agency", "practice growth")
    if any(k in combined_text for k in agency_keywords):
        return True

    return False


def derive_default_content_strategy(client_context: dict[str, Any]) -> dict[str, Any]:
    """Infer an intelligent, grounded baseline Content Strategy from tenant intelligence."""
    brand_name = (
        client_context.get("business_name")
        or client_context.get("practice_name")
        or client_context.get("name")
        or "Our Company"
    ).strip()
    city = (client_context.get("city") or "").strip()
    state = (client_context.get("state") or "").strip()
    location_str = f"{city}, {state}".strip(", ") if city else "our local area"
    industry = (client_context.get("industry") or "").strip()
    services = [str(s).strip() for s in (client_context.get("services") or []) if str(s).strip()]
    if not services:
        services = [str(s).strip() for s in (client_context.get("gbp_services") or []) if str(s).strip()]

    is_agency = is_agency_or_b2b_profile(client_context, None)

    if is_agency:
        return {
            "business_type": "Marketing Agency for Medical & Elective Practices",
            "target_audience": "Elective medical practice owners, plastic surgeons, med spa medical directors, and healthcare practice managers",
            "brand_positioning": (
                f"{brand_name} is a high-performance marketing and practice growth partner helping elective healthcare "
                "practices, surgeons, and aesthetic clinics scale patient volume through proven digital marketing, SEO, and paid media."
            ),
            "tone_of_voice": "Authoritative, strategic, data-driven practice growth expert",
            "content_pillars": [
                "Elective Medical Practice Growth & Patient Acquisition",
                "Medical SEO & AI Search Visibility for Clinics",
                "High-ROI Google & Social Ads for Surgeons & Med Spas",
                "Practice Website Conversion & Consultation Booking Optimization",
                "Healthcare Reputation, Review Generation & Local Authority",
            ],
            "prohibited_angles": (
                "Never present this business as a medical clinic, doctor, or treatment facility. "
                "Do NOT provide clinical medical advice or discuss patient clinical recovery timelines as if we perform procedures. "
                "This business provides marketing, patient acquisition, and practice growth services for elective healthcare practices."
            ),
            "topic_recommendations": [
                f"How Elective Medical Clinics Can Dominate Local Search in {city or '2026'}",
                "Why Generic Marketing Agencies Waste Medical Ad Budgets",
                "5 Proven Ways Plastic Surgeons Convert Website Traffic Into Consultations",
                "HIPAA-Compliant Ad Strategies That Fill Clinic Schedules",
                "The Complete Patient Acquisition Blueprint for Medical Spas",
                "How AI Search is Changing How Patients Find Elective Surgeons",
                "Cost Per Acquisition vs Lifetime Value: What Medical Practices Must Know",
            ],
            "custom_guidance": "Focus on ROI, consultation volume, patient acquisition cost, and practice authority. Address practice owners, surgeons, and clinic managers directly.",
        }
    else:
        inferred_type = industry or "Aesthetic & Specialty Medical Practice"
        top_services = services[:5] if services else ["Elective Consultations", "Specialty Care"]
        return {
            "business_type": inferred_type,
            "target_audience": f"Prospective patients in {location_str} seeking safe, expert elective treatments and individualized clinical care",
            "brand_positioning": f"{brand_name} provides personalized, patient-centric care focused on natural results, clinical safety, and exceptional outcomes in {location_str}.",
            "tone_of_voice": "Compassionate, clinical, educational, reassuring medical expert",
            "content_pillars": [
                f"{svc} Educational Guides & Candidacy" for svc in top_services[:3]
            ] + [
                "Patient Safety, Preparation & Recovery Timelines",
                "What to Expect During Your In-Person Consultation",
            ],
            "prohibited_angles": (
                "Do not make guaranteed clinical outcome claims or promise 100% cure/safety. "
                "Do not provide prescriptive diagnostic advice over social media. "
                "Always direct specific treatment decisions to an in-person clinical consultation."
            ),
            "topic_recommendations": [
                f"What Patients in {city or 'Our Area'} Should Know Before Considering {top_services[0]}",
                f"How to Prepare for Your First Consultation at {brand_name}",
                f"Comparing Your Options: What to Look for in {top_services[0]}",
                f"Realistic Recovery Timelines: What Week 1 Truly Looks Like",
            ],
            "custom_guidance": "Emphasize board-certified standards, patient education, realistic expectations, and in-person consultations.",
        }

EVENT_TYPE_PROFILE = "social_studio_profile"
EVENT_TYPE_SLOT = "social_studio_slot"
EVENT_TYPE_POST = "social_studio_post"


class SocialStudioBase(DeclarativeBase):
    """Standalone microservice declarative base (isolated from main Base.metadata)."""
    pass


class StandaloneBastionAuditEventBase(DeclarativeBase):
    """Separate base for mapping existing `bastion_audit_events` in standalone Cloud SQL mode."""
    pass


class StandaloneBastionAuditEvent(StandaloneBastionAuditEventBase):
    __tablename__ = "bastion_audit_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False, index=True)
    actor = Column(String(36), nullable=False, default="system")
    event_type = Column(String(50), nullable=False)
    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class SocialTenantProfileRow(SocialStudioBase):
    __tablename__ = "social_tenant_profiles"

    tenant_id = Column(String(64), primary_key=True)
    uploadpost_username = Column(String(64), nullable=False, unique=True)
    connected_platforms_json = Column(Text, nullable=False, default="[]")
    facebook_page_id = Column(String(128), nullable=True)
    facebook_page_name = Column(String(255), nullable=True)
    instagram_account_id = Column(String(128), nullable=True)
    instagram_handle = Column(String(128), nullable=True)
    youtube_channel_id = Column(String(128), nullable=True)
    youtube_channel_title = Column(String(255), nullable=True)
    reddit_city_subreddit = Column(String(64), nullable=False, default="")
    reddit_secondary_subreddits_json = Column(Text, nullable=False, default="[]")
    reddit_flair_id = Column(String(128), nullable=True)
    reddit_flair_text = Column(String(128), nullable=True)
    reddit_mode = Column(String(32), nullable=False, default="official_api")
    last_jwt_url = Column(Text, nullable=True)
    last_jwt_expires_at = Column(DateTime(timezone=True), nullable=True)
    automation_enabled = Column(Boolean, nullable=False, default=True)
    posts_per_week = Column(Integer, nullable=False, default=3)
    schedule_days_json = Column(Text, nullable=False, default='["mon", "wed", "fri"]')
    preferred_time = Column(String(16), nullable=False, default="09:30")
    timezone = Column(String(64), nullable=False, default="America/New_York")
    auto_repurpose_blogs = Column(Boolean, nullable=False, default=True)
    min_citation_score = Column(Integer, nullable=False, default=80)
    default_voice_id = Column(String(32), nullable=False, default="Kore")
    default_platforms_json = Column(
        Text, nullable=False, default='["instagram", "facebook", "youtube", "reddit"]'
    )
    content_strategy_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class SocialWeeklySlotRow(SocialStudioBase):
    __tablename__ = "social_weekly_slots"
    __table_args__ = (
        UniqueConstraint("tenant_id", "iso_year", "iso_week", "slot_index", name="uq_social_weekly_slot"),
    )

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(64), nullable=False, index=True)
    iso_year = Column(Integer, nullable=False)
    iso_week = Column(Integer, nullable=False)
    slot_index = Column(Integer, nullable=False)  # 1, 2, or 3
    post_id = Column(String(64), nullable=True)
    scheduled_for = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(32), nullable=False, default="claimed")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class SocialPostRow(SocialStudioBase):
    __tablename__ = "social_posts"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(64), nullable=False, index=True)
    source_type = Column(String(32), nullable=False, default="blog_post")
    source_ref_id = Column(String(128), nullable=True)
    source_title = Column(String(500), nullable=False, default="")
    canonical_source_url = Column(Text, nullable=False, default="")
    format_type = Column(String(32), nullable=False, default="educational_video_package")
    topic = Column(String(500), nullable=False, default="")
    target_platforms_json = Column(Text, nullable=False, default="[]")

    # Video & Transcript payload
    voiceover_script = Column(Text, nullable=False, default="")
    transcript_srt = Column(Text, nullable=False, default="")
    hook_headline = Column(String(300), nullable=False, default="")
    cta_line = Column(String(300), nullable=False, default="")

    # Per-platform tailored copy
    ig_reels_caption = Column(Text, nullable=False, default="")
    fb_reels_caption = Column(Text, nullable=False, default="")
    youtube_shorts_title = Column(String(150), nullable=False, default="")
    youtube_shorts_description = Column(Text, nullable=False, default="")
    reddit_subreddit = Column(String(64), nullable=False, default="")
    reddit_title = Column(String(300), nullable=False, default="")
    reddit_selftext = Column(Text, nullable=False, default="")

    # Media metadata & GCS storage references
    video_gcs_object = Column(String(500), nullable=True)
    feed_image_gcs_object = Column(String(500), nullable=True)
    media_delivery_url = Column(Text, nullable=True)
    feed_image_delivery_url = Column(Text, nullable=True)
    media_metadata_json = Column(Text, nullable=False, default="{}")

    # Qualification & LLM Citation scoring
    qualification_status = Column(String(32), nullable=False, default="pending")
    llm_citation_score = Column(Integer, nullable=False, default=0)
    qualification_report_json = Column(Text, nullable=False, default="{}")

    # Scheduling & Per-platform isolated execution status
    status = Column(String(32), nullable=False, default="generating")
    iso_year = Column(Integer, nullable=True)
    iso_week = Column(Integer, nullable=True)
    slot_index = Column(Integer, nullable=True)
    scheduled_for = Column(DateTime(timezone=True), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    uploadpost_request_ids_json = Column(Text, nullable=False, default="{}")
    platform_results_json = Column(Text, nullable=False, default="{}")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


@dataclass
class SocialTenantProfile:
    tenant_id: str
    uploadpost_username: str
    connected_platforms: list[str] = field(default_factory=list)
    facebook_page_id: Optional[str] = None
    facebook_page_name: Optional[str] = None
    available_facebook_pages: list[dict[str, str]] = field(default_factory=list)
    instagram_account_id: Optional[str] = None
    instagram_handle: Optional[str] = None
    youtube_channel_id: Optional[str] = None
    youtube_channel_title: Optional[str] = None
    reddit_city_subreddit: str = ""
    reddit_secondary_subreddits: list[str] = field(default_factory=list)
    reddit_flair_id: Optional[str] = None
    reddit_flair_text: Optional[str] = None
    reddit_mode: str = "official_api"
    last_jwt_url: Optional[str] = None
    last_jwt_expires_at: Optional[datetime] = None
    automation_enabled: bool = True
    posts_per_week: int = 3
    schedule_days: list[str] = field(default_factory=lambda: list(DEFAULT_SCHEDULE_DAYS))
    preferred_time: str = "09:30"
    timezone: str = "America/New_York"
    auto_repurpose_blogs: bool = True
    min_citation_score: int = 80
    default_voice_id: str = "Kore"
    default_platforms: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_PLATFORMS))
    content_strategy: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "uploadpost_username": self.uploadpost_username,
            "connected_platforms": list(self.connected_platforms),
            "facebook_page_id": self.facebook_page_id,
            "facebook_page_name": self.facebook_page_name,
            "available_facebook_pages": list(self.available_facebook_pages),
            "instagram_account_id": self.instagram_account_id,
            "instagram_handle": self.instagram_handle,
            "youtube_channel_id": self.youtube_channel_id,
            "youtube_channel_title": self.youtube_channel_title,
            "reddit_city_subreddit": self.reddit_city_subreddit,
            "reddit_secondary_subreddits": list(self.reddit_secondary_subreddits),
            "reddit_flair_id": self.reddit_flair_id,
            "reddit_flair_text": self.reddit_flair_text,
            "reddit_mode": self.reddit_mode,
            "last_jwt_url": self.last_jwt_url,
            "last_jwt_expires_at": _fmt_dt(self.last_jwt_expires_at),
            "automation_enabled": bool(self.automation_enabled),
            "posts_per_week": min(max(int(self.posts_per_week or 3), 1), MAX_POSTS_PER_WEEK),
            "schedule_days": list(self.schedule_days),
            "preferred_time": self.preferred_time,
            "timezone": self.timezone,
            "auto_repurpose_blogs": bool(self.auto_repurpose_blogs),
            "min_citation_score": int(self.min_citation_score or 80),
            "default_voice_id": str(self.default_voice_id or "Kore"),
            "default_platforms": list(self.default_platforms),
            "content_strategy": dict(self.content_strategy or {}),
            "created_at": _fmt_dt(self.created_at),
            "updated_at": _fmt_dt(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SocialTenantProfile":
        return cls(
            tenant_id=str(data["tenant_id"]),
            uploadpost_username=str(data.get("uploadpost_username") or derive_uploadpost_username(str(data["tenant_id"]))),
            connected_platforms=list(data.get("connected_platforms") or []),
            facebook_page_id=data.get("facebook_page_id"),
            facebook_page_name=data.get("facebook_page_name"),
            available_facebook_pages=list(data.get("available_facebook_pages") or []),
            instagram_account_id=data.get("instagram_account_id"),
            instagram_handle=data.get("instagram_handle"),
            youtube_channel_id=data.get("youtube_channel_id"),
            youtube_channel_title=data.get("youtube_channel_title"),
            reddit_city_subreddit=str(data.get("reddit_city_subreddit") or ""),
            reddit_secondary_subreddits=list(data.get("reddit_secondary_subreddits") or []),
            reddit_flair_id=data.get("reddit_flair_id"),
            reddit_flair_text=data.get("reddit_flair_text"),
            reddit_mode=str(data.get("reddit_mode") or "official_api"),
            last_jwt_url=data.get("last_jwt_url"),
            last_jwt_expires_at=_parse_dt(data.get("last_jwt_expires_at")),
            automation_enabled=bool(data.get("automation_enabled", True)),
            posts_per_week=min(max(int(data.get("posts_per_week", 3)), 1), MAX_POSTS_PER_WEEK),
            schedule_days=normalize_schedule_days(data.get("schedule_days")),
            preferred_time=str(data.get("preferred_time") or "09:30"),
            timezone=str(data.get("timezone") or "America/New_York"),
            auto_repurpose_blogs=bool(data.get("auto_repurpose_blogs", True)),
            min_citation_score=int(data.get("min_citation_score", 80)),
            default_voice_id=str(data.get("default_voice_id") or "Kore"),
            default_platforms=list(data.get("default_platforms") or DEFAULT_TARGET_PLATFORMS),
            content_strategy=dict(data.get("content_strategy") or {}),
            created_at=_parse_dt(data.get("created_at")) or utcnow(),
            updated_at=_parse_dt(data.get("updated_at")) or utcnow(),
        )


@dataclass
class SocialPost:
    id: str
    tenant_id: str
    source_type: str = "blog_post"  # blog_post | service_page | gbp_service | custom
    source_ref_id: Optional[str] = None
    source_title: str = ""
    canonical_source_url: str = ""
    format_type: str = "educational_video_package"
    topic: str = ""
    target_platforms: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_PLATFORMS))

    # Spoken script & timed transcript
    voiceover_script: str = ""
    transcript_srt: str = ""
    hook_headline: str = ""
    cta_line: str = ""

    # Per-platform tailored copy
    ig_reels_caption: str = ""
    fb_reels_caption: str = ""
    youtube_shorts_title: str = ""
    youtube_shorts_description: str = ""
    reddit_subreddit: str = ""
    reddit_title: str = ""
    reddit_selftext: str = ""

    # Media metadata & URLs
    video_gcs_object: Optional[str] = None
    feed_image_gcs_object: Optional[str] = None
    media_delivery_url: Optional[str] = None
    feed_image_delivery_url: Optional[str] = None
    media_metadata: dict[str, Any] = field(default_factory=dict)

    # Qualification & LLM Citation readiness
    qualification_status: str = "pending"  # qualified | healed | disqualified | pending
    llm_citation_score: int = 0
    qualification_report: dict[str, Any] = field(default_factory=dict)

    # Scheduling & Delivery
    status: str = "generating"  # generating | qualified_ready | scheduled | publishing | published | partial_failure | quarantined | cancelled
    iso_year: Optional[int] = None
    iso_week: Optional[int] = None
    slot_index: Optional[int] = None
    scheduled_for: Optional[datetime] = None
    published_at: Optional[datetime] = None
    uploadpost_request_ids: dict[str, str] = field(default_factory=dict)
    platform_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "source_type": self.source_type,
            "source_ref_id": self.source_ref_id,
            "source_title": self.source_title,
            "canonical_source_url": self.canonical_source_url,
            "format_type": self.format_type,
            "topic": self.topic,
            "target_platforms": list(self.target_platforms),
            "voiceover_script": self.voiceover_script,
            "transcript_srt": self.transcript_srt,
            "hook_headline": self.hook_headline,
            "cta_line": self.cta_line,
            "ig_reels_caption": self.ig_reels_caption,
            "fb_reels_caption": self.fb_reels_caption,
            "youtube_shorts_title": self.youtube_shorts_title,
            "youtube_shorts_description": self.youtube_shorts_description,
            "reddit_subreddit": self.reddit_subreddit,
            "reddit_title": self.reddit_title,
            "reddit_selftext": self.reddit_selftext,
            "video_gcs_object": self.video_gcs_object,
            "feed_image_gcs_object": self.feed_image_gcs_object,
            "media_delivery_url": self.media_delivery_url,
            "feed_image_delivery_url": self.feed_image_delivery_url,
            "media_metadata": dict(self.media_metadata),
            "qualification_status": self.qualification_status,
            "llm_citation_score": int(self.llm_citation_score or 0),
            "qualification_report": dict(self.qualification_report),
            "status": self.status,
            "iso_year": self.iso_year,
            "iso_week": self.iso_week,
            "slot_index": self.slot_index,
            "scheduled_for": _fmt_dt(self.scheduled_for),
            "published_at": _fmt_dt(self.published_at),
            "uploadpost_request_ids": dict(self.uploadpost_request_ids),
            "platform_results": dict(self.platform_results),
            "error": self.error,
            "created_at": _fmt_dt(self.created_at),
            "updated_at": _fmt_dt(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SocialPost":
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            tenant_id=str(data["tenant_id"]),
            source_type=str(data.get("source_type") or "blog_post"),
            source_ref_id=data.get("source_ref_id"),
            source_title=str(data.get("source_title") or ""),
            canonical_source_url=str(data.get("canonical_source_url") or ""),
            format_type=str(data.get("format_type") or "educational_video_package"),
            topic=str(data.get("topic") or ""),
            target_platforms=list(data.get("target_platforms") or DEFAULT_TARGET_PLATFORMS),
            voiceover_script=str(data.get("voiceover_script") or ""),
            transcript_srt=str(data.get("transcript_srt") or ""),
            hook_headline=str(data.get("hook_headline") or ""),
            cta_line=str(data.get("cta_line") or ""),
            ig_reels_caption=str(data.get("ig_reels_caption") or ""),
            fb_reels_caption=str(data.get("fb_reels_caption") or ""),
            youtube_shorts_title=str(data.get("youtube_shorts_title") or ""),
            youtube_shorts_description=str(data.get("youtube_shorts_description") or ""),
            reddit_subreddit=str(data.get("reddit_subreddit") or ""),
            reddit_title=str(data.get("reddit_title") or ""),
            reddit_selftext=str(data.get("reddit_selftext") or ""),
            video_gcs_object=data.get("video_gcs_object"),
            feed_image_gcs_object=data.get("feed_image_gcs_object"),
            media_delivery_url=data.get("media_delivery_url"),
            feed_image_delivery_url=data.get("feed_image_delivery_url"),
            media_metadata=dict(data.get("media_metadata") or {}),
            qualification_status=str(data.get("qualification_status") or "pending"),
            llm_citation_score=int(data.get("llm_citation_score") or 0),
            qualification_report=dict(data.get("qualification_report") or {}),
            status=str(data.get("status") or "generating"),
            iso_year=data.get("iso_year"),
            iso_week=data.get("iso_week"),
            slot_index=data.get("slot_index"),
            scheduled_for=_parse_dt(data.get("scheduled_for")),
            published_at=_parse_dt(data.get("published_at")),
            uploadpost_request_ids=dict(data.get("uploadpost_request_ids") or {}),
            platform_results=dict(data.get("platform_results") or {}),
            error=data.get("error"),
            created_at=_parse_dt(data.get("created_at")) or utcnow(),
            updated_at=_parse_dt(data.get("updated_at")) or utcnow(),
        )


class SocialStudioStore:
    """Dual-mode persistence store supporting standalone tables or BastionAuditEvent."""

    @staticmethod
    def _get_audit_event_model():
        try:
            import importlib
            mod = importlib.import_module("models.bastion")
            return getattr(mod, "BastionAuditEvent")
        except Exception:
            return StandaloneBastionAuditEvent

    @staticmethod
    def _has_standalone_tables(db: Session) -> bool:
        try:
            conn = db.connection()
            return bool(inspect(conn).has_table("social_posts"))
        except Exception:
            return False

    @classmethod
    def get_or_create_profile(
        cls,
        db: Session,
        tenant_id: str,
        *,
        tenant_slug: Optional[str] = None,
        city: Optional[str] = None,
        industry: Optional[str] = None,
        commit: bool = True,
    ) -> SocialTenantProfile:
        default_username = derive_uploadpost_username(tenant_id, tenant_slug)
        default_subreddit = derive_default_city_subreddit(city, industry)

        if cls._has_standalone_tables(db):
            row = db.query(SocialTenantProfileRow).filter_by(tenant_id=tenant_id).first()
            if row is None:
                row = SocialTenantProfileRow(
                    tenant_id=tenant_id,
                    uploadpost_username=default_username,
                    reddit_city_subreddit=default_subreddit,
                )
                db.add(row)
                if commit:
                    db.commit()
                    db.refresh(row)
                else:
                    db.flush()
            return SocialTenantProfile(
                tenant_id=row.tenant_id,
                uploadpost_username=row.uploadpost_username,
                connected_platforms=json.loads(row.connected_platforms_json or "[]"),
                facebook_page_id=row.facebook_page_id,
                facebook_page_name=row.facebook_page_name,
                instagram_account_id=row.instagram_account_id,
                instagram_handle=row.instagram_handle,
                youtube_channel_id=row.youtube_channel_id,
                youtube_channel_title=row.youtube_channel_title,
                reddit_city_subreddit=row.reddit_city_subreddit or default_subreddit,
                reddit_secondary_subreddits=json.loads(row.reddit_secondary_subreddits_json or "[]"),
                reddit_flair_id=row.reddit_flair_id,
                reddit_flair_text=row.reddit_flair_text,
                reddit_mode=row.reddit_mode or "official_api",
                last_jwt_url=row.last_jwt_url,
                last_jwt_expires_at=_parse_dt(row.last_jwt_expires_at),
                automation_enabled=bool(row.automation_enabled),
                posts_per_week=min(max(int(row.posts_per_week or 3), 1), MAX_POSTS_PER_WEEK),
                schedule_days=json.loads(row.schedule_days_json or '["mon", "wed", "fri"]'),
                preferred_time=row.preferred_time or "09:30",
                timezone=row.timezone or "America/New_York",
                auto_repurpose_blogs=bool(row.auto_repurpose_blogs),
                min_citation_score=int(row.min_citation_score or 80),
                default_voice_id=getattr(row, "default_voice_id", None) or "Kore",
                default_platforms=json.loads(row.default_platforms_json or "[]") or list(DEFAULT_TARGET_PLATFORMS),
                content_strategy=json.loads(getattr(row, "content_strategy_json", None) or "{}") if getattr(row, "content_strategy_json", None) else {},
                created_at=_parse_dt(row.created_at) or utcnow(),
                updated_at=_parse_dt(row.updated_at) or utcnow(),
            )

        # Shared DB zero-DDL mode via BastionAuditEvent
        BastionAuditEvent = cls._get_audit_event_model()

        event = (
            db.query(BastionAuditEvent)
            .filter(
                BastionAuditEvent.tenant_id == tenant_id,
                BastionAuditEvent.event_type == EVENT_TYPE_PROFILE,
            )
            .order_by(BastionAuditEvent.created_at.desc())
            .first()
        )
        if event is None:
            profile = SocialTenantProfile(
                tenant_id=tenant_id,
                uploadpost_username=default_username,
                reddit_city_subreddit=default_subreddit,
            )
            cls.save_profile(db, profile, commit=commit)
            return profile
        payload = dict(event.payload or {})
        payload.setdefault("tenant_id", tenant_id)
        if not payload.get("uploadpost_username"):
            payload["uploadpost_username"] = default_username
        if not payload.get("reddit_city_subreddit") and default_subreddit:
            payload["reddit_city_subreddit"] = default_subreddit
        return SocialTenantProfile.from_dict(payload)

    @classmethod
    def save_profile(cls, db: Session, profile: SocialTenantProfile, *, commit: bool = True) -> SocialTenantProfile:
        profile.updated_at = utcnow()
        profile.posts_per_week = min(max(int(profile.posts_per_week or 3), 1), MAX_POSTS_PER_WEEK)
        profile.schedule_days = normalize_schedule_days(profile.schedule_days)

        if cls._has_standalone_tables(db):
            row = db.query(SocialTenantProfileRow).filter_by(tenant_id=profile.tenant_id).first()
            if row is None:
                row = SocialTenantProfileRow(
                    tenant_id=profile.tenant_id,
                    uploadpost_username=profile.uploadpost_username,
                )
                db.add(row)
            row.uploadpost_username = profile.uploadpost_username
            row.connected_platforms_json = json.dumps(profile.connected_platforms)
            row.facebook_page_id = profile.facebook_page_id
            row.facebook_page_name = profile.facebook_page_name
            row.instagram_account_id = profile.instagram_account_id
            row.instagram_handle = profile.instagram_handle
            row.youtube_channel_id = profile.youtube_channel_id
            row.youtube_channel_title = profile.youtube_channel_title
            row.reddit_city_subreddit = profile.reddit_city_subreddit
            row.reddit_secondary_subreddits_json = json.dumps(profile.reddit_secondary_subreddits)
            row.reddit_flair_id = profile.reddit_flair_id
            row.reddit_flair_text = profile.reddit_flair_text
            row.reddit_mode = profile.reddit_mode
            row.last_jwt_url = profile.last_jwt_url
            row.last_jwt_expires_at = profile.last_jwt_expires_at
            row.automation_enabled = profile.automation_enabled
            row.posts_per_week = profile.posts_per_week
            row.schedule_days_json = json.dumps(profile.schedule_days)
            row.preferred_time = profile.preferred_time
            row.timezone = profile.timezone
            row.auto_repurpose_blogs = profile.auto_repurpose_blogs
            row.min_citation_score = profile.min_citation_score
            if hasattr(row, "default_voice_id"):
                row.default_voice_id = profile.default_voice_id or "Kore"
            row.default_platforms_json = json.dumps(profile.default_platforms)
            if hasattr(row, "content_strategy_json"):
                row.content_strategy_json = json.dumps(profile.content_strategy or {})
            row.updated_at = profile.updated_at
            if commit:
                db.commit()
            else:
                db.flush()
            return profile

        BastionAuditEvent = cls._get_audit_event_model()

        event = (
            db.query(BastionAuditEvent)
            .filter(
                BastionAuditEvent.tenant_id == profile.tenant_id,
                BastionAuditEvent.event_type == EVENT_TYPE_PROFILE,
            )
            .order_by(BastionAuditEvent.created_at.desc())
            .first()
        )
        if event is None:
            event = BastionAuditEvent(
                id=str(uuid.uuid4()),
                tenant_id=profile.tenant_id,
                event_type=EVENT_TYPE_PROFILE,
                actor="system:social_studio",
                payload=profile.to_dict(),
                created_at=profile.created_at,
            )
            db.add(event)
        else:
            event.payload = profile.to_dict()
        if commit:
            db.commit()
        else:
            db.flush()
        return profile

    @classmethod
    def list_posts(cls, db: Session, tenant_id: str, limit: Optional[int] = 50) -> list[SocialPost]:
        if cls._has_standalone_tables(db):
            q = (
                db.query(SocialPostRow)
                .filter_by(tenant_id=tenant_id)
                .order_by(SocialPostRow.created_at.desc())
            )
            if limit is not None:
                q = q.limit(limit)
            return [cls._row_to_post(r) for r in q.all()]

        BastionAuditEvent = cls._get_audit_event_model()

        q = (
            db.query(BastionAuditEvent)
            .filter(
                BastionAuditEvent.tenant_id == tenant_id,
                BastionAuditEvent.event_type == EVENT_TYPE_POST,
            )
            .order_by(BastionAuditEvent.created_at.desc())
        )
        if limit is not None:
            q = q.limit(limit)
        events = q.all()
        posts: list[SocialPost] = []
        for ev in events:
            payload = dict(ev.payload or {})
            payload.setdefault("id", ev.id)
            payload.setdefault("tenant_id", tenant_id)
            posts.append(SocialPost.from_dict(payload))
        return posts

    @classmethod
    def get_post(cls, db: Session, tenant_id: str, post_id: str) -> Optional[SocialPost]:
        if cls._has_standalone_tables(db):
            row = db.query(SocialPostRow).filter_by(tenant_id=tenant_id, id=post_id).first()
            return cls._row_to_post(row) if row else None

        BastionAuditEvent = cls._get_audit_event_model()

        ev = (
            db.query(BastionAuditEvent)
            .filter(
                BastionAuditEvent.tenant_id == tenant_id,
                BastionAuditEvent.event_type == EVENT_TYPE_POST,
                BastionAuditEvent.id == post_id,
            )
            .first()
        )
        if ev is None:
            return None
        payload = dict(ev.payload or {})
        payload.setdefault("id", ev.id)
        payload.setdefault("tenant_id", tenant_id)
        return SocialPost.from_dict(payload)

    @classmethod
    def find_post_by_request_id(cls, db: Session, request_id: str) -> Optional[SocialPost]:
        if not request_id:
            return None
        if cls._has_standalone_tables(db):
            rows = (
                db.query(SocialPostRow)
                .filter(SocialPostRow.uploadpost_request_ids_json.contains(request_id))
                .all()
            )
            for r in rows:
                post = cls._row_to_post(r)
                if request_id in post.uploadpost_request_ids.values():
                    return post
            return None

        BastionAuditEvent = cls._get_audit_event_model()

        events = (
            db.query(BastionAuditEvent)
            .filter(BastionAuditEvent.event_type == EVENT_TYPE_POST)
            .order_by(BastionAuditEvent.created_at.desc())
            .all()
        )
        for ev in events:
            payload = dict(ev.payload or {})
            req_map = payload.get("uploadpost_request_ids") or {}
            if request_id in req_map.values():
                payload.setdefault("id", ev.id)
                payload.setdefault("tenant_id", ev.tenant_id)
                return SocialPost.from_dict(payload)
        return None

    @classmethod
    def save_post(cls, db: Session, post: SocialPost, *, commit: bool = True) -> SocialPost:
        post.updated_at = utcnow()
        if cls._has_standalone_tables(db):
            row = db.query(SocialPostRow).filter_by(id=post.id).first()
            if row is None:
                row = SocialPostRow(id=post.id, tenant_id=post.tenant_id, created_at=post.created_at)
                db.add(row)
            row.source_type = post.source_type
            row.source_ref_id = post.source_ref_id
            row.source_title = post.source_title
            row.canonical_source_url = post.canonical_source_url
            row.format_type = post.format_type
            row.topic = post.topic
            row.target_platforms_json = json.dumps(post.target_platforms)
            row.voiceover_script = post.voiceover_script
            row.transcript_srt = post.transcript_srt
            row.hook_headline = post.hook_headline
            row.cta_line = post.cta_line
            row.ig_reels_caption = post.ig_reels_caption
            row.fb_reels_caption = post.fb_reels_caption
            row.youtube_shorts_title = post.youtube_shorts_title
            row.youtube_shorts_description = post.youtube_shorts_description
            row.reddit_subreddit = post.reddit_subreddit
            row.reddit_title = post.reddit_title
            row.reddit_selftext = post.reddit_selftext
            row.video_gcs_object = post.video_gcs_object
            row.feed_image_gcs_object = post.feed_image_gcs_object
            row.media_delivery_url = post.media_delivery_url
            row.feed_image_delivery_url = post.feed_image_delivery_url
            row.media_metadata_json = json.dumps(post.media_metadata)
            row.qualification_status = post.qualification_status
            row.llm_citation_score = post.llm_citation_score
            row.qualification_report_json = json.dumps(post.qualification_report)
            row.status = post.status
            row.iso_year = post.iso_year
            row.iso_week = post.iso_week
            row.slot_index = post.slot_index
            row.scheduled_for = post.scheduled_for
            row.published_at = post.published_at
            row.uploadpost_request_ids_json = json.dumps(post.uploadpost_request_ids)
            row.platform_results_json = json.dumps(post.platform_results)
            row.error = post.error
            row.updated_at = post.updated_at
            if commit:
                db.commit()
            else:
                db.flush()
            return post

        BastionAuditEvent = cls._get_audit_event_model()

        ev = (
            db.query(BastionAuditEvent)
            .filter(
                BastionAuditEvent.tenant_id == post.tenant_id,
                BastionAuditEvent.event_type == EVENT_TYPE_POST,
                BastionAuditEvent.id == post.id,
            )
            .first()
        )
        if ev is None:
            ev = BastionAuditEvent(
                id=post.id,
                tenant_id=post.tenant_id,
                event_type=EVENT_TYPE_POST,
                actor="system:social_studio",
                payload=post.to_dict(),
                created_at=post.created_at,
            )
            db.add(ev)
        else:
            ev.payload = post.to_dict()
        if commit:
            db.commit()
        else:
            db.flush()
        return post

    @classmethod
    def get_weekly_slots_used(
        cls,
        db: Session,
        tenant_id: str,
        iso_year: int,
        iso_week: int,
    ) -> list[int]:
        """Return the occupied slot numbers (1..3) for the given tenant and ISO week."""
        if cls._has_standalone_tables(db):
            rows = (
                db.query(SocialPostRow)
                .filter(
                    SocialPostRow.tenant_id == tenant_id,
                    SocialPostRow.iso_year == iso_year,
                    SocialPostRow.iso_week == iso_week,
                    SocialPostRow.status != "cancelled",
                )
                .all()
            )
            used = {
                int(r.slot_index)
                for r in rows
                if r.slot_index in (1, 2, 3)
            }
            return sorted(used)

        posts = cls.list_posts(db, tenant_id, limit=None)
        used = {
            int(p.slot_index)
            for p in posts
            if p.iso_year == iso_year
            and p.iso_week == iso_week
            and p.slot_index in (1, 2, 3)
            and p.status != "cancelled"
        }
        return sorted(used)

    @classmethod
    def list_due_scheduled_posts(cls, db: Session, now: Optional[datetime] = None) -> list[SocialPost]:
        """Return all posts with status in ('scheduled', 'partial_failure') whose scheduled_for timestamp is <= now."""
        cutoff = now or utcnow()
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        if cls._has_standalone_tables(db):
            rows = (
                db.query(SocialPostRow)
                .filter(
                    SocialPostRow.status.in_(["scheduled", "partial_failure"]),
                    SocialPostRow.scheduled_for.isnot(None),
                    SocialPostRow.scheduled_for <= cutoff,
                )
                .order_by(SocialPostRow.scheduled_for.asc())
                .all()
            )
            return [cls._row_to_post(r) for r in rows]

        BastionAuditEvent = cls._get_audit_event_model()
        events = (
            db.query(BastionAuditEvent)
            .filter(BastionAuditEvent.event_type == EVENT_TYPE_POST)
            .order_by(BastionAuditEvent.created_at.asc())
            .all()
        )
        due: list[SocialPost] = []
        for ev in events:
            payload = dict(ev.payload or {})
            if payload.get("status") not in {"scheduled", "partial_failure"}:
                continue
            sched_dt = _parse_dt(payload.get("scheduled_for"))
            if sched_dt is not None and sched_dt <= cutoff:
                payload.setdefault("id", ev.id)
                payload.setdefault("tenant_id", ev.tenant_id)
                due.append(SocialPost.from_dict(payload))
        due.sort(key=lambda p: p.scheduled_for or cutoff)
        return due

    @classmethod
    def list_automation_profiles(cls, db: Session) -> list[SocialTenantProfile]:
        """Return all SocialTenantProfile records with automation_enabled=True."""
        if cls._has_standalone_tables(db):
            rows = (
                db.query(SocialTenantProfileRow)
                .filter(SocialTenantProfileRow.automation_enabled.is_(True))
                .all()
            )
            return [
                SocialTenantProfile(
                    tenant_id=row.tenant_id,
                    uploadpost_username=row.uploadpost_username,
                    connected_platforms=json.loads(row.connected_platforms_json or "[]"),
                    facebook_page_id=row.facebook_page_id,
                    facebook_page_name=row.facebook_page_name,
                    instagram_account_id=row.instagram_account_id,
                    instagram_handle=row.instagram_handle,
                    youtube_channel_id=row.youtube_channel_id,
                    youtube_channel_title=row.youtube_channel_title,
                    reddit_city_subreddit=row.reddit_city_subreddit or "",
                    reddit_secondary_subreddits=json.loads(row.reddit_secondary_subreddits_json or "[]"),
                    reddit_flair_id=row.reddit_flair_id,
                    reddit_flair_text=row.reddit_flair_text,
                    reddit_mode=row.reddit_mode or "official_api",
                    last_jwt_url=row.last_jwt_url,
                    last_jwt_expires_at=_parse_dt(row.last_jwt_expires_at),
                    automation_enabled=bool(row.automation_enabled),
                    posts_per_week=min(max(int(row.posts_per_week or 3), 1), MAX_POSTS_PER_WEEK),
                    schedule_days=json.loads(row.schedule_days_json or '["mon", "wed", "fri"]'),
                    preferred_time=row.preferred_time or "09:30",
                    timezone=row.timezone or "America/New_York",
                    auto_repurpose_blogs=bool(row.auto_repurpose_blogs),
                    min_citation_score=int(row.min_citation_score or 80),
                    default_voice_id=getattr(row, "default_voice_id", None) or "Kore",
                    default_platforms=json.loads(row.default_platforms_json or "[]") or list(DEFAULT_TARGET_PLATFORMS),
                    created_at=_parse_dt(row.created_at) or utcnow(),
                    updated_at=_parse_dt(row.updated_at) or utcnow(),
                )
                for row in rows
            ]

        BastionAuditEvent = cls._get_audit_event_model()
        events = (
            db.query(BastionAuditEvent)
            .filter(BastionAuditEvent.event_type == EVENT_TYPE_PROFILE)
            .order_by(BastionAuditEvent.created_at.desc())
            .all()
        )
        seen: set[str] = set()
        profiles: list[SocialTenantProfile] = []
        for ev in events:
            tid = str(ev.tenant_id)
            if tid in seen:
                continue
            seen.add(tid)
            payload = dict(ev.payload or {})
            payload.setdefault("tenant_id", tid)
            prof = SocialTenantProfile.from_dict(payload)
            if prof.automation_enabled:
                profiles.append(prof)
        return profiles

    @classmethod
    def claim_next_weekly_slot(
        cls,
        db: Session,
        tenant_id: str,
        iso_year: int,
        iso_week: int,
        max_slots: int = MAX_POSTS_PER_WEEK,
    ) -> Optional[int]:
        """Return the next available weekly slot index (1..max_slots), or None if all 3 are full."""
        cap = min(max(int(max_slots or 3), 1), MAX_POSTS_PER_WEEK)
        used = set(cls.get_weekly_slots_used(db, tenant_id, iso_year, iso_week))
        for idx in range(1, cap + 1):
            if idx not in used:
                return idx
        return None

    @staticmethod
    def _row_to_post(row: SocialPostRow) -> SocialPost:
        return SocialPost(
            id=row.id,
            tenant_id=row.tenant_id,
            source_type=row.source_type,
            source_ref_id=row.source_ref_id,
            source_title=row.source_title or "",
            canonical_source_url=row.canonical_source_url or "",
            format_type=row.format_type,
            topic=row.topic or "",
            target_platforms=json.loads(row.target_platforms_json or "[]"),
            voiceover_script=row.voiceover_script or "",
            transcript_srt=row.transcript_srt or "",
            hook_headline=row.hook_headline or "",
            cta_line=row.cta_line or "",
            ig_reels_caption=row.ig_reels_caption or "",
            fb_reels_caption=row.fb_reels_caption or "",
            youtube_shorts_title=row.youtube_shorts_title or "",
            youtube_shorts_description=row.youtube_shorts_description or "",
            reddit_subreddit=row.reddit_subreddit or "",
            reddit_title=row.reddit_title or "",
            reddit_selftext=row.reddit_selftext or "",
            video_gcs_object=row.video_gcs_object,
            feed_image_gcs_object=row.feed_image_gcs_object,
            media_delivery_url=row.media_delivery_url,
            feed_image_delivery_url=row.feed_image_delivery_url,
            media_metadata=json.loads(row.media_metadata_json or "{}"),
            qualification_status=row.qualification_status or "pending",
            llm_citation_score=int(row.llm_citation_score or 0),
            qualification_report=json.loads(row.qualification_report_json or "{}"),
            status=row.status or "generating",
            iso_year=row.iso_year,
            iso_week=row.iso_week,
            slot_index=row.slot_index,
            scheduled_for=_parse_dt(row.scheduled_for),
            published_at=_parse_dt(row.published_at),
            uploadpost_request_ids=json.loads(row.uploadpost_request_ids_json or "{}"),
            platform_results=json.loads(row.platform_results_json or "{}"),
            error=row.error,
            created_at=_parse_dt(row.created_at) or utcnow(),
            updated_at=_parse_dt(row.updated_at) or utcnow(),
        )
