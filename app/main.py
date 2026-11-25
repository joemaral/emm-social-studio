"""
EMM Social Studio Microservice (`emm-social-studio`) FastAPI Application & Service Orchestrator.

Provides:
1. Standalone FastAPI microservice endpoints (`/healthz`, `/readyz`, `/v1/tenants/...`, `/v1/webhooks/upload-post`)
   protected by HMAC-SHA256 (`X-Social-Studio-Signature`).
2. Direct Python orchestration functions (`SocialStudioService`) so the main Client Marketing Suite
   can either proxy over HTTP to `SOCIAL_STUDIO_SERVICE_URL` or invoke the service engine in-process.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .creator_engine import (
    compute_weekly_slot_datetime,
    dispatch_qualified_post_to_platforms,
    generate_ai_content_strategy,
    generate_educational_post_package,
    reconcile_post_status_from_results,
    resolve_slot_weekday,
    select_next_qualifying_source,
)
from .media_normalizer import (
    DEFAULT_GEMINI_TTS_VOICE,
    GEMINI_TTS_VOICES,
    generate_clinical_placeholder_scene_jpeg,
    generate_timed_srt_transcript,
    get_voice_metadata,
    normalize_feed_image,
    normalize_voice_id,
    render_and_normalize_reel_video,
    synthesize_gemini_tts_wav,
)
from .models import (
    DEFAULT_SCHEDULE_DAYS,
    DEFAULT_TARGET_PLATFORMS,
    MAX_POSTS_PER_WEEK,
    SocialPost,
    SocialStudioBase,
    SocialStudioStore,
    SocialTenantProfile,
    derive_default_city_subreddit,
    derive_default_content_strategy,
    normalize_schedule_days,
    utcnow,
)
from .qualification_gate import (
    auto_heal_post_package,
    qualify_candidate_topic,
    qualify_generated_post,
)
from .reddit_client import get_reddit_api_status, normalize_subreddit_name
from .uploadpost_client import (
    ensure_user_profile,
    fetch_user_profile,
    generate_white_label_connect_jwt,
    is_uploadpost_configured,
)

logger = logging.getLogger(__name__)

# In-memory LRU media cache for immediate preview & publishing within the container lifecycle
_MEDIA_VIDEO_CACHE: dict[str, bytes] = {}
_MEDIA_IMAGE_CACHE: dict[str, bytes] = {}


def _cache_media(post_id: str, video_bytes: Optional[bytes] = None, image_bytes: Optional[bytes] = None) -> None:
    if video_bytes:
        if len(_MEDIA_VIDEO_CACHE) > 32:
            oldest = next(iter(_MEDIA_VIDEO_CACHE))
            _MEDIA_VIDEO_CACHE.pop(oldest, None)
        _MEDIA_VIDEO_CACHE[post_id] = video_bytes
    if image_bytes:
        if len(_MEDIA_IMAGE_CACHE) > 32:
            oldest = next(iter(_MEDIA_IMAGE_CACHE))
            _MEDIA_IMAGE_CACHE.pop(oldest, None)
        _MEDIA_IMAGE_CACHE[post_id] = image_bytes


def get_or_render_post_video_bytes(post: SocialPost) -> bytes:
    """Return cached normalized MP4 bytes for a post, or deterministically re-render on demand."""
    cached = _MEDIA_VIDEO_CACHE.get(post.id)
    if cached:
        return cached
    scenes = [
        generate_clinical_placeholder_scene_jpeg(post.hook_headline, post.cta_line, theme_index=0),
        generate_clinical_placeholder_scene_jpeg(post.hook_headline, post.cta_line, theme_index=1),
        generate_clinical_placeholder_scene_jpeg(post.hook_headline, post.cta_line, theme_index=2),
    ]
    bundle = render_and_normalize_reel_video(
        scene_images=scenes,
        voiceover_script=post.voiceover_script,
        hook_headline=post.hook_headline,
        cta_line=post.cta_line,
        voice_id=(post.media_metadata or {}).get("voice_id") or DEFAULT_GEMINI_TTS_VOICE,
    )
    video_bytes = bundle["video_bytes"]
    _cache_media(post.id, video_bytes=video_bytes)
    return video_bytes


def get_or_render_post_image_bytes(post: SocialPost) -> bytes:
    """Return cached 1080x1350 (4:5) JPEG cover image for a post, or render on demand."""
    cached = _MEDIA_IMAGE_CACHE.get(post.id)
    if cached:
        return cached
    raw_scene = generate_clinical_placeholder_scene_jpeg(post.hook_headline, post.cta_line, theme_index=0)
    norm_res = normalize_feed_image(raw_scene)
    norm_img = norm_res["bytes"]
    _cache_media(post.id, image_bytes=norm_img)
    return norm_img


def _clean_opt_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    cleaned = str(val).strip()
    return cleaned or None


def _merge_uploadpost_profile_info(
    profile: SocialTenantProfile,
    info: dict[str, Any],
    *,
    overwrite_existing: bool = False,
) -> bool:
    """Merge Upload-Post connected accounts, handles, channels, and Facebook Pages onto a tenant profile."""
    if not isinstance(info, dict) or not info.get("configured"):
        return False
    # If profile lookup explicitly reported profile does not exist yet, don't clobber local fields
    if info.get("found") is False:
        return False

    changed = False
    resolved_user = _clean_opt_str(info.get("username"))
    if resolved_user and resolved_user != profile.uploadpost_username:
        profile.uploadpost_username = resolved_user
        changed = True

    remote_platforms = [
        str(p).lower().strip()
        for p in (info.get("connected_platforms") or [])
        if str(p).strip()
    ]
    remote_set = set(remote_platforms)

    # Keep "reddit" if Reddit API is configured or already listed, while syncing IG/FB/YT from Upload-Post
    preserved_non_up = [p for p in profile.connected_platforms if p not in {"instagram", "facebook", "youtube"}]
    if get_reddit_api_status().get("configured") and "reddit" not in preserved_non_up:
        preserved_non_up.append("reddit")
    merged_platforms = sorted(set(remote_platforms + preserved_non_up))
    if merged_platforms != sorted(set(profile.connected_platforms)):
        profile.connected_platforms = merged_platforms
        changed = True

    pages = [
        p for p in (info.get("available_facebook_pages") or [])
        if isinstance(p, dict) and p.get("page_id")
    ]
    if pages != profile.available_facebook_pages:
        profile.available_facebook_pages = pages
        changed = True

    # Clear stale IDs/handles when Upload-Post explicitly reports a platform disconnected
    if "instagram" not in remote_set:
        if profile.instagram_account_id is not None or profile.instagram_handle is not None:
            profile.instagram_account_id = None
            profile.instagram_handle = None
            changed = True
    if "youtube" not in remote_set:
        if profile.youtube_channel_id is not None or profile.youtube_channel_title is not None:
            profile.youtube_channel_id = None
            profile.youtube_channel_title = None
            changed = True
    if "facebook" not in remote_set and not pages:
        if profile.facebook_page_id is not None or profile.facebook_page_name is not None:
            profile.facebook_page_id = None
            profile.facebook_page_name = None
            changed = True

    # Auto-populate Facebook Page ID & Name if unset or matching page name updated
    if pages:
        matching_page = next((p for p in pages if p["page_id"] == profile.facebook_page_id), None)
        if matching_page:
            if matching_page.get("page_name") and matching_page["page_name"] != profile.facebook_page_name:
                if overwrite_existing or not profile.facebook_page_name:
                    profile.facebook_page_name = matching_page["page_name"]
                    changed = True
        elif not profile.facebook_page_id:
            profile.facebook_page_id = pages[0]["page_id"]
            profile.facebook_page_name = pages[0].get("page_name") or pages[0]["page_id"]
            changed = True

    attr_platform = {
        "facebook_page_id": "facebook",
        "facebook_page_name": "facebook",
        "instagram_account_id": "instagram",
        "instagram_handle": "instagram",
        "youtube_channel_id": "youtube",
        "youtube_channel_title": "youtube",
    }
    for attr, plat in attr_platform.items():
        if plat not in remote_set and not (plat == "facebook" and pages):
            continue
        remote_val = _clean_opt_str(info.get(attr))
        if not remote_val:
            continue
        current_val = getattr(profile, attr)
        if attr in ("facebook_page_id", "facebook_page_name") and profile.facebook_page_id and pages:
            # Preserve explicit user-selected Facebook Page from pages list
            continue
        if overwrite_existing:
            if remote_val != current_val:
                setattr(profile, attr, remote_val)
                changed = True
        else:
            # In passive overview sync, only fill blank fields — never clobber manual edits
            if not current_val:
                setattr(profile, attr, remote_val)
                changed = True

    return changed


class SocialStudioService:
    """Core business logic shared between standalone HTTP microservice and BFF embedded mode."""

    @staticmethod
    def get_overview(
        db: Session,
        tenant_id: str,
        client_context: dict[str, Any],
    ) -> dict[str, Any]:
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=client_context.get("slug"),
            city=client_context.get("city"),
            industry=client_context.get("industry"),
        )
        dirty = False
        # Ensure city subreddit has a sensible default if empty
        if not profile.reddit_city_subreddit:
            profile.reddit_city_subreddit = derive_default_city_subreddit(
                client_context.get("city"),
                client_context.get("industry"),
            )
            dirty = True

        sync_error: Optional[str] = None
        if is_uploadpost_configured():
            try:
                info = fetch_user_profile(profile.uploadpost_username)
                if _merge_uploadpost_profile_info(profile, info, overwrite_existing=False):
                    dirty = True
            except Exception as exc:
                sync_error = str(exc)
                logger.warning("Upload-Post overview sync skipped for %s: %s", tenant_id, exc)

        if dirty:
            SocialStudioStore.save_profile(db, profile)

        posts = SocialStudioStore.list_posts(db, tenant_id, limit=50)
        now = utcnow()
        iso_year, iso_week, _ = now.isocalendar()
        slots_used = SocialStudioStore.get_weekly_slots_used(db, tenant_id, iso_year, iso_week)

        # Build the 3 weekly cadence slots derived from profile.schedule_days
        weekly_slots: list[dict[str, Any]] = []
        for idx in range(1, MAX_POSTS_PER_WEEK + 1):
            slot_post = next(
                (
                    p
                    for p in posts
                    if p.iso_year == iso_year
                    and p.iso_week == iso_week
                    and p.slot_index == idx
                    and p.status != "cancelled"
                ),
                None,
            )
            _, day_label = resolve_slot_weekday(idx, profile.schedule_days)
            slot_dt = compute_weekly_slot_datetime(
                iso_year,
                iso_week,
                idx,
                preferred_time=profile.preferred_time,
                tz_name=profile.timezone,
                schedule_days=profile.schedule_days,
            )
            display_sched = (
                slot_post.scheduled_for.isoformat()
                if slot_post and slot_post.scheduled_for
                else slot_dt.isoformat()
            )
            weekly_slots.append(
                {
                    "slot_index": idx,
                    "day_label": day_label,
                    "iso_year": iso_year,
                    "iso_week": iso_week,
                    "scheduled_for": display_sched,
                    "occupied": slot_post is not None,
                    "enabled_within_cap": idx <= profile.posts_per_week,
                    "post": slot_post.to_dict() if slot_post else None,
                }
            )

        reddit_status = get_reddit_api_status()
        recent_topics = [p.topic for p in posts if p.status != "cancelled" and p.topic][:30]
        next_recommended_source = select_next_qualifying_source(client_context, recent_topics)
        strategy_resolved = dict(profile.content_strategy or {}) or derive_default_content_strategy(client_context)

        return {
            "tenant_id": tenant_id,
            "profile": profile.to_dict(),
            "content_strategy": strategy_resolved,
            "cadence": {
                "iso_year": iso_year,
                "iso_week": iso_week,
                "posts_per_week_cap": profile.posts_per_week,
                "max_posts_per_week": MAX_POSTS_PER_WEEK,
                "slots_used": slots_used,
                "slots_remaining": max(0, profile.posts_per_week - len(slots_used)),
                "weekly_slots": weekly_slots,
            },
            "integrations": {
                "uploadpost_configured": is_uploadpost_configured(),
                "sync_error": sync_error,
                "reddit_api": reddit_status,
            },
            "available_voices": GEMINI_TTS_VOICES,
            "next_recommended_source": next_recommended_source,
            "available_sources": {
                "blogs": list(client_context.get("blogs") or []),
                "service_pages": list(client_context.get("service_pages") or []),
                "gbp_services": list(client_context.get("gbp_services") or []),
            },
            "posts": [p.to_dict() for p in posts],
        }

    @staticmethod
    def update_profile(
        db: Session,
        tenant_id: str,
        updates: dict[str, Any],
        client_context: Optional[dict[str, Any]] = None,
    ) -> SocialTenantProfile:
        ctx = client_context or {}
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=ctx.get("slug"),
            city=ctx.get("city"),
            industry=ctx.get("industry"),
        )
        if "connected_platforms" in updates and isinstance(updates["connected_platforms"], list):
            profile.connected_platforms = [str(p).lower().strip() for p in updates["connected_platforms"] if p]
        if "facebook_page_id" in updates:
            profile.facebook_page_id = _clean_opt_str(updates["facebook_page_id"])
            if profile.facebook_page_id and profile.available_facebook_pages:
                matched = next(
                    (p for p in profile.available_facebook_pages if p.get("page_id") == profile.facebook_page_id),
                    None,
                )
                if matched and matched.get("page_name"):
                    profile.facebook_page_name = matched["page_name"]
        if "facebook_page_name" in updates:
            profile.facebook_page_name = _clean_opt_str(updates["facebook_page_name"])
        if "instagram_account_id" in updates:
            profile.instagram_account_id = _clean_opt_str(updates["instagram_account_id"])
        if "instagram_handle" in updates:
            profile.instagram_handle = _clean_opt_str(updates["instagram_handle"])
        if "youtube_channel_id" in updates:
            profile.youtube_channel_id = _clean_opt_str(updates["youtube_channel_id"])
        if "youtube_channel_title" in updates:
            profile.youtube_channel_title = _clean_opt_str(updates["youtube_channel_title"])
        if "reddit_city_subreddit" in updates:
            profile.reddit_city_subreddit = normalize_subreddit_name(str(updates["reddit_city_subreddit"] or ""))
        if "reddit_secondary_subreddits" in updates and isinstance(updates["reddit_secondary_subreddits"], list):
            profile.reddit_secondary_subreddits = [
                normalize_subreddit_name(str(s)) for s in updates["reddit_secondary_subreddits"] if str(s).strip()
            ]
        if "reddit_flair_id" in updates:
            profile.reddit_flair_id = _clean_opt_str(updates["reddit_flair_id"])
        if "reddit_flair_text" in updates:
            profile.reddit_flair_text = _clean_opt_str(updates["reddit_flair_text"])
        if "automation_enabled" in updates:
            profile.automation_enabled = bool(updates["automation_enabled"])
        if "posts_per_week" in updates and updates["posts_per_week"] is not None:
            profile.posts_per_week = min(max(int(updates["posts_per_week"]), 1), MAX_POSTS_PER_WEEK)
        if "schedule_days" in updates and isinstance(updates["schedule_days"], list):
            profile.schedule_days = normalize_schedule_days(updates["schedule_days"])
        if "preferred_time" in updates and updates["preferred_time"]:
            profile.preferred_time = str(updates["preferred_time"]).strip()
        if "timezone" in updates and updates["timezone"]:
            tz_cand = str(updates["timezone"]).strip()
            try:
                from zoneinfo import ZoneInfo
                ZoneInfo(tz_cand)
                profile.timezone = tz_cand
            except Exception:
                pass
        if "auto_repurpose_blogs" in updates:
            profile.auto_repurpose_blogs = bool(updates["auto_repurpose_blogs"])
        if "min_citation_score" in updates and updates["min_citation_score"] is not None:
            profile.min_citation_score = min(max(int(updates["min_citation_score"]), 50), 100)
        if "default_voice_id" in updates and updates["default_voice_id"]:
            profile.default_voice_id = normalize_voice_id(str(updates["default_voice_id"]))
        if "default_platforms" in updates and isinstance(updates["default_platforms"], list):
            valid_plats = {"instagram", "facebook", "youtube", "reddit"}
            cleaned_plats = [
                str(p).lower().strip()
                for p in updates["default_platforms"]
                if str(p).lower().strip() in valid_plats
            ]
            profile.default_platforms = list(dict.fromkeys(cleaned_plats)) or list(DEFAULT_TARGET_PLATFORMS)
        if "content_strategy" in updates and isinstance(updates["content_strategy"], dict):
            profile.content_strategy = dict(updates["content_strategy"])

        return SocialStudioStore.save_profile(db, profile)

    @staticmethod
    def get_content_strategy(
        db: Session,
        tenant_id: str,
        client_context: dict[str, Any],
    ) -> dict[str, Any]:
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=client_context.get("slug"),
            city=client_context.get("city"),
            industry=client_context.get("industry"),
        )
        if profile.content_strategy:
            return dict(profile.content_strategy)
        return derive_default_content_strategy(client_context)

    @staticmethod
    def generate_content_strategy(
        client_context: dict[str, Any],
    ) -> dict[str, Any]:
        return generate_ai_content_strategy(client_context)

    @staticmethod
    def update_content_strategy(
        db: Session,
        tenant_id: str,
        strategy_updates: dict[str, Any],
        client_context: dict[str, Any],
    ) -> dict[str, Any]:
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=client_context.get("slug"),
            city=client_context.get("city"),
            industry=client_context.get("industry"),
        )
        current = dict(profile.content_strategy or {})
        if not current:
            current = derive_default_content_strategy(client_context)
        for k, v in strategy_updates.items():
            if v is not None:
                current[k] = v
        profile.content_strategy = current
        SocialStudioStore.save_profile(db, profile)
        return current

    @staticmethod
    def sync_connected_accounts(
        db: Session,
        tenant_id: str,
        client_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Pull connected social accounts, handles, YouTube channel, and Facebook Page IDs from Upload-Post."""
        ctx = client_context or {}
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=ctx.get("slug"),
            city=ctx.get("city"),
            industry=ctx.get("industry"),
        )
        configured = is_uploadpost_configured()
        info: dict[str, Any] = {}
        if configured:
            info = fetch_user_profile(profile.uploadpost_username)
            if not info.get("found"):
                info = ensure_user_profile(profile.uploadpost_username)
            _merge_uploadpost_profile_info(profile, info, overwrite_existing=True)
            SocialStudioStore.save_profile(db, profile)

        return {
            "profile": profile.to_dict(),
            "sync": {
                "synced": configured,
                "uploadpost_configured": configured,
                "connected_platforms": list(profile.connected_platforms),
                "available_facebook_pages": list(profile.available_facebook_pages),
                "note": info.get("note")
                or (
                    "Synced connected accounts from Upload-Post."
                    if configured
                    else "Upload-Post API key is not configured in this environment."
                ),
            },
        }

    @staticmethod
    def generate_connect_jwt(
        db: Session,
        tenant_id: str,
        client_context: dict[str, Any],
        *,
        redirect_url: Optional[str] = None,
        platforms: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=client_context.get("slug"),
            city=client_context.get("city"),
            industry=client_context.get("industry"),
        )
        business_name = (
            client_context.get("practice_name")
            or client_context.get("business_name")
            or client_context.get("name")
            or "Medical Practice"
        )
        jwt_res = generate_white_label_connect_jwt(
            username=profile.uploadpost_username,
            redirect_url=redirect_url or f"https://elevatedmarketingexperts.com/agency/clients/{tenant_id}/social-media?social_connected=1",
            connect_title=f"Connect {business_name} Social Accounts",
            connect_description="Link Instagram Reels, Facebook Reels, and YouTube Shorts for automated social distribution.",
            platforms=platforms or ["instagram", "facebook", "youtube"],
        )
        if _clean_opt_str(jwt_res.get("username")):
            profile.uploadpost_username = str(jwt_res["username"])
        if isinstance(jwt_res.get("profile_info"), dict):
            _merge_uploadpost_profile_info(profile, jwt_res["profile_info"], overwrite_existing=False)
        profile.last_jwt_url = jwt_res["access_url"]
        profile.last_jwt_expires_at = datetime.fromisoformat(jwt_res["expires_at"])
        SocialStudioStore.save_profile(db, profile)
        safe_jwt_res = {k: v for k, v in jwt_res.items() if k != "profile_info"}
        return {
            "tenant_id": tenant_id,
            "uploadpost_username": profile.uploadpost_username,
            **safe_jwt_res,
        }

    @staticmethod
    def check_topic_qualification(
        db: Session,
        tenant_id: str,
        client_context: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=client_context.get("slug"),
            city=client_context.get("city"),
            industry=client_context.get("industry"),
        )
        posts = SocialStudioStore.list_posts(db, tenant_id, limit=50)
        now = utcnow()
        iso_year, iso_week, _ = now.isocalendar()
        slots_used = SocialStudioStore.get_weekly_slots_used(db, tenant_id, iso_year, iso_week)
        recent_topics = [p.topic for p in posts if p.status != "cancelled" and p.topic][:30]

        topic = str(payload.get("topic") or "").strip()
        source_type = str(payload.get("source_type") or "blog_post").strip()
        canonical_url = str(
            payload.get("canonical_source_url")
            or client_context.get("website_url")
            or ""
        ).strip()
        requested_platforms = payload.get("target_platforms") or profile.default_platforms

        return qualify_candidate_topic(
            topic=topic,
            source_type=source_type,
            canonical_source_url=canonical_url,
            client_context=client_context,
            weekly_slots_used=slots_used,
            max_posts_per_week=profile.posts_per_week,
            recent_topics=recent_topics,
            requested_platforms=requested_platforms,
        )

    @staticmethod
    def generate_post(
        db: Session,
        tenant_id: str,
        client_context: dict[str, Any],
        payload: dict[str, Any],
        *,
        commit: bool = True,
    ) -> dict[str, Any]:
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=client_context.get("slug"),
            city=client_context.get("city"),
            industry=client_context.get("industry"),
            commit=commit,
        )
        posts = SocialStudioStore.list_posts(db, tenant_id, limit=50)
        now = utcnow()
        iso_year = int(payload.get("iso_year") or now.isocalendar()[0])
        iso_week = int(payload.get("iso_week") or now.isocalendar()[1])
        slots_used = SocialStudioStore.get_weekly_slots_used(db, tenant_id, iso_year, iso_week)
        recent_topics = [p.topic for p in posts if p.status != "cancelled" and p.topic][:30]

        if len(slots_used) >= profile.posts_per_week:
            raise ValueError(
                f"Weekly cadence cap reached ({len(slots_used)}/{profile.posts_per_week} slots filled for ISO week {iso_year}-W{iso_week:02d})."
            )

        slot_index = payload.get("slot_index")
        if slot_index is None:
            slot_index = SocialStudioStore.claim_next_weekly_slot(
                db, tenant_id, iso_year, iso_week, max_slots=profile.posts_per_week
            )
        else:
            slot_index = int(slot_index)
            if slot_index < 1 or slot_index > MAX_POSTS_PER_WEEK:
                raise ValueError(f"slot_index must be between 1 and {MAX_POSTS_PER_WEEK}.")
            if slot_index > profile.posts_per_week:
                raise ValueError(
                    f"Slot {slot_index} exceeds weekly cadence cap ({profile.posts_per_week} posts/week)."
                )
            if slot_index in slots_used:
                raise ValueError(
                    f"Slot {slot_index} is already occupied for ISO week {iso_year}-W{iso_week:02d}."
                )

        if slot_index is None:
            raise ValueError(
                f"Weekly cadence cap reached ({len(slots_used)}/{profile.posts_per_week} slots filled for ISO week {iso_year}-W{iso_week:02d})."
            )

        scheduled_for = compute_weekly_slot_datetime(
            iso_year,
            iso_week,
            int(slot_index),
            preferred_time=profile.preferred_time,
            tz_name=profile.timezone,
            schedule_days=profile.schedule_days,
        )

        source_override: Optional[dict[str, Any]] = None
        if payload.get("topic") or payload.get("source_ref_id"):
            source_override = {
                "source_type": payload.get("source_type") or "blog_post",
                "source_ref_id": payload.get("source_ref_id"),
                "source_title": payload.get("source_title") or payload.get("topic"),
                "topic": payload.get("topic") or payload.get("source_title"),
                "canonical_source_url": payload.get("canonical_source_url") or client_context.get("website_url") or "",
                "summary": payload.get("summary") or "",
            }

        selected_voice_id = normalize_voice_id(
            payload.get("voice_id") or getattr(profile, "default_voice_id", None) or DEFAULT_GEMINI_TTS_VOICE
        )
        post, report, video_bytes = generate_educational_post_package(
            tenant_id=tenant_id,
            profile=profile,
            client_context=client_context,
            source_override=source_override,
            target_platforms=payload.get("target_platforms") or profile.default_platforms,
            recent_topics=recent_topics,
            weekly_slots_used=slots_used,
            scheduled_for=scheduled_for,
            iso_year=iso_year,
            iso_week=iso_week,
            slot_index=int(slot_index),
            voice_id=selected_voice_id,
        )
        _cache_media(post.id, video_bytes=video_bytes)

        if payload.get("auto_dispatch", False) and post.qualification_status in {"qualified", "healed"}:
            post = dispatch_qualified_post_to_platforms(post, profile, video_bytes, publish_now=False)

        SocialStudioStore.save_post(db, post, commit=commit)
        return {
            "post": post.to_dict(),
            "qualification_report": report,
        }

    @staticmethod
    def update_post_copy(
        db: Session,
        tenant_id: str,
        post_id: str,
        client_context: dict[str, Any],
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=client_context.get("slug"),
            city=client_context.get("city"),
            industry=client_context.get("industry"),
        )
        post = SocialStudioStore.get_post(db, tenant_id, post_id)
        if post is None:
            raise LookupError(f"Social post '{post_id}' not found.")
        if post.status in {"published", "cancelled", "publishing"}:
            raise ValueError(f"Cannot edit copy for a post with status '{post.status}'.")

        editable_str_fields = (
            "topic",
            "hook_headline",
            "cta_line",
            "canonical_source_url",
            "voiceover_script",
            "ig_reels_caption",
            "fb_reels_caption",
            "youtube_shorts_title",
            "youtube_shorts_description",
            "reddit_subreddit",
            "reddit_title",
            "reddit_selftext",
        )
        media_dirty = False
        for field_name in editable_str_fields:
            if field_name in updates and updates[field_name] is not None:
                new_val = str(updates[field_name]).strip()
                if field_name == "reddit_subreddit":
                    new_val = normalize_subreddit_name(new_val)
                if new_val != getattr(post, field_name):
                    setattr(post, field_name, new_val)
                    if field_name in {"voiceover_script", "hook_headline", "cta_line"}:
                        media_dirty = True

        if "voice_id" in updates and updates["voice_id"]:
            new_voice_id = normalize_voice_id(str(updates["voice_id"]))
            voice_meta = get_voice_metadata(new_voice_id)
            meta = dict(post.media_metadata or {})
            if meta.get("voice_id") != new_voice_id:
                meta["voice_id"] = new_voice_id
                meta["voice_label"] = voice_meta["label"]
                post.media_metadata = meta
                media_dirty = True

        if media_dirty:
            if post.voiceover_script.strip():
                srt_text, clamped_dur = generate_timed_srt_transcript(post.voiceover_script)
                post.transcript_srt = srt_text
                meta = dict(post.media_metadata or {})
                meta["duration_seconds"] = clamped_dur
                post.media_metadata = meta
            _MEDIA_VIDEO_CACHE.pop(post.id, None)
            _MEDIA_IMAGE_CACHE.pop(post.id, None)

        posts = SocialStudioStore.list_posts(db, tenant_id, limit=50)
        recent_topics = [
            p.topic for p in posts
            if p.id != post.id and p.status != "cancelled" and p.topic
        ][:30]

        tier2 = qualify_generated_post(
            post,
            client_context,
            recent_topics=recent_topics,
            min_citation_score=profile.min_citation_score,
        )
        post.llm_citation_score = int(tier2["llm_citation_score"])
        report = dict(post.qualification_report or {})
        report["tier2"] = tier2
        post.qualification_report = report

        if tier2["qualified"]:
            post.qualification_status = "qualified"
            if post.status == "quarantined":
                post.status = "qualified_ready"
                post.error = None
        else:
            post.qualification_status = "disqualified"
            if post.status in {"qualified_ready", "generating", "scheduled", "partial_failure", "failed"}:
                post.status = "quarantined"
                post.error = "Edited post failed 6-point submission qualification checks."

        SocialStudioStore.save_post(db, post)
        return {
            "post": post.to_dict(),
            "qualification_report": post.qualification_report,
        }

    @staticmethod
    def publish_post(
        db: Session,
        tenant_id: str,
        post_id: str,
        client_context: Optional[dict[str, Any]] = None,
        *,
        publish_now: bool = False,
        only_platforms: Optional[list[str]] = None,
    ) -> SocialPost:
        ctx = client_context or {}
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=ctx.get("slug"),
            city=ctx.get("city"),
            industry=ctx.get("industry"),
        )
        post = SocialStudioStore.get_post(db, tenant_id, post_id)
        if post is None:
            raise LookupError(f"Social post '{post_id}' not found.")

        if not publish_now and post.scheduled_for is None and post.iso_year and post.iso_week and post.slot_index:
            post.scheduled_for = compute_weekly_slot_datetime(
                post.iso_year,
                post.iso_week,
                post.slot_index,
                preferred_time=profile.preferred_time,
                tz_name=profile.timezone,
                schedule_days=profile.schedule_days,
            )

        video_bytes = get_or_render_post_video_bytes(post)
        post = dispatch_qualified_post_to_platforms(
            post,
            profile,
            video_bytes,
            only_platforms=only_platforms,
            publish_now=publish_now,
        )
        return SocialStudioStore.save_post(db, post)

    @staticmethod
    def cancel_post(db: Session, tenant_id: str, post_id: str) -> SocialPost:
        post = SocialStudioStore.get_post(db, tenant_id, post_id)
        if post is None:
            raise LookupError(f"Social post '{post_id}' not found.")
        post.status = "cancelled"
        return SocialStudioStore.save_post(db, post)

    @staticmethod
    def preview_voiceover(
        db: Session,
        tenant_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Synthesize spoken voiceover audio for a script without mutating existing posts unless explicitly requested on a draft."""
        voice_id = normalize_voice_id(payload.get("voice_id") or DEFAULT_GEMINI_TTS_VOICE)
        post_id = payload.get("post_id")
        script_text = str(payload.get("text") or payload.get("voiceover_script") or "").strip()
        apply_to_post = bool(payload.get("apply_to_post", False))

        if post_id:
            post = SocialStudioStore.get_post(db, tenant_id, str(post_id))
            if post is not None:
                if not script_text:
                    script_text = post.voiceover_script
                if apply_to_post and post.status in {"generating", "qualified_ready", "quarantined"}:
                    meta = dict(post.media_metadata or {})
                    voice_meta = get_voice_metadata(voice_id)
                    meta["voice_id"] = voice_id
                    meta["voice_label"] = voice_meta["label"]
                    post.media_metadata = meta
                    _MEDIA_VIDEO_CACHE.pop(post.id, None)
                    SocialStudioStore.save_post(db, post)

        tts_res = synthesize_gemini_tts_wav(
            script_text or "Here is what patients should know before scheduling a consultation.",
            voice_id=voice_id,
        )
        return {
            "voice_id": tts_res["voice_id"],
            "voice_label": tts_res["voice_label"],
            "gender": tts_res["gender"],
            "tone": tts_res["tone"],
            "provider": tts_res["provider"],
            "duration_seconds": tts_res["duration_seconds"],
            "audio_data_url": tts_res["audio_data_url"],
        }

    @staticmethod
    def run_weekly_batch(
        db: Session,
        tenant_id: str,
        client_context: dict[str, Any],
        *,
        auto_dispatch: bool = True,
    ) -> dict[str, Any]:
        """Fill any remaining open slots in the client's 3-post/week cadence."""
        profile = SocialStudioStore.get_or_create_profile(
            db,
            tenant_id,
            tenant_slug=client_context.get("slug"),
            city=client_context.get("city"),
            industry=client_context.get("industry"),
        )
        now = utcnow()
        iso_year, iso_week, _ = now.isocalendar()
        generated_posts: list[SocialPost] = []
        skipped_reasons: list[str] = []

        # Generate all open slots first (sequentially so topic deduplication sees prior slots)
        for _ in range(MAX_POSTS_PER_WEEK):
            next_slot = SocialStudioStore.claim_next_weekly_slot(
                db, tenant_id, iso_year, iso_week, max_slots=profile.posts_per_week
            )
            if next_slot is None:
                break
            try:
                res = SocialStudioService.generate_post(
                    db,
                    tenant_id,
                    client_context,
                    {
                        "iso_year": iso_year,
                        "iso_week": iso_week,
                        "slot_index": next_slot,
                        "auto_dispatch": False,
                    },
                )
                post_obj = SocialStudioStore.get_post(db, tenant_id, res["post"]["id"])
                if post_obj is not None:
                    generated_posts.append(post_obj)
            except ValueError as exc:
                skipped_reasons.append(str(exc))
                break

        # Dispatch newly created qualified posts concurrently (pure HTTP, no DB access in worker threads)
        if auto_dispatch and generated_posts:
            dispatchable = [
                p for p in generated_posts
                if p.qualification_status in {"qualified", "healed"}
            ]
            if dispatchable:
                def _dispatch_one(post_obj: SocialPost) -> SocialPost:
                    video_bytes = get_or_render_post_video_bytes(post_obj)
                    return dispatch_qualified_post_to_platforms(
                        post_obj,
                        profile,
                        video_bytes,
                        publish_now=False,
                    )

                if len(dispatchable) > 1:
                    with ThreadPoolExecutor(max_workers=min(3, len(dispatchable))) as pool:
                        updated_posts = list(pool.map(_dispatch_one, dispatchable))
                else:
                    updated_posts = [_dispatch_one(dispatchable[0])]

                updated_by_id = {p.id: p for p in updated_posts}
                for idx, post_obj in enumerate(generated_posts):
                    if post_obj.id in updated_by_id:
                        saved = SocialStudioStore.save_post(db, updated_by_id[post_obj.id])
                        generated_posts[idx] = saved

        created_posts = [p.to_dict() for p in generated_posts]
        slots_used = SocialStudioStore.get_weekly_slots_used(db, tenant_id, iso_year, iso_week)
        return {
            "tenant_id": tenant_id,
            "iso_year": iso_year,
            "iso_week": iso_week,
            "created_count": len(created_posts),
            "created_posts": created_posts,
            "slots_used": slots_used,
            "skipped_reasons": skipped_reasons,
        }

    @staticmethod
    def run_due_scheduled_posts(db: Session) -> dict[str, Any]:
        """Dispatch any due scheduled social posts (e.g. deferred Reddit posts whose slot time has arrived)."""
        now = utcnow()
        due_posts = SocialStudioStore.list_due_scheduled_posts(db, now=now)
        dispatched_ids: list[str] = []
        failed_ids: list[str] = []

        for post in due_posts:
            profile = SocialStudioStore.get_or_create_profile(db, post.tenant_id)
            results = dict(post.platform_results or {})
            due_platforms: list[str] = []

            for plat in post.target_platforms:
                p_norm = str(plat).lower().strip()
                p_res = results.get(p_norm) or {}
                p_status = str(p_res.get("status") or "").lower().strip()
                if p_norm == "reddit":
                    # Deferred Reddit posts are stored with status="scheduled" and no post_id yet
                    if p_status in {"scheduled", ""} and not p_res.get("post_id"):
                        due_platforms.append("reddit")
                elif p_norm in {"instagram", "facebook", "youtube"}:
                    # Only dispatch video platforms if they weren't already uploaded to Upload-Post
                    req_id = str(p_res.get("request_id") or post.uploadpost_request_ids.get(p_norm) or "")
                    if p_status == "" or (p_status == "queued_for_api_key" and is_uploadpost_configured()) or (p_status == "scheduled" and not req_id):
                        due_platforms.append(p_norm)

            if not due_platforms:
                continue

            needs_video = any(p in {"instagram", "facebook", "youtube"} for p in due_platforms)
            video_bytes = get_or_render_post_video_bytes(post) if needs_video else b""
            updated = dispatch_qualified_post_to_platforms(
                post,
                profile,
                video_bytes,
                only_platforms=due_platforms,
                publish_now=False,
            )
            SocialStudioStore.save_post(db, updated)
            if updated.status in {"failed", "partial_failure"}:
                failed_ids.append(updated.id)
            else:
                dispatched_ids.append(updated.id)

        return {
            "checked": len(due_posts),
            "dispatched_count": len(dispatched_ids),
            "dispatched_ids": dispatched_ids,
            "failed_ids": failed_ids,
        }

    @staticmethod
    def process_uploadpost_webhook(db: Session, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id") or payload.get("job_id") or "").strip()
        if not request_id:
            return {"ok": False, "reason": "missing_request_id"}

        post = SocialStudioStore.find_post_by_request_id(db, request_id)
        if post is None:
            return {"ok": False, "reason": "post_not_found", "request_id": request_id}

        platform = str(payload.get("platform") or "").lower().strip()
        raw_status = str(payload.get("status") or "").lower().strip()
        err_detail = payload.get("error") or payload.get("message") or payload.get("detail")
        if not raw_status:
            if payload.get("success") is False or err_detail:
                status = "failed"
            elif payload.get("success") is True:
                status = "published"
            else:
                # Never treat an empty/unknown webhook status without explicit success=true as published
                return {
                    "ok": True,
                    "post_id": post.id,
                    "status": post.status,
                    "ignored_empty_status": True,
                }
        else:
            status = raw_status

        post_url = payload.get("url") or payload.get("post_url")

        results = dict(post.platform_results or {})
        target_platforms = [platform] if platform else [
            k for k, v in post.uploadpost_request_ids.items() if v == request_id
        ]
        for p in target_platforms:
            existing = dict(results.get(p) or {})
            existing["status"] = status
            existing["ok"] = status in {"published", "completed", "success", "scheduled"}
            if post_url:
                existing["url"] = post_url
            if status in {"failed", "error"}:
                existing["error"] = str(err_detail or existing.get("error") or f"Upload-Post {p} webhook reported {status}")
            elif existing["ok"]:
                existing.pop("error", None)
            results[p] = existing

        post.platform_results = results
        reconcile_post_status_from_results(post)

        SocialStudioStore.save_post(db, post)
        return {"ok": True, "post_id": post.id, "status": post.status}


# ---------------------------------------------------------------------------
# Standalone FastAPI Microservice Application (`emm-social-studio`)
# ---------------------------------------------------------------------------

def _create_standalone_session_factory():
    db_url = (
        os.environ.get("SOCIAL_STUDIO_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "sqlite:///./social_studio.db"
    ).strip()
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://") :]
    if db_url.startswith("postgresql://"):
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            db_url = "postgresql+psycopg://" + db_url[len("postgresql://") :]
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args, pool_pre_ping=True)
    if db_url.startswith("sqlite") or os.environ.get("SOCIAL_STUDIO_AUTO_CREATE_TABLES") == "1":
        SocialStudioBase.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


_SessionLocal = None


def get_microservice_db():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = _create_standalone_session_factory()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def verify_hmac_signature(
    request: Request,
    x_social_studio_signature: Optional[str] = Header(default=None),
) -> None:
    secret = (os.environ.get("SOCIAL_STUDIO_HMAC_SECRET") or "").strip()
    if not secret:
        # In local dev/test without secret configured, allow requests
        return
    if not x_social_studio_signature:
        raise HTTPException(status_code=401, detail="Missing X-Social-Studio-Signature header")
    body = await request.body()
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, x_social_studio_signature.strip()):
        raise HTTPException(status_code=401, detail="Invalid X-Social-Studio-Signature")


app = FastAPI(
    title="EMM Social Studio Microservice",
    version="1.0.0",
    description="Automated Educational Social Video & LLM-Citation Distribution Microservice",
)


class MicroserviceEnvelope(BaseModel):
    client_context: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "service": "emm-social-studio"}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    return {
        "status": "ready",
        "service": "emm-social-studio",
        "uploadpost_configured": is_uploadpost_configured(),
        "reddit_api": get_reddit_api_status(),
    }


@app.post("/v1/tenants/{tenant_id}/overview", dependencies=[Depends(verify_hmac_signature)])
def api_tenant_overview(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    return SocialStudioService.get_overview(db, tenant_id, body.client_context)


@app.put("/v1/tenants/{tenant_id}/profile", dependencies=[Depends(verify_hmac_signature)])
def api_update_profile(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    profile = SocialStudioService.update_profile(db, tenant_id, body.payload, body.client_context)
    return {"profile": profile.to_dict()}


@app.post("/v1/tenants/{tenant_id}/strategy/get", dependencies=[Depends(verify_hmac_signature)])
def api_get_strategy(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    strategy = SocialStudioService.get_content_strategy(db, tenant_id, body.client_context)
    return {"strategy": strategy}


@app.post("/v1/tenants/{tenant_id}/strategy/generate", dependencies=[Depends(verify_hmac_signature)])
def api_generate_strategy(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    strategy = SocialStudioService.generate_content_strategy(body.client_context)
    return {"strategy": strategy}


@app.put("/v1/tenants/{tenant_id}/strategy", dependencies=[Depends(verify_hmac_signature)])
def api_update_strategy(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    strategy = SocialStudioService.update_content_strategy(db, tenant_id, body.payload, body.client_context)
    return {"strategy": strategy}


@app.post("/v1/tenants/{tenant_id}/sync-accounts", dependencies=[Depends(verify_hmac_signature)])
def api_sync_accounts(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    return SocialStudioService.sync_connected_accounts(db, tenant_id, body.client_context)


@app.post("/v1/tenants/{tenant_id}/connect-jwt", dependencies=[Depends(verify_hmac_signature)])
def api_connect_jwt(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    return SocialStudioService.generate_connect_jwt(
        db,
        tenant_id,
        body.client_context,
        redirect_url=body.payload.get("redirect_url"),
        platforms=body.payload.get("platforms"),
    )


@app.post("/v1/tenants/{tenant_id}/qualify-topic", dependencies=[Depends(verify_hmac_signature)])
def api_qualify_topic(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    return SocialStudioService.check_topic_qualification(db, tenant_id, body.client_context, body.payload)


@app.post("/v1/tenants/{tenant_id}/voice-preview", dependencies=[Depends(verify_hmac_signature)])
def api_voice_preview(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    return SocialStudioService.preview_voiceover(db, tenant_id, body.payload)


@app.post("/v1/tenants/{tenant_id}/posts/generate", dependencies=[Depends(verify_hmac_signature)])
def api_generate_post(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    try:
        return SocialStudioService.generate_post(db, tenant_id, body.client_context, body.payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/v1/tenants/{tenant_id}/posts/{post_id}", dependencies=[Depends(verify_hmac_signature)])
def api_update_post_copy(
    tenant_id: str,
    post_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    try:
        return SocialStudioService.update_post_copy(db, tenant_id, post_id, body.client_context, body.payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/tenants/{tenant_id}/posts/{post_id}/publish", dependencies=[Depends(verify_hmac_signature)])
def api_publish_post(
    tenant_id: str,
    post_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    try:
        post = SocialStudioService.publish_post(
            db,
            tenant_id,
            post_id,
            body.client_context,
            publish_now=bool(body.payload.get("publish_now", False)),
            only_platforms=body.payload.get("only_platforms"),
        )
        return {"post": post.to_dict()}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/tenants/{tenant_id}/posts/{post_id}/retry-platform", dependencies=[Depends(verify_hmac_signature)])
def api_retry_platform(
    tenant_id: str,
    post_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    platform = str(body.payload.get("platform") or "").lower().strip()
    if not platform:
        raise HTTPException(status_code=422, detail="platform is required")
    try:
        post = SocialStudioService.publish_post(
            db,
            tenant_id,
            post_id,
            body.client_context,
            publish_now=bool(body.payload.get("publish_now", False)),
            only_platforms=[platform],
        )
        return {"post": post.to_dict()}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/v1/tenants/{tenant_id}/posts/{post_id}", dependencies=[Depends(verify_hmac_signature)])
def api_cancel_post(
    tenant_id: str,
    post_id: str,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    try:
        post = SocialStudioService.cancel_post(db, tenant_id, post_id)
        return {"post": post.to_dict()}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/tenants/{tenant_id}/run-weekly-batch", dependencies=[Depends(verify_hmac_signature)])
def api_run_weekly_batch(
    tenant_id: str,
    body: MicroserviceEnvelope,
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    return SocialStudioService.run_weekly_batch(
        db,
        tenant_id,
        body.client_context,
        auto_dispatch=bool(body.payload.get("auto_dispatch", True)),
    )


async def verify_uploadpost_webhook_auth(request: Request) -> None:
    secret = (
        os.environ.get("UPLOAD_POST_WEBHOOK_SECRET")
        or os.environ.get("SOCIAL_STUDIO_WEBHOOK_SECRET")
        or ""
    ).strip()
    if not secret:
        return
    body = await request.body()
    expected_hex = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()

    for header_name in ("x-upload-post-signature", "x-hub-signature-256", "x-webhook-signature", "x-social-studio-signature"):
        sig = (request.headers.get(header_name) or "").strip()
        if sig:
            normalized_sig = sig[len("sha256="):] if sig.lower().startswith("sha256=") else sig
            if hmac.compare_digest(expected_hex, normalized_sig):
                return

    for header_name in ("x-webhook-secret", "x-upload-post-secret"):
        val = (request.headers.get(header_name) or "").strip()
        if val and hmac.compare_digest(secret, val):
            return

    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer ") and hmac.compare_digest(secret, auth[7:].strip()):
        return

    q_token = (request.query_params.get("secret") or request.query_params.get("token") or "").strip()
    if q_token and hmac.compare_digest(secret, q_token):
        return

    raise HTTPException(status_code=401, detail="Invalid Upload-Post webhook signature or secret")


@app.post("/v1/webhooks/upload-post", dependencies=[Depends(verify_uploadpost_webhook_auth)])
def api_uploadpost_webhook(
    payload: dict[str, Any],
    db: Session = Depends(get_microservice_db),
) -> dict[str, Any]:
    return SocialStudioService.process_uploadpost_webhook(db, payload)
