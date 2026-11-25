"""
Upload-Post API Client (`https://api.upload-post.com/api`) for EMM Social Studio.

Handles:
1. Multi-tenant User Profile provisioning (`POST /api/uploadposts/users`, `GET /api/uploadposts/users`)
2. White-Label Client Account Linking JWT generation (`POST /api/uploadposts/users/generate-jwt`)
3. Isolated Per-Platform Video Reel & Shorts publishing (`POST /api/upload`):
   - Instagram Reels: `platform[]="instagram"`, `media_type="REELS"`, `share_to_feed="true"`
   - Facebook Reels: `platform[]="facebook"`, `facebook_media_type="REELS"`, `facebook_page_id`
   - YouTube Shorts: `platform[]="youtube"`, `title` (`#Shorts`), `description` (Citation + Verbatim Transcript), `privacyStatus="public"`
4. Feed Photo publishing (`POST /api/upload_photos` with normalized sRGB `.jpg`)
5. Async upload status polling (`GET /api/uploadposts/status?request_id=...`)
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_UPLOADPOST_BASE_URL = "https://api.upload-post.com/api"


class UploadPostError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def _api_key() -> str:
    return (os.environ.get("UPLOAD_POST_API_KEY") or "").strip()


def _base_url() -> str:
    return (os.environ.get("UPLOAD_POST_BASE_URL") or DEFAULT_UPLOADPOST_BASE_URL).rstrip("/")


def is_uploadpost_configured() -> bool:
    return bool(_api_key())


def _headers(json_body: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Apikey {_api_key()}",
        "Accept": "application/json",
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _extract_error_detail(resp: requests.Response) -> str:
    try:
        data = resp.json() if resp.content else {}
        if isinstance(data, dict):
            msg = data.get("message") or data.get("error") or data.get("detail")
            if msg:
                return str(msg)
    except Exception:
        pass
    text = (resp.text or "").strip()
    return text[:300] if text else f"HTTP {resp.status_code}"


def fetch_facebook_pages(username: str) -> list[dict[str, str]]:
    """Query `GET /api/uploadposts/facebook/pages?profile=<username>` for connected Facebook Pages."""
    if not is_uploadpost_configured() or not username:
        return []
    base = _base_url()
    try:
        resp = requests.get(
            f"{base}/uploadposts/facebook/pages",
            headers=_headers(),
            params={"profile": username},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.info("Upload-Post facebook/pages for %s returned HTTP %s", username, resp.status_code)
            return []
        payload = resp.json() if resp.content else {}
        raw_pages = payload.get("pages") or payload.get("data") or (payload if isinstance(payload, list) else [])
        pages: list[dict[str, str]] = []
        for p in raw_pages:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("page_id") or p.get("id") or "").strip()
            pname = str(p.get("page_name") or p.get("name") or p.get("display_name") or "").strip()
            if pid:
                pages.append({"page_id": pid, "page_name": pname or pid})
        return pages
    except Exception as exc:
        logger.warning("Upload-Post facebook/pages lookup failed for %s: %s", username, exc)
        return []


def fetch_user_profile(username: str) -> dict[str, Any]:
    """Fetch the latest connected social accounts and Facebook Pages for a tenant profile."""
    if not is_uploadpost_configured():
        return {
            "username": username,
            "configured": False,
            "connected_platforms": [],
            "available_facebook_pages": [],
            "note": "UPLOAD_POST_API_KEY is not configured yet; running in preview/staging mode.",
        }

    base = _base_url()
    try:
        prof_raw: Optional[dict[str, Any]] = None
        resolved_username = username

        # 1. Try direct profile lookup first (`GET /api/uploadposts/users/{username}`)
        single_resp = requests.get(
            f"{base}/uploadposts/users/{username}",
            headers=_headers(),
            timeout=15,
        )
        if single_resp.status_code == 200:
            payload = single_resp.json() if single_resp.content else {}
            candidate = payload.get("profile") or payload.get("user") or payload
            if isinstance(candidate, dict) and (candidate.get("username") or candidate.get("social_accounts") is not None):
                prof_raw = candidate
                resolved_username = str(candidate.get("username") or username)

        # 2. Fall back to listing all profiles (`GET /api/uploadposts/users`)
        if prof_raw is None:
            list_resp = requests.get(f"{base}/uploadposts/users", headers=_headers(), timeout=15)
            if list_resp.status_code != 200:
                raise UploadPostError(
                    f"Upload-Post profile lookup failed ({_extract_error_detail(list_resp)})",
                    status_code=list_resp.status_code,
                    retryable=list_resp.status_code in {429, 500, 502, 503, 504},
                )
            payload = list_resp.json() if list_resp.content else {}
            profiles = payload.get("profiles") or payload.get("users") or (payload if isinstance(payload, list) else [])
            existing_profiles = [p for p in profiles if isinstance(p, dict)]
            for prof in existing_profiles:
                if prof.get("username") == username:
                    prof_raw = prof
                    resolved_username = username
                    break

        if prof_raw is None:
            return {
                "username": username,
                "configured": True,
                "found": False,
                "connected_platforms": [],
                "available_facebook_pages": [],
            }

        fb_pages = fetch_facebook_pages(resolved_username)
        normalized = _normalize_profile_response(resolved_username, prof_raw, facebook_pages=fb_pages)
        normalized["found"] = True
        return normalized
    except UploadPostError:
        raise
    except requests.RequestException as exc:
        raise UploadPostError(f"Upload-Post network error: {exc}", retryable=True) from exc


def ensure_user_profile(username: str) -> dict[str, Any]:
    """Ensure the tenant's Upload-Post profile exists and return connected platform details."""
    if not is_uploadpost_configured():
        return {
            "username": username,
            "configured": False,
            "connected_platforms": [],
            "available_facebook_pages": [],
            "note": "UPLOAD_POST_API_KEY is not configured yet; running in preview/staging mode.",
        }

    base = _base_url()
    try:
        # 1. Inspect existing profiles first
        existing_profiles: list[dict[str, Any]] = []
        resp = requests.get(f"{base}/uploadposts/users", headers=_headers(), timeout=20)
        if resp.status_code == 200:
            payload = resp.json() if resp.content else {}
            profiles = payload.get("profiles") or payload.get("users") or (payload if isinstance(payload, list) else [])
            existing_profiles = [p for p in profiles if isinstance(p, dict)]
            for prof in existing_profiles:
                if prof.get("username") == username:
                    fb_pages = fetch_facebook_pages(username)
                    return _normalize_profile_response(username, prof, facebook_pages=fb_pages)

        # 2. Create profile if not found
        create_resp = requests.post(
            f"{base}/uploadposts/users",
            headers=_headers(json_body=True),
            json={"username": username},
            timeout=20,
        )
        if create_resp.status_code in (200, 201):
            data = create_resp.json() if create_resp.content else {}
            fb_pages = fetch_facebook_pages(username)
            return _normalize_profile_response(username, data, facebook_pages=fb_pages)

        if create_resp.status_code == 409:
            data = create_resp.json() if create_resp.content else {}
            if isinstance(data, dict) and (data.get("profile") or data.get("user") or data.get("social_accounts") is not None):
                fb_pages = fetch_facebook_pages(username)
                return _normalize_profile_response(username, data, facebook_pages=fb_pages)
            fetched = fetch_user_profile(username)
            if fetched.get("found"):
                return fetched
            return _normalize_profile_response(username, data, facebook_pages=fetch_facebook_pages(username))

        raise UploadPostError(
            f"Upload-Post profile creation failed ({_extract_error_detail(create_resp)})",
            status_code=create_resp.status_code,
            retryable=create_resp.status_code in {429, 500, 502, 503, 504},
        )
    except UploadPostError:
        raise
    except requests.RequestException as exc:
        raise UploadPostError(f"Upload-Post network error: {exc}", retryable=True) from exc


def _normalize_profile_response(
    username: str,
    raw: dict[str, Any],
    *,
    facebook_pages: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    inner = raw.get("profile") or raw.get("user") or raw
    if not isinstance(inner, dict):
        inner = raw
    social_accounts = inner.get("social_accounts") or inner.get("connected_platforms") or {}
    connected: list[str] = []
    if isinstance(social_accounts, dict):
        for platform, info in social_accounts.items():
            if isinstance(info, dict):
                if info.get("connected", True) is False:
                    continue
                connected.append(str(platform).lower())
            elif info:
                connected.append(str(platform).lower())
    elif isinstance(social_accounts, list):
        connected = [str(p).lower() for p in social_accounts if p]

    fb_info = social_accounts.get("facebook") if isinstance(social_accounts, dict) else None
    fb_dict = fb_info if isinstance(fb_info, dict) else {}
    ig_info = social_accounts.get("instagram") if isinstance(social_accounts, dict) else None
    ig_dict = ig_info if isinstance(ig_info, dict) else {}
    yt_info = social_accounts.get("youtube") if isinstance(social_accounts, dict) else None
    yt_dict = yt_info if isinstance(yt_info, dict) else {}

    pages_list: list[dict[str, str]] = list(facebook_pages or [])
    if not pages_list and isinstance(fb_dict.get("pages"), list):
        for p in fb_dict["pages"]:
            if isinstance(p, dict) and (p.get("page_id") or p.get("id")):
                pid = str(p.get("page_id") or p.get("id")).strip()
                pname = str(p.get("page_name") or p.get("name") or pid).strip()
                pages_list.append({"page_id": pid, "page_name": pname})

    if pages_list and "facebook" not in connected:
        connected.append("facebook")

    facebook_page_id = (
        inner.get("facebook_page_id")
        or fb_dict.get("page_id")
        or fb_dict.get("facebook_page_id")
        or fb_dict.get("id")
        or (pages_list[0]["page_id"] if pages_list else None)
    )
    facebook_page_name = (
        inner.get("facebook_page_name")
        or fb_dict.get("page_name")
        or fb_dict.get("display_name")
        or fb_dict.get("name")
        or (pages_list[0]["page_name"] if pages_list else None)
    )
    instagram_account_id = (
        inner.get("instagram_account_id")
        or ig_dict.get("account_id")
        or ig_dict.get("ig_id")
        or ig_dict.get("user_id")
        or ig_dict.get("id")
    )
    instagram_handle = (
        inner.get("instagram_handle")
        or ig_dict.get("handle")
        or ig_dict.get("username")
        or ig_dict.get("display_name")
        or ig_dict.get("name")
        or (ig_info if isinstance(ig_info, str) and ig_info not in ("connected", "true") else None)
    )
    youtube_channel_id = (
        inner.get("youtube_channel_id")
        or yt_dict.get("channel_id")
        or yt_dict.get("id")
    )
    youtube_channel_title = (
        inner.get("youtube_channel_title")
        or yt_dict.get("display_name")
        or yt_dict.get("title")
        or yt_dict.get("channel_title")
        or yt_dict.get("name")
        or yt_dict.get("handle")
        or (yt_info if isinstance(yt_info, str) and yt_info not in ("connected", "true") else None)
    )

    return {
        "username": username,
        "configured": True,
        "connected_platforms": sorted(set(connected)),
        "facebook_page_id": str(facebook_page_id).strip() if facebook_page_id else None,
        "facebook_page_name": str(facebook_page_name).strip() if facebook_page_name else None,
        "available_facebook_pages": pages_list,
        "instagram_account_id": str(instagram_account_id).strip() if instagram_account_id else None,
        "instagram_handle": str(instagram_handle).strip() if instagram_handle else None,
        "youtube_channel_id": str(youtube_channel_id).strip() if youtube_channel_id else None,
        "youtube_channel_title": str(youtube_channel_title).strip() if youtube_channel_title else None,
    }


def generate_white_label_connect_jwt(
    *,
    username: str,
    redirect_url: str,
    connect_title: str,
    connect_description: str,
    logo_image: Optional[str] = None,
    platforms: Optional[list[str]] = None,
    profile_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generate a 48-hour white-label connection URL via `POST /api/uploadposts/users/generate-jwt`."""
    expires_at = datetime.now(timezone.utc) + timedelta(hours=48)
    # Reddit is handled directly via the Official Reddit Developer API (reddit_client.py),
    # never through Upload-Post.
    filtered = [p for p in (platforms or ["instagram", "facebook", "youtube"]) if p.lower() != "reddit"]
    target_platforms = filtered or ["instagram", "facebook", "youtube"]

    if not is_uploadpost_configured():
        # Return a safe preview link when API key is not yet provisioned
        return {
            "access_url": f"https://app.upload-post.com/connect?user={username}&preview=1",
            "expires_at": expires_at.isoformat(),
            "username": username,
            "platforms": target_platforms,
            "configured": False,
        }

    prof_info = profile_info if profile_info is not None else ensure_user_profile(username)
    resolved_username = str(prof_info.get("username") or username)
    body: dict[str, Any] = {
        "username": resolved_username,
        "redirect_url": redirect_url,
        "connect_title": connect_title,
        "connect_description": connect_description,
        "platforms": target_platforms,
        "show_calendar": False,
    }
    if logo_image:
        body["logo_image"] = logo_image

    try:
        resp = requests.post(
            f"{_base_url()}/uploadposts/users/generate-jwt",
            headers=_headers(json_body=True),
            json=body,
            timeout=20,
        )
        if resp.status_code not in (200, 201):
            raise UploadPostError(
                f"Upload-Post generate-jwt failed ({_extract_error_detail(resp)})",
                status_code=resp.status_code,
                retryable=resp.status_code in {429, 500, 502, 503, 504},
            )
        data = resp.json() if resp.content else {}
        access_url = data.get("access_url") or data.get("url") or ""
        if not access_url:
            raise UploadPostError("Upload-Post generate-jwt did not return an access_url.")
        return {
            "access_url": access_url,
            "expires_at": expires_at.isoformat(),
            "username": resolved_username,
            "profile_info": prof_info,
            "platforms": target_platforms,
            "configured": True,
        }
    except UploadPostError:
        raise
    except requests.RequestException as exc:
        raise UploadPostError(f"Upload-Post generate-jwt network error: {exc}", retryable=True) from exc


def publish_platform_video(
    *,
    platform: str,
    username: str,
    video_bytes: bytes,
    title: str,
    description: Optional[str] = None,
    facebook_page_id: Optional[str] = None,
    scheduled_date: Optional[str] = None,
    timezone_name: Optional[str] = None,
    first_comment: Optional[str] = None,
) -> dict[str, Any]:
    """Dispatch a single platform video (Instagram Reel, Facebook Reel, or YouTube Short) in isolation.

    Isolating each platform call ensures a retry on one network never double-posts to another.
    """
    platform_norm = platform.lower().strip()
    if platform_norm not in {"instagram", "facebook", "youtube"}:
        raise ValueError(f"Unsupported video platform for Upload-Post: {platform}")

    form_data: list[tuple[str, str]] = [
        ("user", username),
        ("platform[]", platform_norm),
        ("title", title),
        ("async_upload", "true"),
    ]

    if platform_norm == "instagram":
        form_data.append(("media_type", "REELS"))
        form_data.append(("share_to_feed", "true"))
        if first_comment:
            form_data.append(("first_comment", first_comment))
    elif platform_norm == "facebook":
        form_data.append(("facebook_media_type", "REELS"))
        if facebook_page_id:
            form_data.append(("facebook_page_id", facebook_page_id))
        if description:
            form_data.append(("description", description))
    elif platform_norm == "youtube":
        form_data.append(("privacyStatus", "public"))
        form_data.append(("categoryId", "27"))  # Education
        form_data.append(("containsSyntheticMedia", "true"))
        if description:
            form_data.append(("description", description))

    if scheduled_date:
        form_data.append(("scheduled_date", scheduled_date))
        form_data.append(("timezone", timezone_name or "UTC"))

    form_params_dict = dict(form_data)
    if not is_uploadpost_configured():
        sim_req_id = f"sim_{platform_norm}_{uuid.uuid4().hex[:10]}"
        return {
            "ok": True,
            "platform": platform_norm,
            "status": "scheduled" if scheduled_date else "queued_for_api_key",
            "request_id": sim_req_id,
            "media_type": form_params_dict.get("media_type"),
            "facebook_media_type": form_params_dict.get("facebook_media_type"),
            "form_parameters": form_params_dict,
            "note": "Queued locally; will dispatch to Upload-Post once UPLOAD_POST_API_KEY is connected.",
        }

    files = {
        "video": (f"{platform_norm}_reel.mp4", video_bytes, "video/mp4"),
    }
    try:
        resp = requests.post(
            f"{_base_url()}/upload",
            headers=_headers(json_body=False),
            data=form_data,
            files=files,
            timeout=120,
        )
        if resp.status_code not in (200, 201, 202):
            raise UploadPostError(
                f"Upload-Post {platform_norm} upload failed ({_extract_error_detail(resp)})",
                status_code=resp.status_code,
                retryable=resp.status_code in {429, 500, 502, 503, 504},
            )
        payload = resp.json() if resp.content else {}
        if isinstance(payload, dict) and payload.get("success") is False:
            err_detail = payload.get("message") or payload.get("error") or "Upload-Post rejected upload"
            raise UploadPostError(f"Upload-Post {platform_norm} rejected upload: {err_detail}")
        request_id = payload.get("request_id") or payload.get("job_id") or f"up_{uuid.uuid4().hex[:10]}"
        post_url = payload.get("url") or (payload.get("results", {}).get(platform_norm, {}) or {}).get("url")
        return {
            "ok": True,
            "platform": platform_norm,
            "status": "scheduled" if scheduled_date else ("published" if post_url else "publishing"),
            "request_id": str(request_id),
            "url": post_url,
            "form_parameters": dict(form_data),
        }
    except UploadPostError:
        raise
    except requests.RequestException as exc:
        raise UploadPostError(f"Upload-Post {platform_norm} upload network error: {exc}", retryable=True) from exc


def get_upload_status(request_id: str) -> dict[str, Any]:
    """Poll `GET /api/uploadposts/status?request_id=...` for async video upload completion."""
    if not is_uploadpost_configured() or request_id.startswith("sim_"):
        return {"request_id": request_id, "status": "scheduled"}

    try:
        resp = requests.get(
            f"{_base_url()}/uploadposts/status",
            headers=_headers(),
            params={"request_id": request_id},
            timeout=20,
        )
        if resp.status_code != 200:
            raise UploadPostError(
                f"Upload-Post status check returned HTTP {resp.status_code}",
                status_code=resp.status_code,
                retryable=resp.status_code in {429, 500, 502, 503, 504},
            )
        return resp.json() if resp.content else {"request_id": request_id, "status": "unknown"}
    except UploadPostError:
        raise
    except requests.RequestException as exc:
        raise UploadPostError(f"Upload-Post status network error: {exc}", retryable=True) from exc
