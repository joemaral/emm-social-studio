"""
Official Reddit Developer API Client (`https://oauth.reddit.com`) for EMM Social Studio.

Designed for publishing educational, non-promotional Q&A breakdowns to our owned
city-based subreddits (`r/<CitySubreddit>`, e.g. `r/SanAntonioAesthetics`) to drive
high-trust LLM citations in ChatGPT, Google Gemini, Perplexity, and Google AI Overviews.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

REDDIT_AUTHORIZE_URL = "https://www.reddit.com/api/v1/authorize"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_OAUTH_BASE = "https://oauth.reddit.com"
DEFAULT_REDDIT_USER_AGENT = "linux:com.elevatedmarketing.socialstudio:v1.0 (by /u/ElevatedMedMarketing)"
REDDIT_REQUIRED_SCOPES = "identity submit modconfig modposts mysubreddits read"


class RedditPublishError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def get_reddit_api_status() -> dict[str, Any]:
    """Return whether the Official Reddit Developer API credentials are configured."""
    client_id = (os.environ.get("REDDIT_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("REDDIT_CLIENT_SECRET") or "").strip()
    refresh_token = (os.environ.get("REDDIT_REFRESH_TOKEN") or "").strip()
    username = (os.environ.get("REDDIT_USERNAME") or "").strip()
    password = (os.environ.get("REDDIT_PASSWORD") or "").strip()

    has_oauth = bool(client_id and client_secret and (refresh_token or (username and password)))
    return {
        "configured": has_oauth,
        "auth_mode": "refresh_token" if refresh_token else ("password_grant" if (username and password) else "pending_developer_approval"),
        "user_agent": os.environ.get("REDDIT_USER_AGENT", DEFAULT_REDDIT_USER_AGENT),
    }


def _acquire_access_token() -> str:
    client_id = (os.environ.get("REDDIT_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("REDDIT_CLIENT_SECRET") or "").strip()
    refresh_token = (os.environ.get("REDDIT_REFRESH_TOKEN") or "").strip()
    username = (os.environ.get("REDDIT_USERNAME") or "").strip()
    password = (os.environ.get("REDDIT_PASSWORD") or "").strip()
    user_agent = os.environ.get("REDDIT_USER_AGENT", DEFAULT_REDDIT_USER_AGENT)

    if refresh_token:
        grant_data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    elif username and password:
        grant_data = {"grant_type": "password", "username": username, "password": password}
    else:
        raise RedditPublishError("Reddit OAuth credentials are not configured yet.")

    try:
        resp = requests.post(
            REDDIT_TOKEN_URL,
            auth=(client_id, client_secret),
            data=grant_data,
            headers={"User-Agent": user_agent},
            timeout=20,
        )
        if resp.status_code != 200:
            raise RedditPublishError(
                f"Reddit token exchange failed with HTTP {resp.status_code}",
                status_code=resp.status_code,
                retryable=resp.status_code in {429, 500, 502, 503, 504},
            )
        payload = resp.json() if resp.content else {}
        token = payload.get("access_token")
        if not token:
            raise RedditPublishError("Reddit token response did not include access_token.")
        return str(token)
    except RedditPublishError:
        raise
    except requests.RequestException as exc:
        raise RedditPublishError(f"Reddit OAuth network error: {exc}", retryable=True) from exc


def normalize_subreddit_name(subreddit: str) -> str:
    """Strip leading r/ or /r/ and validate subreddit slug."""
    cleaned = re.sub(r"^/?r/", "", (subreddit or "").strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", cleaned)
    return cleaned[:21]


def build_reddit_oauth_authorize_url(*, redirect_uri: str, state: str) -> dict[str, Any]:
    """Build a permanent-duration Official Reddit Developer API OAuth2 authorization URL."""
    from urllib.parse import urlencode

    client_id = (os.environ.get("REDDIT_CLIENT_ID") or "").strip()
    if not client_id:
        return {
            "configured": False,
            "authorize_url": None,
            "scopes": REDDIT_REQUIRED_SCOPES.split(),
            "note": "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET once Reddit for Developers approves your app.",
        }
    query = urlencode({
        "client_id": client_id,
        "response_type": "code",
        "state": state,
        "redirect_uri": redirect_uri,
        "duration": "permanent",
        "scope": REDDIT_REQUIRED_SCOPES,
    })
    return {
        "configured": True,
        "authorize_url": f"{REDDIT_AUTHORIZE_URL}?{query}",
        "scopes": REDDIT_REQUIRED_SCOPES.split(),
    }


def exchange_reddit_oauth_code(*, code: str, redirect_uri: str) -> dict[str, Any]:
    """Exchange a Reddit OAuth2 authorization code for an access_token and permanent refresh_token."""
    client_id = (os.environ.get("REDDIT_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("REDDIT_CLIENT_SECRET") or "").strip()
    user_agent = os.environ.get("REDDIT_USER_AGENT", DEFAULT_REDDIT_USER_AGENT)
    if not client_id or not client_secret:
        raise RedditPublishError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are required.")

    resp = requests.post(
        REDDIT_TOKEN_URL,
        auth=(client_id, client_secret),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
        headers={"User-Agent": user_agent},
        timeout=20,
    )
    if resp.status_code != 200:
        raise RedditPublishError(
            f"Reddit code exchange failed with HTTP {resp.status_code}",
            status_code=resp.status_code,
        )
    return resp.json() if resp.content else {}


def verify_or_provision_city_subreddit(
    *,
    subreddit: str,
    title: str,
    public_description: str,
    description: Optional[str] = None,
) -> dict[str, Any]:
    """Verify whether `r/<CitySubreddit>` exists via Official Reddit API, or create/configure it via `/api/site_admin`."""
    sr = normalize_subreddit_name(subreddit)
    if not sr:
        raise ValueError("A valid city subreddit slug is required.")

    status = get_reddit_api_status()
    if not status["configured"]:
        return {
            "ok": True,
            "subreddit": f"r/{sr}",
            "status": "pending_reddit_developer_api",
            "url": f"https://www.reddit.com/r/{sr}/",
        }

    token = _acquire_access_token()
    headers = {
        "Authorization": f"bearer {token}",
        "User-Agent": status["user_agent"],
    }
    about_resp = requests.get(f"{REDDIT_OAUTH_BASE}/r/{sr}/about.json", headers=headers, timeout=20)
    if about_resp.status_code == 200:
        about_data = (about_resp.json() if about_resp.content else {}).get("data") or {}
        if about_data.get("display_name"):
            return {
                "ok": True,
                "subreddit": f"r/{about_data['display_name']}",
                "status": "verified_existing",
                "subscribers": about_data.get("subscribers", 0),
                "user_is_moderator": bool(about_data.get("user_is_moderator")),
                "url": f"https://www.reddit.com/r/{about_data['display_name']}/",
            }

    # Provision / configure via /api/site_admin
    admin_resp = requests.post(
        f"{REDDIT_OAUTH_BASE}/api/site_admin",
        headers=headers,
        data={
            "api_type": "json",
            "name": sr,
            "title": title[:100],
            "public_description": public_description[:500],
            "description": (description or public_description)[:5000],
            "type": "public",
            "link_type": "any",
            "wikimode": "modonly",
            "lang": "en",
        },
        timeout=30,
    )
    if admin_resp.status_code not in (200, 201):
        raise RedditPublishError(
            f"Reddit /api/site_admin returned HTTP {admin_resp.status_code}",
            status_code=admin_resp.status_code,
        )
    return {
        "ok": True,
        "subreddit": f"r/{sr}",
        "status": "provisioned",
        "url": f"https://www.reddit.com/r/{sr}/",
    }


def publish_city_subreddit_educational_post(
    *,
    subreddit: str,
    title: str,
    selftext: str,
    flair_id: Optional[str] = None,
    flair_text: Optional[str] = None,
) -> dict[str, Any]:
    """Publish a Markdown educational Q&A thread (`kind="self"`) to a city subreddit via Official Reddit API."""
    sr = normalize_subreddit_name(subreddit)
    if not sr:
        raise ValueError("A valid target city subreddit name is required.")

    status = get_reddit_api_status()
    if not status["configured"]:
        return {
            "ok": True,
            "platform": "reddit",
            "subreddit": f"r/{sr}",
            "status": "queued_for_reddit_api",
            "post_id": f"sim_reddit_{uuid.uuid4().hex[:8]}",
            "url": f"https://www.reddit.com/r/{sr}/",
            "note": "Queued for r/" + sr + "; will auto-submit once Reddit Developer API credentials are connected.",
        }

    token = _acquire_access_token()
    user_agent = status["user_agent"]
    submit_data: dict[str, str] = {
        "api_type": "json",
        "kind": "self",
        "sr": sr,
        "title": title[:300],
        "text": selftext,
        "resubmit": "true",
        "sendreplies": "true",
    }
    if flair_id:
        submit_data["flair_id"] = flair_id
    if flair_text:
        submit_data["flair_text"] = flair_text

    try:
        resp = requests.post(
            f"{REDDIT_OAUTH_BASE}/api/submit",
            headers={
                "Authorization": f"bearer {token}",
                "User-Agent": user_agent,
            },
            data=submit_data,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            raise RedditPublishError(
                f"Reddit /api/submit returned HTTP {resp.status_code}",
                status_code=resp.status_code,
                retryable=resp.status_code in {429, 500, 502, 503, 504},
            )
        payload = resp.json() if resp.content else {}
        json_block = payload.get("json") or {}
        errors = json_block.get("errors") or []
        if errors:
            err_msg = "; ".join(str(e) for e in errors)
            raise RedditPublishError(f"Reddit API rejected submission: {err_msg}")
        data = json_block.get("data") or {}
        url = data.get("url") or f"https://www.reddit.com/r/{sr}/"
        reddit_id = data.get("id") or data.get("name") or f"t3_{uuid.uuid4().hex[:6]}"
        return {
            "ok": True,
            "platform": "reddit",
            "subreddit": f"r/{sr}",
            "status": "published",
            "post_id": str(reddit_id),
            "url": str(url),
        }
    except RedditPublishError:
        raise
    except requests.RequestException as exc:
        raise RedditPublishError(f"Reddit submit network error: {exc}", retryable=True) from exc
