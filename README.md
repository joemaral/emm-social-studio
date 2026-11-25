# EMM Social Studio

Educational multi-platform publishing service for healthcare and marketing practices.

## Reddit Integration Overview (`app/reddit_client.py`)

- **Purpose:** Publishes human-reviewed, non-promotional educational Q&A guides (`kind="self"` Markdown threads) strictly to dedicated local subreddits owned and moderated by our account.
- **Official OAuth2 Endpoints:**
  - `https://www.reddit.com/api/v1/authorize` / `https://www.reddit.com/api/v1/access_token`
  - `GET https://oauth.reddit.com/r/{subreddit}/about.json` (verifies owned subreddit)
  - `POST https://oauth.reddit.com/api/submit` (publishes self-post Q&A thread)
- **Scopes:** `identity submit modconfig modposts mysubreddits read`
- **Hard Cadence & Compliance Safeguards (`app/qualification_gate.py`):**
  - Hard-capped at **3 posts/week per subreddit**
  - Blocks promotional coupons/discounts, prescriptive medical commands, and guaranteed outcome claims
  - Requires canonical educational source citation and medical disclaimer
  - Zero scraping, zero AI model training on Reddit data, zero unsolicited commenting/DMing
