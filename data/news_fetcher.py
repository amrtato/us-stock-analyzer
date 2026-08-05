"""
News fetcher: RSS feeds + optional NewsAPI.
Returns headlines tagged to specific tickers where possible.
"""
import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import requests

from config import NEWS_FEEDS, NEWS_API_KEY

log = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "surge", "soar", "rally", "beat", "upgrade", "bullish", "growth", "profit",
    "record", "breakout", "strong", "gain", "rise", "outperform", "buy", "boost",
    "exceed", "expand", "partnership", "deal", "innovation", "revenue", "earnings",
    "dividend", "buyback", "acquisition", "approved", "launch", "award",
}

NEGATIVE_WORDS = {
    "plunge", "crash", "slump", "miss", "downgrade", "bearish", "loss", "decline",
    "weak", "fall", "underperform", "sell", "cut", "layoff", "recall", "fine",
    "lawsuit", "fraud", "investigation", "debt", "default", "warning", "risk",
    "volatile", "concern", "drop", "tumble", "disappointing", "tariff", "ban",
}


def _score_headline(text: str) -> float:
    """Return sentiment score in [-1, +1] from keyword counting."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _is_recent(published: str, hours: int = 48) -> bool:
    """Check if a news item is within the last N hours."""
    try:
        import email.utils
        dt = email.utils.parsedate_to_datetime(published)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return dt > cutoff
    except Exception:
        return True  # assume recent if we can't parse


def fetch_rss_news(tickers: list[str], max_per_feed: int = 30) -> dict[str, list[dict]]:
    """
    Pull headlines from configured RSS feeds and map them to tickers.
    Returns {ticker: [{"title": ..., "score": ..., "source": ..., "url": ...}]}
    """
    # Build name→ticker reverse map for matching
    ticker_set = set(tickers)
    all_headlines: list[dict] = []

    for feed_url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_per_feed]:
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                link    = entry.get("link", "")
                pub     = entry.get("published", "")

                if not _is_recent(pub):
                    continue

                combined = f"{title} {summary}"
                score = _score_headline(combined)

                # Match tickers mentioned in headline
                found = []
                for ticker in ticker_set:
                    # Match "$AAPL" or "AAPL " style mentions
                    pattern = rf"\b{re.escape(ticker)}\b"
                    if re.search(pattern, combined):
                        found.append(ticker)

                all_headlines.append({
                    "title":    title,
                    "summary":  summary[:300],
                    "score":    score,
                    "tickers":  found,
                    "source":   feed.feed.get("title", feed_url),
                    "url":      link,
                    "published": pub,
                })
        except Exception as exc:
            log.warning("RSS feed failed (%s): %s", feed_url, exc)

    # Group by ticker
    result: dict[str, list[dict]] = {t: [] for t in tickers}
    general_market: list[dict] = []

    for item in all_headlines:
        if item["tickers"]:
            for t in item["tickers"]:
                if t in result:
                    result[t].append(dict(item, generic=False))
        else:
            general_market.append(dict(item, generic=True))

    # Market-wide headlines are still attached — they are useful context — but
    # they are tagged `generic` so the sentiment scorer can exclude them.
    #
    # Previously they were merged in untagged, which meant that for the great
    # majority of tickers (few headlines literally contain "MRVL") the entire
    # 18%-weighted sentiment factor was the *same* market-wide noise applied to
    # every stock. Worse, RSS ordering changes between polls, so the same stock
    # scored differently on consecutive refreshes and rankings reshuffled with
    # no market move behind it.
    for t in tickers:
        result[t].extend(general_market[:5])

    return result


def fetch_newsapi(tickers: list[str], api_key: str = NEWS_API_KEY) -> dict[str, list[dict]]:
    """
    Pull stock-specific news from NewsAPI (free tier: 100 req/day).
    Only called when NEWS_API_KEY is set.
    """
    if not api_key:
        return {}

    result: dict[str, list[dict]] = {}
    base = "https://newsapi.org/v2/everything"
    from_dt = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    for ticker in tickers[:20]:  # conserve quota
        try:
            resp = requests.get(base, params={
                "q":        f'"{ticker}" stock',
                "from":     from_dt,
                "sortBy":   "relevancy",
                "language": "en",
                "pageSize": 10,
                "apiKey":   api_key,
            }, timeout=10)
            data = resp.json()
            articles = data.get("articles", [])
            result[ticker] = [
                {
                    "title":   a.get("title", ""),
                    "summary": (a.get("description") or "")[:300],
                    "score":   _score_headline(f"{a.get('title','')} {a.get('description','')}"),
                    "source":  a.get("source", {}).get("name", ""),
                    "url":     a.get("url", ""),
                    "tickers": [ticker],
                }
                for a in articles if a.get("title")
            ]
        except Exception as exc:
            log.warning("NewsAPI failed for %s: %s", ticker, exc)

    return result


def aggregate_ticker_news(ticker: str, news_map: dict) -> dict:
    """
    Summarise news signals for a single ticker.

    Only headlines that actually name the ticker drive `avg_score` and `count`.
    Market-wide items are returned separately as context so the UI can still
    show them, but they no longer move a single stock's sentiment score — that
    is what made every stock share one headline and made scores drift between
    refreshes.
    """
    articles = news_map.get(ticker, [])
    specific = [a for a in articles if not a.get("generic")]
    generic  = [a for a in articles if a.get("generic")]

    if not specific:
        return {
            "count":       0,
            "avg_score":   0.0,
            "headline":    "",
            "snippets":    [],
            "market_context": [
                f"{a['title']} ({a['source']})" for a in generic[:3]
            ],
        }

    scores    = [a["score"] for a in specific]
    avg_score = sum(scores) / len(scores)
    top       = sorted(specific, key=lambda x: abs(x["score"]), reverse=True)

    return {
        "count":     len(specific),
        "avg_score": avg_score,
        "headline":  top[0]["title"] if top else "",
        "snippets":  [f"{a['title']} ({a['source']})" for a in top[:5]],
        "market_context": [
            f"{a['title']} ({a['source']})" for a in generic[:3]
        ],
    }
