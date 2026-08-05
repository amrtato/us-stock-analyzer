"""
News & market sentiment analyser.

Combines:
  1. RSS headline keyword sentiment
  2. Volume of news (buzz signal)
  3. Price-implied sentiment (gap, volatility)
Returns a score [0-100] and signal strings.
"""
import logging
import math

log = logging.getLogger(__name__)


def _buzz_score(article_count: int) -> float:
    """More recent news = higher market attention."""
    if article_count >= 10: return 80
    if article_count >= 5:  return 65
    if article_count >= 2:  return 55
    if article_count == 1:  return 50
    # Absence of ticker-specific coverage is genuinely unknown, not bearish —
    # return true neutral. (This used to return 40 while every stock was fed
    # market-wide headlines, so the branch was effectively unreachable; now
    # that only real per-ticker news counts, it is the common case.)
    return 50


def _sentiment_to_score(avg_score: float) -> float:
    """Map raw sentiment [-1,+1] to [0,100]."""
    clamped = max(-1.0, min(1.0, avg_score))
    return 50 + clamped * 40


def _price_sentiment(change_pct: float, vol_ratio: float) -> float:
    """
    Day's price action is itself a sentiment proxy.
    Big up-day on volume = very positive; down-day on volume = very negative.
    """
    direction = change_pct          # e.g. +2.3 or -1.7
    conviction = min(vol_ratio, 3)  # cap at 3x

    raw = direction * conviction    # range roughly [-9, +9]
    score = 50 + raw * 4           # scale: ±9 → ±36 around 50
    return max(10, min(90, score))


def analyse_sentiment(
    ticker: str,
    news_summary: dict,
    quote: dict,
) -> dict:
    """
    news_summary: output of aggregate_ticker_news()
    quote: {price, change_pct, volume, prev_close, ...} from fetcher
    """
    count     = news_summary.get("count", 0)
    avg_score = news_summary.get("avg_score", 0.0)
    headline  = news_summary.get("headline", "")
    snippets  = news_summary.get("snippets", [])

    change_pct = quote.get("change_pct", 0.0) if quote else 0.0
    vol_ratio  = quote.get("volume", 1) / max(quote.get("avg_volume", 1), 1) if quote else 1.0

    # Sub-scores
    s_news_score = _sentiment_to_score(avg_score)
    s_buzz       = _buzz_score(count)
    s_price_act  = _price_sentiment(change_pct, vol_ratio)

    composite = s_news_score * 0.45 + s_buzz * 0.25 + s_price_act * 0.30

    signals = []
    if avg_score > 0.3:    signals.append(f"Positive news sentiment (+{avg_score:.2f})")
    elif avg_score < -0.3: signals.append(f"Negative news sentiment ({avg_score:.2f})")
    if count >= 5:         signals.append(f"High news volume ({count} articles today)")
    if change_pct > 2:    signals.append(f"Strong price action +{change_pct:.1f}%")
    elif change_pct < -2: signals.append(f"Negative price action {change_pct:.1f}%")
    if headline:           signals.append(f"Top headline: {headline[:80]}")

    return {
        "score":    round(composite, 2),
        "signals":  signals,
        "snippets": snippets,
        "subscores": {
            "news_sentiment": round(s_news_score, 1),
            "buzz":           round(s_buzz, 1),
            "price_action":   round(s_price_act, 1),
        },
        "raw": {
            "avg_sentiment": round(avg_score, 3),
            "article_count": count,
            "change_pct":    round(change_pct, 2),
        },
    }
