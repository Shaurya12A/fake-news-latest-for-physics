"""
==============================================================================
 FAKE NEWS DETECTOR  -  app.py
==============================================================================
A single-file Streamlit application that analyzes a news article (via URL or
pasted text) and produces:

  - Sensationalism Score
  - Clickbait Score
  - Web Corroboration Score (live search via DuckDuckGo + Google News RSS)
  - Journalistic Tone Score
  - Composite Truth Index + verdict
  - Session history with CSV / JSON download

No paid APIs are used anywhere. Web corroboration relies on:
  - duckduckgo-search (free, unofficial DuckDuckGo client)
  - Google News RSS   (free, public RSS endpoint, no key required)

--------------------------------------------------------------------------
INSTALL
--------------------------------------------------------------------------
pip install streamlit requests beautifulsoup4 feedparser duckduckgo-search \
            scikit-learn textstat textblob lxml

RUN
--------------------------------------------------------------------------
streamlit run app.py
==============================================================================
"""

import io
import re
import csv
import json
import difflib
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup
import feedparser
import streamlit as st

# ---------------------------------------------------------------------------
# Optional / soft dependencies - the app degrades gracefully if missing
# ---------------------------------------------------------------------------
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


# ==============================================================================
# LEXICONS / REFERENCE DATA  (starter lists - expand freely)
# ==============================================================================

# Trimmed to genuinely hyperbolic/tabloid words. Common words that show up
# constantly in ordinary, legitimate journalism ("slams", "blast", "crisis",
# "chaos", "panic", "erupts", "brutal") were removed - they were causing
# real, sober news reporting to get mis-scored as sensationalist.
SENSATIONAL_WORDS = {
    "shocking", "bombshell", "explosive", "outrageous", "unbelievable",
    "miracle", "scandalous", "meltdown", "apocalyptic", "doomsday", "insane",
    "jaw-dropping", "mind-blowing", "mind-boggling", "terrifying", "horrifying",
    "sensational", "astonishing", "secretly", "you won't believe",
}

# Each pattern carries its own weight. Ambiguous patterns that legitimate
# journalism also uses often (e.g. explainer headlines ending in "?", or
# "Here's why...") were removed or down-weighted so real news stops
# getting flagged as clickbait.
CLICKBAIT_PATTERNS = [
    (r"\byou won'?t believe\b", 22),
    (r"\bwhat happens next\b", 18),
    (r"\bnumber \d+ will\b", 18),
    (r"\bwill blow your mind\b", 20),
    (r"\bcan'?t even\b", 12),
    (r"^\s*\d+\s+(reasons|ways|things|facts|signs|secrets|times)", 12),
    (r"\bthis one (weird|simple)?\s*trick\b", 22),
    (r"\bdoctors hate\b", 22),
    (r"\bwhat (they|he|she) did next\b", 16),
    (r"\bgone (wrong|viral)\b", 10),
    (r"\bbroke the internet\b", 14),
]

# Attribution phrases used by the tone analyzer
ATTRIBUTION_PHRASES = [
    "according to", "said", "stated", "reported", "told", "confirmed",
    "announced", "claims", "alleges", "sources say", "officials said",
]

# Small illustrative starter lists - NOT exhaustive. Extend for production use.
CREDIBLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "bloomberg.com", "aljazeera.com", "cnn.com", "abcnews.go.com",
    "cbsnews.com", "nbcnews.com", "pbs.org", "economist.com", "ft.com",
}

LOW_CREDIBILITY_DOMAINS = {
    "theonion.com", "worldnewsdailyreport.com", "empirenews.net",
    "nationalreport.net", "huzlers.com", "clickhole.com",
}


# ==============================================================================
# HELPERS
# ==============================================================================

def split_sentences(text):
    """Lightweight sentence splitter (no NLTK download required)."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def get_domain(url):
    if not url:
        return ""
    try:
        return urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        return ""


# ==============================================================================
# 1. ARTICLE EXTRACTION (from URL)
# ==============================================================================

def extract_article_from_url(url):
    """Fetch a URL and pull out a best-effort title + body text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 FakeNewsDetector/1.0"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
    except Exception as e:
        return None, f"Could not fetch the URL ({e})"

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        return None, f"Could not parse the page ({e})"

    # Title: prefer og:title, fall back to <title>
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.title and soup.title.text:
        title = soup.title.text.strip()

    # Body: gather substantial <p> tags
    paragraphs = soup.find_all("p")
    body_parts = [p.get_text(" ", strip=True) for p in paragraphs]
    body_parts = [p for p in body_parts if len(p) > 40]
    body = "\n".join(body_parts)

    if not body:
        body = soup.get_text(separator="\n", strip=True)

    if not title and body:
        title = split_sentences(body)[0][:120] if split_sentences(body) else ""

    domain = get_domain(url)
    return {"title": title, "text": body, "domain": domain, "url": url}, None


# ==============================================================================
# 2. CLICKBAIT DETECTOR
# ==============================================================================

def compute_clickbait_score(title):
    """
    Returns (score, hits). If no headline is available, returns (0.0, [])
    rather than treating the absence of a headline as a red flag.
    """
    if not title or not title.strip():
        return 0.0, []
    t = title.strip()
    score = 0
    hits = []

    for pat, weight in CLICKBAIT_PATTERNS:
        if re.search(pat, t, re.IGNORECASE | re.MULTILINE):
            score += weight
            hits.append(pat)

    exclaim = t.count("!")
    if exclaim:
        score += min(exclaim * 6, 12)
        hits.append(f"{exclaim} exclamation mark(s)")

    caps_words = re.findall(r"\b[A-Z]{3,}\b", t)
    if caps_words:
        score += min(len(caps_words) * 6, 12)
        hits.append(f"{len(caps_words)} ALL-CAPS word(s)")

    return round(min(score, 100), 1), hits


# ==============================================================================
# 3. SENSATIONALISM SCORER
# ==============================================================================

def compute_sensationalism_score(text):
    if not text or not text.strip():
        return 0.0, []
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0, []

    found = [w for w in words if w in SENSATIONAL_WORDS]
    density = len(found) / len(words)

    sentence_enders = max(len(re.findall(r"[.!?]", text)), 1)
    exclaim_density = text.count("!") / sentence_enders

    caps_words = re.findall(r"\b[A-Z]{4,}\b", text)

    score = min(100, density * 800 + exclaim_density * 100 + len(caps_words) * 2)
    top_words = [w for w, _ in _count_and_sort(found)[:8]]
    return round(score, 1), top_words


def _count_and_sort(items):
    from collections import Counter
    return Counter(items).most_common()


# ==============================================================================
# 4. JOURNALISTIC TONE ANALYZER
# ==============================================================================

def analyze_journalistic_tone(text):
    result = {
        "score": 50.0,
        "attribution_count": 0,
        "quote_count": 0,
        "passive_ratio": 0.0,
        "flesch_reading_ease": None,
        "subjectivity": None,
        "polarity": None,
    }
    if not text or not text.strip():
        return result

    lower = text.lower()
    sentences = split_sentences(text)
    n_sent = max(len(sentences), 1)

    attribution_count = sum(lower.count(p) for p in ATTRIBUTION_PHRASES)
    quote_count = text.count('"') // 2 + text.count("\u201c")
    passive_matches = len(re.findall(r"\b(is|was|are|were|been|being|be)\s+\w+ed\b", lower))
    passive_ratio = passive_matches / n_sent
    exclaim_count = text.count("!")
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", text))

    subjectivity = None
    polarity = None
    if TEXTBLOB_AVAILABLE:
        try:
            tb = TextBlob(text)
            subjectivity = tb.sentiment.subjectivity
            polarity = tb.sentiment.polarity
        except Exception:
            pass

    flesch = None
    if TEXTSTAT_AVAILABLE:
        try:
            flesch = textstat.flesch_reading_ease(text)
        except Exception:
            pass

    score = 50.0
    score += min(attribution_count * 3, 20)
    score += min(quote_count * 2, 10)
    score -= min(exclaim_count * 5, 20)
    score -= min(caps_words * 3, 15)
    if subjectivity is not None:
        score -= subjectivity * 15
    score = max(0.0, min(100.0, score))

    result.update({
        "score": round(score, 1),
        "attribution_count": attribution_count,
        "quote_count": quote_count,
        "passive_ratio": round(passive_ratio, 3),
        "flesch_reading_ease": round(flesch, 1) if flesch is not None else None,
        "subjectivity": round(subjectivity, 3) if subjectivity is not None else None,
        "polarity": round(polarity, 3) if polarity is not None else None,
    })
    return result


# ==============================================================================
# 5. LIVE WEB SEARCH  (DuckDuckGo + Google News RSS)
# ==============================================================================

def duckduckgo_search(query, num=8):
    if not DDGS_AVAILABLE or not query.strip():
        return []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=num))
        return [
            {
                "title": r.get("title", ""),
                "link": r.get("href", r.get("link", "")),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    except Exception:
        return []


def google_news_rss_search(query, num=8):
    if not query.strip():
        return []
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        out = []
        for entry in feed.entries[:num]:
            out.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "snippet": entry.get("summary", ""),
            })
        return out
    except Exception:
        return []


# ==============================================================================
# 6. WEB CORROBORATION SCORE
# ==============================================================================

def compute_web_corroboration(headline, body_text):
    """
    Returns a dict with a 'status' field that distinguishes:
      - 'no_query'          : nothing to search with (no headline/text at all)
      - 'search_unavailable': the search backends returned literally nothing -
                               this usually means DuckDuckGo/RSS is rate-limited,
                               blocked, or unreachable, NOT that the story is fake.
                               Treated as NEUTRAL (does not penalize the article).
      - 'no_confident_match' : search worked but nothing matched closely enough.
      - 'corroborated'       : one or more matching sources were found.
    """
    if not headline or not headline.strip():
        return {
            "score": 50.0, "matches": [], "num_distinct_domains": 0,
            "credible_sources": 0, "status": "no_query",
        }

    ddg_results = duckduckgo_search(headline, num=8)
    rss_results = google_news_rss_search(headline, num=8)
    combined = ddg_results + rss_results

    # Dedupe by link
    seen = set()
    unique = []
    for r in combined:
        link = r.get("link", "")
        if link and link not in seen:
            seen.add(link)
            unique.append(r)

    if not unique:
        # The search itself came back empty-handed (both DuckDuckGo and Google
        # News RSS). A real query almost always returns *something*, so this
        # most likely means search access is currently unavailable (rate limit,
        # network block, library issue) rather than proof the story is fake.
        # Score neutrally instead of penalizing the article.
        return {
            "score": 50.0, "matches": [], "num_distinct_domains": 0,
            "credible_sources": 0, "status": "search_unavailable",
        }

    reference_doc = f"{headline} {body_text[:800]}"
    candidate_docs = [f"{u['title']} {u.get('snippet', '')}" for u in unique]

    similarities = [0.0] * len(unique)
    if SKLEARN_AVAILABLE:
        try:
            docs = [reference_doc] + candidate_docs
            vec = TfidfVectorizer(stop_words="english").fit_transform(docs)
            sims = cosine_similarity(vec[0:1], vec[1:]).flatten()
            similarities = sims.tolist()
        except Exception:
            similarities = None

    if similarities is None:
        similarities = [
            difflib.SequenceMatcher(None, headline.lower(), u["title"].lower()).ratio()
            for u in unique
        ]

    matches = []
    credible_count = 0
    for u, sim in zip(unique, similarities):
        domain = get_domain(u.get("link", ""))
        if sim >= 0.10:
            is_credible = domain in CREDIBLE_DOMAINS
            if is_credible:
                credible_count += 1
            matches.append({
                "title": u.get("title", ""),
                "link": u.get("link", ""),
                "domain": domain,
                "similarity": round(float(sim), 3),
                "credible": is_credible,
            })

    matches.sort(key=lambda m: m["similarity"], reverse=True)
    distinct_domains = len({m["domain"] for m in matches})
    avg_sim = (sum(m["similarity"] for m in matches) / len(matches)) if matches else 0.0

    if not matches:
        # Search worked (we got candidates back) but none resembled the
        # article closely enough to count as corroboration. This is a
        # genuinely useful signal (unlike search_unavailable above), but we
        # still keep it moderate rather than zeroing the article out, since
        # very fresh or niche/local stories can legitimately have few
        # indexed matches yet.
        return {
            "score": 30.0, "matches": [], "num_distinct_domains": 0,
            "credible_sources": 0, "status": "no_confident_match",
        }

    score = min(100.0, distinct_domains * 15 + credible_count * 10 + avg_sim * 40)

    return {
        "score": round(score, 1),
        "matches": matches,
        "num_distinct_domains": distinct_domains,
        "credible_sources": credible_count,
        "status": "corroborated",
    }


# ==============================================================================
# 7. DOMAIN CREDIBILITY CHECK
# ==============================================================================

def check_domain_credibility(url):
    domain = get_domain(url)
    if not domain:
        return {"label": "No URL provided (pasted text)", "modifier": 0, "domain": ""}
    if domain in CREDIBLE_DOMAINS:
        return {"label": f"'{domain}' is on the known reputable-outlet list", "modifier": 15, "domain": domain}
    if domain in LOW_CREDIBILITY_DOMAINS:
        return {"label": f"'{domain}' is a known satire/low-credibility source", "modifier": -25, "domain": domain}
    return {"label": f"'{domain}' is unrated - verify independently", "modifier": 0, "domain": domain}


# ==============================================================================
# 8. TRUTH INDEX (composite score)
# ==============================================================================

def compute_truth_index(corroboration_score, corroboration_status, sensationalism_score,
                         clickbait_score, domain_modifier):
    """
    Composite score. Corroboration is the strongest signal, but when the
    search backend simply returned no data (corroboration_status ==
    'search_unavailable') that is NOT treated as evidence against the
    article - it's treated as neutral, so real news doesn't get dragged
    down to "fake" just because a free search API had a bad moment.
    """
    base = 50.0

    if corroboration_status != "search_unavailable":
        base += (corroboration_score - 50) * 0.45

    base -= sensationalism_score * 0.12
    base -= clickbait_score * 0.10
    base += domain_modifier
    base = max(0.0, min(100.0, base))

    if base >= 65:
        verdict = "Likely Reliable"
    elif base >= 40:
        verdict = "Mixed / Needs Verification"
    else:
        verdict = "Likely Unreliable / Fake"

    return round(base, 1), verdict


# ==============================================================================
# 9. FULL ANALYSIS PIPELINE
# ==============================================================================

def run_full_analysis(title, text, url=""):
    """
    `title` is optional. If left blank, we derive a search query from the
    opening sentence of the article text so web corroboration can still run -
    but the clickbait score is only computed against a real headline, since
    clickbait is fundamentally a headline-framing phenomenon.
    """
    title = (title or "").strip()
    text = text or ""

    derived_query = ""
    if not title and text.strip():
        sentences = split_sentences(text)
        derived_query = sentences[0][:150] if sentences else text[:150]

    query_for_search = title if title else derived_query

    clickbait_score, clickbait_hits = compute_clickbait_score(title)
    sensationalism_score, sensational_words_found = compute_sensationalism_score(text)
    tone = analyze_journalistic_tone(text)
    corroboration = compute_web_corroboration(query_for_search, text)
    domain_info = check_domain_credibility(url)

    truth_index, verdict = compute_truth_index(
        corroboration_score=corroboration["score"],
        corroboration_status=corroboration.get("status", "corroborated"),
        sensationalism_score=sensationalism_score,
        clickbait_score=clickbait_score,
        domain_modifier=domain_info["modifier"],
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "derived_query": derived_query,
        "url": url,
        "text_preview": (text[:300] + "...") if len(text) > 300 else text,
        "clickbait_score": clickbait_score,
        "clickbait_signals": clickbait_hits,
        "sensationalism_score": sensationalism_score,
        "sensational_words_found": sensational_words_found,
        "tone": tone,
        "corroboration": corroboration,
        "domain_info": domain_info,
        "truth_index": truth_index,
        "verdict": verdict,
    }


# ==============================================================================
# 10. SESSION HISTORY EXPORT HELPERS
# ==============================================================================

def flatten_record(rec):
    """Flatten a nested analysis record into a single-level dict for CSV export."""
    return {
        "timestamp": rec["timestamp"],
        "title": rec["title"] or "(no headline provided)",
        "url": rec["url"],
        "verdict": rec["verdict"],
        "truth_index": rec["truth_index"],
        "clickbait_score": rec["clickbait_score"],
        "sensationalism_score": rec["sensationalism_score"],
        "journalistic_tone_score": rec["tone"]["score"],
        "web_corroboration_score": rec["corroboration"]["score"],
        "corroboration_status": rec["corroboration"].get("status", ""),
        "distinct_corroborating_domains": rec["corroboration"]["num_distinct_domains"],
        "credible_sources_found": rec["corroboration"]["credible_sources"],
        "domain_credibility_note": rec["domain_info"]["label"],
    }


def history_to_csv(history):
    if not history:
        return ""
    rows = [flatten_record(r) for r in history]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def history_to_json(history):
    return json.dumps(history, indent=2, default=str)


# ==============================================================================
# 11. STREAMLIT UI
# ==============================================================================

def render_score_bar(label, value, help_text="", higher_is_better=True):
    st.metric(label, f"{value:.1f} / 100")
    if help_text:
        st.caption(help_text)
    st.progress(min(max(value, 0), 100) / 100)


def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="🕵️", layout="wide")

    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("🕵️ Fake News Detector")
    st.caption(
        "Live web corroboration via DuckDuckGo + Google News RSS, plus clickbait, "
        "sensationalism, and journalistic-tone scoring."
    )

    tab_analyze, tab_history = st.tabs(["🔍 Analyze", "🗂️ Session History"])

    # -------------------------------------------------------------------
    # ANALYZE TAB
    # -------------------------------------------------------------------
    with tab_analyze:
        input_mode = st.radio("Input type", ["Article URL", "Paste text manually"], horizontal=True)

        title = ""
        text = ""
        url = ""

        if input_mode == "Article URL":
            url = st.text_input("Paste a news article URL", placeholder="https://example.com/news/story")
            fetch_clicked = st.button("Fetch & Analyze", type="primary")
            if fetch_clicked and url.strip():
                with st.spinner("Fetching article..."):
                    article, err = extract_article_from_url(url.strip())
                if err:
                    st.error(err)
                    st.stop()
                title, text = article["title"], article["text"]
                if not text.strip():
                    st.error("Could not extract readable article text from this URL.")
                    st.stop()
                st.session_state["_pending_analysis"] = (title, text, url.strip())

        else:
            title = st.text_input(
                "Headline / Title (optional)",
                placeholder="Enter the article headline - or leave blank to auto-detect from the text",
            )
            text = st.text_area("Article text", height=250, placeholder="Paste the full article text here...")
            analyze_clicked = st.button("Analyze", type="primary")
            if analyze_clicked:
                if not text.strip():
                    st.error("Please paste some article text to analyze.")
                    st.stop()
                st.session_state["_pending_analysis"] = (title.strip(), text, "")

        if "_pending_analysis" in st.session_state:
            p_title, p_text, p_url = st.session_state.pop("_pending_analysis")
            with st.spinner("Running analysis (searching the live web for corroboration)..."):
                result = run_full_analysis(p_title, p_text, p_url)
            st.session_state.history.append(result)
            st.session_state["_last_result"] = result

        if "_last_result" in st.session_state:
            r = st.session_state["_last_result"]
            st.divider()
            display_title = r["title"] or r.get("derived_query") or "(untitled)"
            st.subheader(f"Results for: {display_title}")
            if not r["title"]:
                st.caption("No headline was provided - the search query above was auto-detected from the article text.")

            verdict_color = {
                "Likely Reliable": "green",
                "Mixed / Needs Verification": "orange",
                "Likely Unreliable / Fake": "red",
            }.get(r["verdict"], "gray")
            st.markdown(f"### Truth Index: **{r['truth_index']}/100** — :{verdict_color}[{r['verdict']}]")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_score_bar("Sensationalism", r["sensationalism_score"],
                                  "Higher = more emotionally-charged language")
            with c2:
                render_score_bar("Clickbait", r["clickbait_score"],
                                  "Higher = more clickbait-style headline patterns")
            with c3:
                render_score_bar("Web Corroboration", r["corroboration"]["score"],
                                  "Higher = more independent sources report this story")
            with c4:
                render_score_bar("Journalistic Tone", r["tone"]["score"],
                                  "Higher = more neutral, attributed, quote-supported style")

            st.markdown(f"**Domain check:** {r['domain_info']['label']}")

            with st.expander("📊 Detailed breakdown"):
                st.markdown("**Clickbait signals detected:**")
                st.write(r["clickbait_signals"] or "None detected")

                st.markdown("**Sensational words found:**")
                st.write(", ".join(r["sensational_words_found"]) or "None detected")

                st.markdown("**Tone analysis details:**")
                st.json(r["tone"])

            with st.expander("🌐 Corroborating sources found on the live web"):
                status = r["corroboration"].get("status", "")
                matches = r["corroboration"]["matches"]
                if matches:
                    for m in matches:
                        badge = "✅ credible outlet" if m["credible"] else ""
                        st.markdown(
                            f"- [{m['title']}]({m['link']}) — `{m['domain']}` "
                            f"(similarity {m['similarity']:.2f}) {badge}"
                        )
                elif status == "search_unavailable":
                    st.info(
                        "DuckDuckGo and Google News RSS both returned no results for this "
                        "query right now - this usually means the free search backend is "
                        "temporarily rate-limited or unreachable, not that the story is "
                        "unverified. The Truth Index has NOT been penalized for this. "
                        "Try again shortly for a live corroboration check."
                    )
                elif status == "no_confident_match":
                    st.write(
                        "Search returned results, but none closely matched this story. "
                        "This can genuinely indicate the story is unverified/niche - or it "
                        "may just be very recent and not yet widely indexed."
                    )
                else:
                    st.write("No corroboration data available for this query.")

    # -------------------------------------------------------------------
    # HISTORY TAB
    # -------------------------------------------------------------------
    with tab_history:
        st.subheader("Session History")
        history = st.session_state.history

        if not history:
            st.info("No analyses yet in this session. Run an analysis in the Analyze tab.")
        else:
            table_rows = [flatten_record(r) for r in history]
            st.dataframe(table_rows, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "⬇️ Download as CSV",
                    data=history_to_csv(history),
                    file_name=f"fake_news_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            with col2:
                st.download_button(
                    "⬇️ Download as JSON",
                    data=history_to_json(history),
                    file_name=f"fake_news_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
            with col3:
                if st.button("🗑️ Clear history"):
                    st.session_state.history = []
                    st.session_state.pop("_last_result", None)
                    st.rerun()


if __name__ == "__main__":
    main()
