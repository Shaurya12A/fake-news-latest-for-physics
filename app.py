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
  - AI-Generated-Text Likelihood Score  (catches "calm, journalistic" AI fakes)
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

--------------------------------------------------------------------------
IMPORTANT DISCLAIMER
--------------------------------------------------------------------------
This tool produces HEURISTIC, EXPLAINABLE scores based on linguistic
patterns and live search corroboration. It is a decision-support aid, not
a certified fact-checker. Always cross-check important claims with primary
sources and established fact-checking organizations.
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

SENSATIONAL_WORDS = {
    "shocking", "bombshell", "explosive", "outrageous", "devastating", "slams",
    "destroys", "terrifying", "horrific", "chilling", "stunning", "unbelievable",
    "miracle", "secret", "exposed", "scandal", "meltdown", "catastrophe", "chaos",
    "panic", "furious", "blast", "epic", "insane", "crisis", "bizarre", "savage",
    "brutal", "slammed", "erupts", "nightmare", "doom", "apocalyptic", "explodes",
}

CLICKBAIT_PATTERNS = [
    r"\byou won'?t believe\b",
    r"\bshocking\b",
    r"\bwhat happens next\b",
    r"\bnumber \d+ will\b",
    r"\bthis is why\b",
    r"\bwill blow your mind\b",
    r"\bcan'?t even\b",
    r"^\s*\d+\s+(reasons|ways|things|facts|signs|secrets|times)",
    r"\bhere'?s why\b",
    r"\bthe truth about\b",
    r"\bgone (wrong|viral)\b",
    r"\bthis one trick\b",
    r"\bdoctors hate\b",
    r"\bwhat (they|he|she) did next\b",
    r"\?\s*$",
]

# Attribution / hedging phrases used by tone + AI-likelihood analyzers
ATTRIBUTION_PHRASES = [
    "according to", "said", "stated", "reported", "told", "confirmed",
    "announced", "claims", "alleges", "sources say", "officials said",
]

AI_HEDGE_PHRASES = [
    "furthermore", "moreover", "additionally", "in conclusion", "overall",
    "in summary", "as a result", "therefore", "consequently",
    "it is important to note", "it should be noted", "in today's world",
    "in recent years", "plays a crucial role", "plays a vital role",
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
    if not title or not title.strip():
        return 0.0, []
    t = title.strip()
    score = 0
    hits = []

    for pat in CLICKBAIT_PATTERNS:
        if re.search(pat, t, re.IGNORECASE | re.MULTILINE):
            score += 12
            hits.append(pat)

    exclaim = t.count("!")
    if exclaim:
        score += min(exclaim * 8, 16)
        hits.append(f"{exclaim} exclamation mark(s)")

    caps_words = re.findall(r"\b[A-Z]{3,}\b", t)
    if caps_words:
        score += min(len(caps_words) * 8, 16)
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
# 5. AI-GENERATED-TEXT LIKELIHOOD (heuristic - flags "calm but fake" AI news)
# ==============================================================================

def analyze_ai_generation_likelihood(text):
    result = {
        "score": 0.0,
        "burstiness": None,
        "type_token_ratio": None,
        "specific_detail_density": None,
        "hedge_density": None,
    }
    if not text or not text.strip():
        return result

    sentences = split_sentences(text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    words = re.findall(r"\b\w+\b", text.lower())

    if not lengths or not words:
        return result

    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    stdev = variance ** 0.5
    burstiness = (stdev / mean_len) if mean_len > 0 else 0.0

    ttr = len(set(words)) / len(words)

    numbers = len(re.findall(r"\b\d{1,4}\b", text))
    proper_nouns = len(re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b", text))
    specific_density = (numbers + proper_nouns) / len(words)

    hedge_count = sum(text.lower().count(h) for h in AI_HEDGE_PHRASES)
    hedge_density = hedge_count / max(len(sentences), 1)

    score = 0.0
    # Low burstiness = uniform sentence lengths = a common LLM fingerprint
    if burstiness < 0.35:
        score += 30
    elif burstiness < 0.5:
        score += 15

    if ttr < 0.35:
        score += 25
    elif ttr < 0.45:
        score += 10

    if specific_density < 0.01:
        score += 25

    if hedge_density > 0.15:
        score += 20

    score = min(score, 100)

    result.update({
        "score": round(score, 1),
        "burstiness": round(burstiness, 3),
        "type_token_ratio": round(ttr, 3),
        "specific_detail_density": round(specific_density, 4),
        "hedge_density": round(hedge_density, 3),
    })
    return result


# ==============================================================================
# 6. LIVE WEB SEARCH  (DuckDuckGo + Google News RSS)
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
# 7. WEB CORROBORATION SCORE
# ==============================================================================

def compute_web_corroboration(headline, body_text):
    if not headline or not headline.strip():
        return {"score": 0.0, "matches": [], "num_distinct_domains": 0, "credible_sources": 0}

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
        return {"score": 0.0, "matches": [], "num_distinct_domains": 0, "credible_sources": 0}

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
        if sim >= 0.12:
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

    score = min(100.0, distinct_domains * 15 + credible_count * 10 + avg_sim * 40)

    return {
        "score": round(score, 1),
        "matches": matches,
        "num_distinct_domains": distinct_domains,
        "credible_sources": credible_count,
    }


# ==============================================================================
# 8. DOMAIN CREDIBILITY CHECK
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
# 9. TRUTH INDEX (composite score)
# ==============================================================================

def compute_truth_index(corroboration_score, sensationalism_score, clickbait_score,
                         ai_likelihood_score, domain_modifier):
    base = 50.0
    base += (corroboration_score - 50) * 0.35
    base -= sensationalism_score * 0.20
    base -= clickbait_score * 0.15
    base -= ai_likelihood_score * 0.15
    base += domain_modifier
    base = max(0.0, min(100.0, base))

    if base >= 70:
        verdict = "Likely Reliable"
    elif base >= 45:
        verdict = "Mixed / Needs Verification"
    else:
        verdict = "Likely Unreliable / Fake"

    return round(base, 1), verdict


# ==============================================================================
# 10. FULL ANALYSIS PIPELINE
# ==============================================================================

def run_full_analysis(title, text, url=""):
    clickbait_score, clickbait_hits = compute_clickbait_score(title)
    sensationalism_score, sensational_words_found = compute_sensationalism_score(text)
    tone = analyze_journalistic_tone(text)
    ai_likelihood = analyze_ai_generation_likelihood(text)
    corroboration = compute_web_corroboration(title, text)
    domain_info = check_domain_credibility(url)

    truth_index, verdict = compute_truth_index(
        corroboration_score=corroboration["score"],
        sensationalism_score=sensationalism_score,
        clickbait_score=clickbait_score,
        ai_likelihood_score=ai_likelihood["score"],
        domain_modifier=domain_info["modifier"],
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "url": url,
        "text_preview": (text[:300] + "...") if len(text) > 300 else text,
        "clickbait_score": clickbait_score,
        "clickbait_signals": clickbait_hits,
        "sensationalism_score": sensationalism_score,
        "sensational_words_found": sensational_words_found,
        "tone": tone,
        "ai_likelihood": ai_likelihood,
        "corroboration": corroboration,
        "domain_info": domain_info,
        "truth_index": truth_index,
        "verdict": verdict,
    }


# ==============================================================================
# 11. SESSION HISTORY EXPORT HELPERS
# ==============================================================================

def flatten_record(rec):
    """Flatten a nested analysis record into a single-level dict for CSV export."""
    return {
        "timestamp": rec["timestamp"],
        "title": rec["title"],
        "url": rec["url"],
        "verdict": rec["verdict"],
        "truth_index": rec["truth_index"],
        "clickbait_score": rec["clickbait_score"],
        "sensationalism_score": rec["sensationalism_score"],
        "journalistic_tone_score": rec["tone"]["score"],
        "ai_generation_likelihood": rec["ai_likelihood"]["score"],
        "web_corroboration_score": rec["corroboration"]["score"],
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
# 12. STREAMLIT UI
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

    st.title("🕵️ Fake News & AI-Generated News Detector")
    st.caption(
        "Free-tooling only: live web corroboration via DuckDuckGo + Google News RSS. "
        "Heuristic scoring - use as a decision-support aid, not a verdict."
    )

    with st.expander("⚠️ Read before using", expanded=False):
        st.write(
            "This tool estimates likelihoods using linguistic patterns and live search "
            "corroboration. It cannot definitively prove an article true or false. Always "
            "verify important claims against primary sources and established fact-checkers."
        )
        deps_missing = []
        if not DDGS_AVAILABLE:
            deps_missing.append("duckduckgo-search")
        if not SKLEARN_AVAILABLE:
            deps_missing.append("scikit-learn")
        if not TEXTSTAT_AVAILABLE:
            deps_missing.append("textstat")
        if not TEXTBLOB_AVAILABLE:
            deps_missing.append("textblob")
        if deps_missing:
            st.warning(
                "Optional packages not installed (some features degraded): "
                + ", ".join(deps_missing)
                + ". Install with: pip install " + " ".join(deps_missing)
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
            title = st.text_input("Headline / Title", placeholder="Enter the article headline")
            text = st.text_area("Article text", height=250, placeholder="Paste the full article text here...")
            analyze_clicked = st.button("Analyze", type="primary")
            if analyze_clicked:
                if not text.strip():
                    st.error("Please paste some article text to analyze.")
                    st.stop()
                st.session_state["_pending_analysis"] = (title or text[:80], text, "")

        if "_pending_analysis" in st.session_state:
            p_title, p_text, p_url = st.session_state.pop("_pending_analysis")
            with st.spinner("Running analysis (searching the live web for corroboration)..."):
                result = run_full_analysis(p_title, p_text, p_url)
            st.session_state.history.append(result)
            st.session_state["_last_result"] = result

        if "_last_result" in st.session_state:
            r = st.session_state["_last_result"]
            st.divider()
            st.subheader(f"Results for: {r['title'] or '(untitled)'}")

            verdict_color = {
                "Likely Reliable": "green",
                "Mixed / Needs Verification": "orange",
                "Likely Unreliable / Fake": "red",
            }.get(r["verdict"], "gray")
            st.markdown(f"### Truth Index: **{r['truth_index']}/100** — :{verdict_color}[{r['verdict']}]")

            c1, c2, c3, c4, c5 = st.columns(5)
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
            with c5:
                render_score_bar("AI-Generation Likelihood", r["ai_likelihood"]["score"],
                                  "Higher = text shows statistical fingerprints of LLM generation")

            st.markdown(f"**Domain check:** {r['domain_info']['label']}")

            with st.expander("📊 Detailed breakdown"):
                st.markdown("**Clickbait signals detected:**")
                st.write(r["clickbait_signals"] or "None detected")

                st.markdown("**Sensational words found:**")
                st.write(", ".join(r["sensational_words_found"]) or "None detected")

                st.markdown("**Tone analysis details:**")
                st.json(r["tone"])

                st.markdown("**AI-generation heuristic details:**")
                st.json(r["ai_likelihood"])

            with st.expander("🌐 Corroborating sources found on the live web"):
                matches = r["corroboration"]["matches"]
                if matches:
                    for m in matches:
                        badge = "✅ credible outlet" if m["credible"] else ""
                        st.markdown(
                            f"- [{m['title']}]({m['link']}) — `{m['domain']}` "
                            f"(similarity {m['similarity']:.2f}) {badge}"
                        )
                else:
                    st.write(
                        "No corroborating sources found via DuckDuckGo / Google News RSS. "
                        "This could mean the story is unverified, very recent, or niche - "
                        "or that search access is currently unavailable."
                    )

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
