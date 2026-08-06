import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import pickle
import unicodedata
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import cv2
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

try:
    from sentence_transformers import CrossEncoder
    HAS_NLI = True
except ImportError:
    HAS_NLI = False

st.set_page_config(
    page_title="VeriFact AI — Misinformation Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Command Center Glassmorphism Styling
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Card Container */
    .command-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Status Badges */
    .badge-real {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .badge-fake {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Metric Gauge Box */
    .metric-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# --- Persistent feedback store & real (but advisory-only) model training ---
# IMPORTANT: everything in this block is READ-ONLY with respect to the main
# verdict logic in the Text Fact-Checker and Media Authenticator sections
# above/below. It never overwrites `verdict`, `truth_index`, `media_verdict`,
# `ai_score`, or `manipulation_score`. Its only job is to persist feedback to
# disk (so it survives app restarts, unlike the old in-memory-only list) and
# to train a real classifier on that feedback for display purposes, behind an
# explicit opt-in toggle. This guarantees the calibrated accuracy of the
# primary predictions is unaffected by this feature.

DATA_DIR = "verifact_data"
FEEDBACK_STORE_PATH = os.path.join(DATA_DIR, "feedback_store.jsonl")
TEXT_MODEL_PATH = os.path.join(DATA_DIR, "text_learned_model.pkl")
MEDIA_MODEL_PATH = os.path.join(DATA_DIR, "media_learned_model.pkl")

TEXT_FEATURE_KEYS = ['overlap_ratio', 'raw_max_sim', 'sensationalism_score', 'journalistic_score', 'debunk_flag', 'known_hoax_flag', 'nli_contradiction_score', 'is_factcheck_source']
MEDIA_FEATURE_KEYS = ['ai_score', 'manipulation_score', 'corroborated', 'ai_signature_found', 'is_video', 'ml_deepfake_score_raw']

# Minimum bar before the learned model is trusted to DRIVE the primary
# verdict instead of just being shown as an advisory note. Both conditions
# must hold: enough samples that the model has generalized rather than
# memorized, AND a genuine held-out accuracy (not an inflated train-set
# score) above this bar. Until then, the rule-based engine stays primary -
# this is what prevents an undertrained model from silently making the app
# less accurate the moment feedback starts coming in.
MIN_SAMPLES_FOR_PRIMARY = 30
MIN_HELDOUT_ACCURACY_FOR_PRIMARY = 75.0

def load_feedback_store():
    """Load persisted feedback from disk. Returns [] if no store exists yet."""
    if not os.path.exists(FEEDBACK_STORE_PATH):
        return []
    entries = []
    try:
        with open(FEEDBACK_STORE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return entries

def append_feedback_entry(entry):
    """Append one feedback entry to the on-disk store. Best-effort - if the
    filesystem isn't writable (e.g. read-only deployment), this fails
    silently and feedback still lives in session_state for the current
    session, matching the old behavior."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(FEEDBACK_STORE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except Exception:
        return False

def train_model_from_feedback(entries, feature_keys, min_samples=8):
    """
    Trains a real LogisticRegression on stored feedback features vs the
    user-corrected ground-truth label. Returns a result dict with the model
    (or None), sample count, class distribution, and an honest accuracy
    estimate - held-out train/test split if there's enough data for one to
    be meaningful, otherwise a plainly-labeled train-set-only accuracy so we
    never overstate confidence on a handful of samples.
    """
    usable = [e for e in entries if e.get('features') and all(k in e['features'] for k in feature_keys) and e.get('corrected_label')]
    n = len(usable)
    labels = [e['corrected_label'] for e in usable]
    n_classes = len(set(labels))

    result = {
        'model': None, 'n_samples': n, 'n_classes': n_classes,
        'accuracy': None, 'accuracy_type': None, 'message': None, 'class_counts': None
    }

    if n < min_samples:
        result['message'] = f"Only {n} labeled sample(s) so far - need at least {min_samples} before training is meaningful."
        return result
    if n_classes < 2:
        result['message'] = f"All {n} samples share the same corrected label - need at least 2 different labels to train a classifier."
        return result

    X = np.array([[float(e['features'][k]) for k in feature_keys] for e in usable])
    y = np.array(labels)

    from collections import Counter
    result['class_counts'] = dict(Counter(labels))

    try:
        if n >= 20:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if min(Counter(y).values()) >= 2 else None)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            # Refit on all data for the deployed model, but report the held-out score
            model.fit(X, y)
            result['accuracy'] = round(acc * 100, 1)
            result['accuracy_type'] = 'held-out test split'
        else:
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            acc = model.score(X, y)
            result['accuracy'] = round(acc * 100, 1)
            result['accuracy_type'] = 'train-set only (too few samples for a held-out split - likely optimistic)'
        result['model'] = model
    except Exception as e:
        result['message'] = f"Training failed: {e}"

    return result

def save_model(model, path, meta=None):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': model, 'meta': meta or {}}, f)
        return True
    except Exception:
        return False

def load_model(path):
    """Returns (model, meta) tuple, or (None, {}) if unavailable/incompatible."""
    if not os.path.exists(path):
        return None, {}
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and 'model' in obj:
            return obj['model'], obj.get('meta', {})
        # Backward-compat: older files stored the bare model with no meta
        return obj, {}
    except Exception:
        return None, {}

def pick_primary_verdict(rule_verdict, rule_truth_index, feature_vector, feature_keys, learned_model, model_meta):
    """
    The hybrid switch. Returns (final_verdict, final_truth_index, source_label,
    learned_pred, learned_confidence) where source_label is 'rule-based' or
    'learned-model'.

    The learned model only becomes primary once it has EARNED it: enough
    training samples AND a genuine held-out accuracy above the bar. This is
    what makes "replace the rule-based system" safe to do automatically -
    early on (little/no feedback), the rule-based verdict always wins by
    default, so accuracy never regresses versus what you have today. It only
    switches over once there's real evidence the learned model outperforms
    manual thresholds.
    """
    learned_pred, learned_confidence = None, None
    if learned_model is not None and feature_vector is not None:
        try:
            X = np.array([[float(feature_vector[k]) for k in feature_keys]])
            learned_pred = learned_model.predict(X)[0]
            learned_confidence = float(max(learned_model.predict_proba(X)[0]))
        except Exception:
            learned_pred, learned_confidence = None, None

    n_samples = model_meta.get('n_samples', 0)
    accuracy = model_meta.get('accuracy', 0) or 0
    accuracy_type = model_meta.get('accuracy_type', '')
    earned_primary = (
        learned_pred is not None
        and n_samples >= MIN_SAMPLES_FOR_PRIMARY
        and accuracy_type == 'held-out test split'
        and accuracy >= MIN_HELDOUT_ACCURACY_FOR_PRIMARY
    )

    if earned_primary:
        # Reuse the rule-based truth_index scale isn't meaningful for the
        # learned model's own probability, so derive an equivalent index from
        # its confidence instead.
        final_truth_index = int(learned_confidence * 100) if "REAL" in str(learned_pred) or "VERIFIED" in str(learned_pred) else int(100 - learned_confidence * 100)
        return learned_pred, final_truth_index, "learned-model", learned_pred, learned_confidence

    return rule_verdict, rule_truth_index, "rule-based", learned_pred, learned_confidence

if 'verification_history' not in st.session_state:
    st.session_state.verification_history = []

if 'feedback_dataset' not in st.session_state:
    _persisted = load_feedback_store()
    if _persisted:
        st.session_state.feedback_dataset = _persisted
    else:
        st.session_state.feedback_dataset = [
            {
                'timestamp': '2026-03-01 10:15:00',
                'type': 'Text Claim',
                'content': 'RBI replacing all currency notes with plastic notes',
                'predicted_verdict': '🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT',
                'is_correct': 'Yes 👍',
                'corrected_label': '🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT',
                'features': None
            },
            {
                'timestamp': '2026-03-02 14:22:10',
                'type': 'Text Claim',
                'content': 'ISRO Gaganyaan engine testing completed',
                'predicted_verdict': '🟢 VERIFIED REAL / HIGHLY LIKELY',
                'is_correct': 'Yes 👍',
                'corrected_label': '🟢 VERIFIED REAL / HIGHLY LIKELY',
                'features': None
            }
        ]

if 'last_analyzed_claim' not in st.session_state:
    st.session_state.last_analyzed_claim = None

if 'text_learned_model' not in st.session_state:
    st.session_state.text_learned_model, st.session_state.text_model_meta = load_model(TEXT_MODEL_PATH)
if 'text_model_meta' not in st.session_state:
    st.session_state.text_model_meta = {}

if 'media_learned_model' not in st.session_state:
    st.session_state.media_learned_model, st.session_state.media_model_meta = load_model(MEDIA_MODEL_PATH)
if 'media_model_meta' not in st.session_state:
    st.session_state.media_model_meta = {}

if 'show_learned_insights' not in st.session_state:
    st.session_state.show_learned_insights = False

TIER1_SOURCES = [
    "pib", "reuters", "bbc", "the hindu", "indian express", "ndtv", 
    "times of india", "altnews", "boomlive", "factly", "pib fact check",
    "isro", "nasa", "who", "rbi", "afp", "associated press"
]

def evaluate_source_authority(source_name):
    clean_name = source_name.lower().strip()
    for t1 in TIER1_SOURCES:
        if t1 in clean_name:
            return {"tier": 1, "tier_label": "Tier 1: High Trust (Verified Outlet)", "badge_color": "#34d399"}
    if any(agg in clean_name for agg in ["news", "daily", "post", "times", "today"]):
        return {"tier": 2, "tier_label": "Tier 2: General News Publisher", "badge_color": "#38bdf8"}
    return {"tier": 3, "tier_label": "Tier 3: Unverified / Social Source", "badge_color": "#fbbf24"}

# Light stopword set used ONLY for query building - deliberately does not
# strip short entity tokens like "RBI", "5G", "AI", "UN".
QUERY_STOP_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'in', 'to', 'for', 'of', 'with',
    'that', 'this', 'it', 'from', 'by', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
    'had', 'will', 'would', 'says', 'said', 'according', 'announced', 'new', 'news', 'breaking'
}

def unicode_words(text):
    """
    Groups consecutive Unicode Letter/Mark/Number characters into words.
    Used instead of an ASCII ([a-zA-Z0-9]) or plain \\w regex, both of which
    either ignore or incorrectly split non-Latin scripts - e.g. Devanagari
    (Hindi) and several other Indian scripts use combining vowel signs
    (Unicode category Mn/Mc) that a plain \\w-complement regex strips out,
    fragmenting every word at each vowel sign. Grouping by category keeps
    those attached to their base letter. For pure ASCII/English text this
    produces byte-identical tokenization to the previous regex-based
    approach (verified) - so this is a correctness fix for non-English text
    with zero behavior change for English.
    """
    words, current = [], []
    for ch in text:
        if unicodedata.category(ch)[0] in ('L', 'M', 'N'):
            current.append(ch)
        else:
            if current:
                words.append(''.join(current))
                current = []
    if current:
        words.append(''.join(current))
    return words

def extract_search_queries(text):
    """
    Builds search queries that preserve short but high-value entity tokens
    (acronyms like RBI/ISRO/WHO, alphanumeric tags like 5G/COVID19, and
    capitalized proper nouns) which a plain word-length filter would drop.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 10]
    lead_sentence = sentences[0] if sentences else text

    raw_tokens = unicode_words(text)
    entities = []
    general_words = []
    for w in raw_tokens:
        if not w:
            continue
        lw = w.lower()
        if lw in QUERY_STOP_WORDS:
            continue
        # Entity-like token: all-caps acronym (2+ chars), contains a digit
        # (5G, COVID19), or a capitalized word of reasonable length. (Only
        # meaningful for cased scripts like Latin - non-Latin scripts like
        # Devanagari have no case, so they fall through to general_words,
        # which is the correct graceful degradation.)
        if (w.isupper() and len(w) >= 2) or re.search(r'\d', w) or (w[0].isupper() and len(w) >= 3):
            entities.append(w)
        elif len(lw) > 3:
            general_words.append(lw)

    entities = list(dict.fromkeys(entities))
    general_words = list(dict.fromkeys(general_words))

    # Query 1: entity-first, fill remaining slots with general keywords
    remaining_slots = max(0, 5 - len(entities[:4]))
    q1_terms = entities[:4] + general_words[:remaining_slots]
    q1 = " ".join(q1_terms) if q1_terms else text[:60]

    # Query 2: lead sentence, slightly wider window than before
    q2 = " ".join(lead_sentence.split()[:8])

    return [q1, q2]

# Language options for the Text Fact-Checker's search step. "English" maps
# to the exact hl/gl/ceid values that were previously hardcoded, so leaving
# the selector on its default produces byte-identical search queries to
# before this feature existed - existing accuracy is unaffected unless the
# user actively picks a different language.
LANGUAGE_OPTIONS = {
    'English': {'hl': 'en-IN', 'gl': 'IN', 'ceid': 'IN:en'},
    'Hindi': {'hl': 'hi-IN', 'gl': 'IN', 'ceid': 'IN:hi'},
    'Tamil': {'hl': 'ta-IN', 'gl': 'IN', 'ceid': 'IN:ta'},
    'Telugu': {'hl': 'te-IN', 'gl': 'IN', 'ceid': 'IN:te'},
    'Bengali': {'hl': 'bn-IN', 'gl': 'IN', 'ceid': 'IN:bn'},
    'Marathi': {'hl': 'mr-IN', 'gl': 'IN', 'ceid': 'IN:mr'},
    'Kannada': {'hl': 'kn-IN', 'gl': 'IN', 'ceid': 'IN:kn'},
    'Gujarati': {'hl': 'gu-IN', 'gl': 'IN', 'ceid': 'IN:gu'},
    'Malayalam': {'hl': 'ml-IN', 'gl': 'IN', 'ceid': 'IN:ml'},
    'Punjabi': {'hl': 'pa-IN', 'gl': 'IN', 'ceid': 'IN:pa'},
}

def fetch_google_news_rss(query, lang='English'):
    try:
        lang_params = LANGUAGE_OPTIONS.get(lang, LANGUAGE_OPTIONS['English'])
        encoded_q = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl={lang_params['hl']}&gl={lang_params['gl']}&ceid={lang_params['ceid']}"
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        items = []
        for item in root.findall('.//item')[:8]:
            title = item.find('title').text if item.find('title') is not None else ''
            link = item.find('link').text if item.find('link') is not None else ''
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
            source_el = item.find('source')
            source = source_el.text if source_el is not None else 'Google News'
            
            items.append({
                'title': title,
                'snippet': title,
                'link': link,
                'source': source,
                'date': pub_date
            })
        return items
    except Exception:
        return []

def fetch_duckduckgo_news(query):
    if not HAS_DDG:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=8))
            items = []
            for r in results:
                items.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('body', r.get('title', '')),
                    'link': r.get('url', ''),
                    'source': r.get('source', 'DuckDuckGo News'),
                    'date': r.get('date', '')
                })
            return items
    except Exception:
        return []

def fetch_live_news_with_fallback(query, lang='English'):
    articles = fetch_google_news_rss(query, lang=lang)
    if not articles:
        # DuckDuckGo fallback has no reliable per-language region parameter
        # for this search type, so it stays English/region-agnostic as a
        # last resort regardless of `lang` - unchanged from before.
        articles = fetch_duckduckgo_news(query)
    return articles

# --- Dedicated fact-check RSS feeds -----------------------------------------
# General news search (Google News / DuckDuckGo above) often has nothing at
# all for old or niche rumors. Dedicated fact-check outlets are much more
# likely to have directly addressed exactly this claim - and their articles
# are, by construction, debunk-relevant. These are public RSS feeds: no API
# key, no request quota.
FACTCHECK_RSS_FEEDS = [
    ("PIB Fact Check", "https://pib.gov.in/PressReleseDetail.aspx?rss=1"),
    ("AltNews", "https://www.altnews.in/feed/"),
    ("BOOM Live", "https://www.boomlive.in/feed"),
    ("Factly", "https://factly.in/feed/"),
]

def fetch_factcheck_rss(query):
    """
    Pulls from dedicated fact-check RSS feeds and keeps only items whose
    title/description overlap with the query terms, since these feeds can't
    be queried directly (they're just their latest-posts feed, not a search
    endpoint) - so we fetch the recent feed and filter locally.
    """
    query_terms = [t.lower() for t in re.findall(r'[a-zA-Z0-9]+', query) if len(t) > 2]
    if not query_terms:
        return []

    results = []
    for source_name, feed_url in FACTCHECK_RSS_FEEDS:
        try:
            req = urllib.request.Request(feed_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            for item in root.findall('.//item')[:20]:
                title_el = item.find('title')
                desc_el = item.find('description')
                link_el = item.find('link')
                date_el = item.find('pubDate')
                title = title_el.text if title_el is not None else ''
                desc = desc_el.text if desc_el is not None else ''
                combined_lower = (title + ' ' + (desc or '')).lower()
                # crude relevance filter: at least 2 query terms present, or 1 for short queries
                hits = sum(1 for t in query_terms if t in combined_lower)
                min_hits_needed = 1 if len(query_terms) <= 2 else 2
                if hits >= min_hits_needed:
                    results.append({
                        'title': title,
                        'snippet': re.sub('<[^<]+?>', '', desc or title),
                        'link': link_el.text if link_el is not None else '',
                        'source': source_name,
                        'date': date_el.text if date_el is not None else '',
                        'source_type': 'factcheck'
                    })
        except Exception:
            continue
    return results

def fetch_all_sources(query, lang='English'):
    """Combines general news (language-aware) + dedicated fact-check feeds
    (English-only sources, unaffected by `lang`) for one query."""
    general = fetch_live_news_with_fallback(query, lang=lang)
    factcheck = fetch_factcheck_rss(query)
    return general + factcheck

# --- NLI-based stance detection (optional, degrades gracefully) ------------
# Keyword matching ("denies", "hoax", etc.) misses debunks phrased any other
# way ("the central bank has no such plan", "officials clarified this claim
# is untrue"). A local NLI (Natural Language Inference) model checks whether
# an article's text actually CONTRADICTS the claim, not just whether it
# shares vocabulary with it.
#
# Uses cross-encoder/nli-deberta-v3-xsmall via sentence-transformers: a
# purpose-built NLI cross-encoder (premise, hypothesis) -> one forward pass,
# ~90MB, ~22M params - versus a general zero-shot pipeline, which needs a
# separate forward pass per candidate label (3x the compute here) and a much
# larger base model. Requires `sentence-transformers` (pulls in `torch`) and,
# once with network access, the model weights - see setup notes in chat. If
# unavailable, every function below returns None/False cleanly and the app
# falls back to the keyword-only check it already had, so behavior without
# this package installed is unchanged.

NLI_LABELS = ['contradiction', 'entailment', 'neutral']  # this model's fixed output order

@st.cache_resource(show_spinner="Loading local NLI model (one-time)...")
def get_nli_model():
    if not HAS_NLI:
        return None
    try:
        return CrossEncoder('cross-encoder/nli-deberta-v3-xsmall')
    except Exception:
        return None

def compute_nli_contradiction_score(claim_text, article_snippet):
    """
    Returns a 0-1 contradiction probability, or None if NLI isn't available
    or the call fails for any reason. Never raises.
    """
    if not article_snippet or not article_snippet.strip():
        return None
    model = get_nli_model()
    if model is None:
        return None
    try:
        # (premise, hypothesis): does the article snippet contradict the claim?
        logits = model.predict([(article_snippet[:512], claim_text[:300])])
        # Single softmax over the 3 raw logits -> probabilities
        row = np.asarray(logits[0], dtype=np.float64)
        exp = np.exp(row - row.max())
        probs = exp / exp.sum()
        contradiction_idx = NLI_LABELS.index('contradiction')
        return float(probs[contradiction_idx])
    except Exception:
        return None

def get_debunk_assessment(claim_text, article):
    """
    Combines the fast keyword check with the (optional) NLI contradiction
    score. Returns (debunk_flag: bool, nli_score: float|None, method: str).
    - If NLI is available and confident (>=0.55), it can flag a debunk on
      its own even without keyword hits - this is what catches denials
      phrased outside the fixed keyword list.
    - Keyword hits still work standalone regardless of NLI availability, so
      behavior is unchanged when transformers/torch aren't installed.
    """
    if not article:
        return False, None, "none"
    keyword_hit = contains_debunk_signal(article)
    nli_score = compute_nli_contradiction_score(claim_text, article.get('snippet', ''))
    if nli_score is not None and nli_score >= 0.55:
        return True, nli_score, "nli"
    if keyword_hit:
        return True, nli_score, "keyword"
    return False, nli_score, "none"

def extract_main_words(text):
    """
    Extracts core nouns, proper nouns, numbers, and key content words from text,
    filtering out stop words and general filler words. Keeps short entity
    tokens (2+ chars) instead of requiring 3+ chars, so acronyms like "AI",
    "5G", "UN" survive.
    """
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'in', 'to', 'for', 'of', 'with',
        'that', 'this', 'it', 'from', 'by', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might',
        'must', 'about', 'above', 'below', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'just', 'now', 'says', 'said', 'according', 'announced', 'new', 'news', 'breaking'
    }
    words = unicode_words(text)
    main_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower not in stop_words and len(w_lower) >= 2:
            main_words.append(w_lower)
    return list(dict.fromkeys(main_words))

# Signals that indicate an article is DENYING/DEBUNKING a claim rather than
# confirming it. Plain word-overlap can't tell "RBI announces X" apart from
# "RBI denies X" - both share the same key nouns - so this catches that case
# explicitly instead of letting overlap alone decide the verdict.
DEBUNK_SIGNAL_WORDS = [
    'false', 'fake', 'hoax', 'debunk', 'debunked', 'myth', 'not true', 'rumor', 'rumour',
    'clarifies', 'clarification', 'denies', 'denied', 'misleading', 'fact check', 'fact-check',
    'no truth', 'baseless', 'untrue', 'fabricated', 'busts', 'pib fact'
]

def contains_debunk_signal(article):
    if not article:
        return False
    combined = (article.get('title', '') + ' ' + article.get('snippet', '')).lower()
    return any(sig in combined for sig in DEBUNK_SIGNAL_WORDS)

# Well-documented, recurring hoax patterns that keep resurfacing (WhatsApp
# forwards etc.) and get repeatedly fact-checked by PIB/AltNews/BOOM. Live
# news search alone is unreliable for these: the debunking articles are
# often old and don't surface in a fresh RSS query, while unrelated real
# news sharing the same entity names (e.g. "RBI") can accidentally score
# high word-overlap. Each entry requires ALL of `all`, AT LEAST ONE of
# `any`, and AT LEAST ONE of `context` to be present in the claim text.
KNOWN_HOAX_PATTERNS = [
    {'all': ['plastic'], 'any': ['currency', 'notes', 'banknote', 'banknotes'], 'context': ['rbi', 'reserve bank']},
    {'all': ['whatsapp'], 'any': ['charge', 'paid', 'fee', 'subscription'], 'context': ['whatsapp', 'message']},
    {'all': ['5g'], 'any': ['virus', 'covid', 'coronavirus'], 'context': ['spread', 'cause', 'link']},
    {'all': ['2000'], 'any': ['chip', 'gps', 'tracking', 'nano'], 'context': ['note', 'currency']},
]

def matches_known_hoax(text):
    lt = text.lower()
    for pattern in KNOWN_HOAX_PATTERNS:
        if all(k in lt for k in pattern['all']) and any(k in lt for k in pattern['any']) and any(k in lt for k in pattern['context']):
            return True
    return False

def calculate_entity_and_vector_match(claim_text, articles):
    """
    Evaluates both key noun/main word overlap in a single article
    and sentence-level TF-IDF vector similarity.
    """
    if not articles:
        return 0.0, 0.0, None, [], 0

    sentences = [s.strip() for s in re.split(r'[.!?]\s+', claim_text) if len(s.strip()) > 10]
    if not sentences:
        sentences = [claim_text]

    max_sim = 0.0
    max_overlap_ratio = 0.0
    best_match = articles[0]
    best_matched_words = []
    total_main_words_count = 0

    for sentence in sentences:
        main_words = extract_main_words(sentence)
        if not main_words:
            continue
        
        total_main_words_count = max(total_main_words_count, len(main_words))

        for article in articles:
            snippet_text = article['snippet'].lower()
            # Check how many main words/nouns from sentence appear in THIS single article
            matched_words = [w for w in main_words if w in snippet_text]
            overlap_ratio = len(matched_words) / len(main_words) if main_words else 0.0

            # Compute TF-IDF vector similarity for this sentence vs article snippet
            try:
                vectorizer = TfidfVectorizer(stop_words='english').fit_transform([sentence, article['snippet']])
                vectors = vectorizer.toarray()
                sim_score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
            except Exception:
                sim_score = 0.0

            # Weight overlap higher for matching headlines
            combined_score = (overlap_ratio * 0.7) + (sim_score * 0.3)
            best_combined = (max_overlap_ratio * 0.7) + (max_sim * 0.3)

            if combined_score > best_combined:
                max_overlap_ratio = overlap_ratio
                max_sim = sim_score
                best_match = article
                best_matched_words = matched_words

    return float(max_overlap_ratio), float(max_sim), best_match, best_matched_words, total_main_words_count

def analyze_linguistic_risk(text):
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl_count = text.count('!')
    
    clickbait_words = ['shocking', 'secret', 'urgent', 'banned', 'leaked', 'viral', 'miracle', 'unbelievable', 'exposed', 'overnight']
    sensational_hits = sum(1 for word in clickbait_words if word in text.lower())
    
    journalistic_phrases = ['official', 'ministry', 'spokesperson', 'according to', 'published', 'statement', 'announced', 'report']
    journalistic_hits = sum(1 for phrase in journalistic_phrases if phrase in text.lower())
    
    sensationalism_score = min(int((sensational_hits * 25) + (caps_ratio * 40) + (excl_count * 10)), 100)
    journalistic_score = min(int(journalistic_hits * 20), 100)
    
    return sensationalism_score, journalistic_score

# --- Media forensics -------------------------------------------------------
# These are genuine, file-content-based checks (not text matching). Some are
# lightweight heuristics (EXIF, ELA, frequency analysis); one (below) is an
# actual pretrained deepfake-detection classifier. All degrade gracefully if
# their dependency isn't installed - see the caveats surfaced in the UI, and
# in chat: even the strongest of these (the ML classifier) is evidence, not
# proof - a 2025 benchmark (Deepfake-Eval-2024) found open-source deepfake
# detectors lose roughly half their claimed accuracy on real in-the-wild
# content versus the curated academic sets they're usually scored on.

AI_GENERATOR_TAGS = [
    'stable diffusion', 'midjourney', 'dall-e', 'dalle', 'dall·e', 'firefly',
    'sora', 'runway', 'pika labs', 'comfyui', 'automatic1111', 'invokeai',
    'leonardo.ai', 'ideogram', 'flux', 'imagen', 'trainedalgorithmicmedia'
]

try:
    from transformers import pipeline as hf_image_pipeline
    HAS_DEEPFAKE_CLF = True
except ImportError:
    HAS_DEEPFAKE_CLF = False

@st.cache_resource(show_spinner="Loading local deepfake image classifier (one-time)...")
def get_deepfake_classifier():
    """
    ViT-based binary real/fake image classifier (prithivMLmods/Deep-Fake-
    Detector-v2-Model), loaded via the standard transformers image-
    classification pipeline - no custom architecture code needed. ~330MB
    download, one-time. Returns None (never raises) if transformers isn't
    installed or the model fails to load, so the app degrades to the
    heuristic-only signals below.
    """
    if not HAS_DEEPFAKE_CLF:
        return None
    try:
        return hf_image_pipeline('image-classification', model="prithivMLmods/Deep-Fake-Detector-v2-Model")
    except Exception:
        return None

def compute_ml_deepfake_score(pil_image):
    """Returns a 0-1 'fake' probability from the pretrained classifier, or
    None if unavailable. Never raises."""
    clf = get_deepfake_classifier()
    if clf is None:
        return None
    try:
        results = clf(pil_image)
        for r in results:
            if 'fake' in r.get('label', '').lower() or 'deepfake' in r.get('label', '').lower():
                return float(r['score'])
        return None
    except Exception:
        return None

# --- OCR (screenshot text extraction) --------------------------------------
# A large share of real-world misinformation isn't an AI-generated photo -
# it's a SCREENSHOT of a fake tweet/article/notice. The forensic checks above
# only look at pixel-level authenticity; they never read what the image
# actually says. This extracts that text via Tesseract OCR (local, free) so
# it can optionally be sent to the Text Fact-Checker's existing, unchanged
# search/verdict pipeline. Purely additive: this function is never called by
# analyze_image_forensics and never touches ai_score/manipulation_score, so
# it cannot affect the existing media verdict logic. Requires `pytesseract`
# (pip) + the `tesseract-ocr` system binary (apt) - degrades to returning
# None if either is missing.

try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

def extract_text_from_image(pil_image, lang='eng'):
    """
    Returns extracted text (str) or None if OCR is unavailable, the image
    has no significant readable text, or extraction fails for any reason.
    Never raises. `lang` uses Tesseract language codes (e.g. 'eng', 'hin',
    'eng+hin' for mixed-language screenshots) - only works for languages
    whose Tesseract data pack is installed; falls back to no text found
    rather than erroring if a pack is missing.
    """
    if not HAS_OCR:
        return None
    try:
        text = pytesseract.image_to_string(pil_image, lang=lang)
        text = text.strip()
        # Filter out noise: OCR on non-text images often returns a handful
        # of garbage characters - require a minimum length to count as a
        # real find.
        if len(text) < 15:
            return None
        return text
    except Exception:
        return None

def analyze_image_forensics(file_bytes):
    """
    Real, file-content-based signals for images:
      1. EXIF inspection - camera-captured photos almost always carry Make/
         Model/DateTimeOriginal tags; AI generators and screenshots usually
         don't. Absence is a weak signal on its own (screenshots also lack
         it), so it's only ever combined with other signals, never decisive
         alone.
      2. AI-generator metadata signature - some tools (Stable Diffusion
         front-ends, Bing/DALL-E, Adobe Firefly's C2PA "Content Credentials")
         embed their name directly in EXIF/PNG text/XMP. If found, this is a
         strong, near-definitive signal.
      3. Error Level Analysis (ELA) - re-compress the image at a fixed JPEG
         quality and measure the difference from the original. Regions that
         were edited/composited re-compress differently than the rest of the
         image, showing up as localized high-error patches. This is a
         standard, real forensic technique (not something invented for this
         app) - though it mainly catches splicing/editing, not high-quality
         pure AI generation, which can be internally consistent.
      4. Pretrained ML deepfake classifier - see compute_ml_deepfake_score.
         The strongest single signal here when available, but still an
         estimate, not proof - see the module-level caveat above.
    Returns a dict of scores/flags; never raises - falls back to neutral
    values with a note if the file can't be parsed.
    """
    from PIL import Image, ExifTags
    import io

    result = {
        'has_camera_exif': False,
        'ai_signature_found': None,
        'ela_score': None,
        'ml_deepfake_score': None,
        'parse_error': None,
    }

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()

        # 1. EXIF / metadata inspection
        exif_data = {}
        try:
            raw_exif = img.getexif()
            if raw_exif:
                exif_data = {ExifTags.TAGS.get(k, k): v for k, v in raw_exif.items()}
        except Exception:
            pass

        camera_tags_present = any(t in exif_data for t in ('Make', 'Model', 'DateTimeOriginal', 'GPSInfo'))
        result['has_camera_exif'] = camera_tags_present

        # Check EXIF Software tag + PNG text chunks + any string metadata for
        # known AI generator signatures
        blob_to_check = " ".join(str(v) for v in exif_data.values())
        blob_to_check += " ".join(f"{k} {v}" for k, v in (img.info or {}).items() if isinstance(v, str))
        blob_to_check = blob_to_check.lower()
        found_tag = next((tag for tag in AI_GENERATOR_TAGS if tag in blob_to_check), None)
        result['ai_signature_found'] = found_tag

        # 2. Error Level Analysis
        rgb_img = img.convert('RGB')
        buf = io.BytesIO()
        rgb_img.save(buf, 'JPEG', quality=90)
        buf.seek(0)
        resaved = Image.open(buf)

        orig_arr = np.asarray(rgb_img).astype(np.int16)
        resaved_arr = np.asarray(resaved).astype(np.int16)
        diff = np.abs(orig_arr - resaved_arr)

        mean_error = float(diff.mean())
        max_error = float(diff.max())
        high_error_pixel_ratio = float((diff.max(axis=2) > 25).mean())

        # Normalize into a 0-100 "manipulation likelihood" score. High mean
        # error + a large share of localized high-error pixels indicates
        # inconsistent recompression history typical of spliced/edited images.
        ela_score = min(int((mean_error * 6) + (high_error_pixel_ratio * 100 * 0.6)), 100)
        result['ela_score'] = ela_score
        result['ela_mean_error'] = round(mean_error, 2)
        result['ela_high_error_ratio'] = round(high_error_pixel_ratio, 3)

        # 3. Pretrained ML deepfake classifier (optional dependency)
        ml_score = compute_ml_deepfake_score(rgb_img)
        result['ml_deepfake_score'] = int(ml_score * 100) if ml_score is not None else None

    except Exception as e:
        result['parse_error'] = str(e)

    return result

def analyze_video_forensics(file_bytes, filename_hint="upload.mp4"):
    """
    Real, file-content-based signals for video:
      1. Container/stream metadata via ffprobe - checks the encoder tag and
         format tags. Most AI video generators either strip metadata
         entirely or leave a generic encoder string; this is a weak signal,
         reported for transparency rather than treated as decisive.
      2. Frame sharpness-consistency heuristic via OpenCV - samples frames
         evenly across the clip and computes each frame's Laplacian
         variance (a focus/sharpness measure). Real camera footage usually
         has smoothly varying sharpness across frames; some generated or
         heavily edited clips show abrupt jumps. This is a coarse proxy,
         not a trained deepfake classifier - it will miss high-quality
         modern generative video and can false-positive on legitimately
         shaky or low-light footage.
      3. Pretrained ML deepfake classifier - same image classifier as
         images, applied to a SMALLER frame sample (max 5, vs 12 for the
         sharpness check) since it's much more compute-heavy per frame -
         this keeps per-upload latency reasonable. Scores are averaged
         across sampled frames.
    Returns a dict of scores/flags; never raises.
    """
    import subprocess
    import tempfile
    import os
    import json as _json
    from PIL import Image as PILImage

    result = {
        'encoder_tag': None,
        'frame_sharpness_std': None,
        'frame_count_sampled': 0,
        'ml_deepfake_score': None,
        'ml_frames_sampled': 0,
        'parse_error': None,
    }

    tmp_path = None
    try:
        suffix = os.path.splitext(filename_hint)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # 1. ffprobe metadata
        try:
            probe = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', tmp_path],
                capture_output=True, text=True, timeout=15
            )
            if probe.returncode == 0:
                meta = _json.loads(probe.stdout)
                fmt_tags = meta.get('format', {}).get('tags', {})
                result['encoder_tag'] = fmt_tags.get('encoder') or fmt_tags.get('software')
        except Exception:
            pass

        # 2. Frame sharpness consistency (OpenCV)
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        sample_count = min(12, total_frames) if total_frames > 0 else 0
        sharpness_values = []
        sampled_frames_bgr = []

        if sample_count > 1:
            indices = np.linspace(0, total_frames - 1, sample_count).astype(int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_values.append(lap_var)
                sampled_frames_bgr.append(frame)
        cap.release()

        if len(sharpness_values) > 1:
            mean_s = float(np.mean(sharpness_values))
            std_s = float(np.std(sharpness_values))
            # Coefficient of variation - normalized so it's comparable across
            # differently-exposed clips
            result['frame_sharpness_std'] = round(std_s / mean_s, 3) if mean_s > 0 else None
            result['frame_count_sampled'] = len(sharpness_values)

        # 3. Pretrained ML deepfake classifier on a smaller frame subsample
        if sampled_frames_bgr and HAS_DEEPFAKE_CLF:
            ml_sample_indices = np.linspace(0, len(sampled_frames_bgr) - 1, min(5, len(sampled_frames_bgr))).astype(int)
            ml_scores = []
            for i in ml_sample_indices:
                try:
                    rgb_frame = cv2.cvtColor(sampled_frames_bgr[int(i)], cv2.COLOR_BGR2RGB)
                    pil_frame = PILImage.fromarray(rgb_frame)
                    s = compute_ml_deepfake_score(pil_frame)
                    if s is not None:
                        ml_scores.append(s)
                except Exception:
                    continue
            if ml_scores:
                result['ml_deepfake_score'] = int(np.mean(ml_scores) * 100)
                result['ml_frames_sampled'] = len(ml_scores)

    except Exception as e:
        result['parse_error'] = str(e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return result

with st.sidebar:
    st.markdown("<h2 style='color:#34d399; margin-bottom:0;'>🛡️ VeriFact AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.8rem;'>Misinformation Command Center</p>", unsafe_allow_html=True)
    st.divider()
    
    analysis_mode = st.radio(
        "Select Pipeline Mode:",
        ["📰 Text / Article Fact-Checker", "📷 Image & Video Authenticator", "🧠 Model Feedback & Active Learning"],
        index=0,
        key="analysis_mode_radio"
    )
    
    st.divider()
    st.markdown("### ⚡ Test Claim Benchmarks")
    if st.button("🔴 Fake: Plastic Currency Rumor"):
        st.session_state.test_claim = "The Reserve Bank of India has announced that all paper currency notes will be replaced with plastic bank notes next month."
    if st.button("🟢 Real: ISRO Gaganyaan Engine"):
        st.session_state.test_claim = "ISRO successfully completed core stage engine testing for the Gaganyaan human spaceflight mission."
    if st.button("🚨 Clickbait: 5G Scalar Waves"):
        st.session_state.test_claim = "BREAKING URGENT: Secret government plot leaked as 5G cell towers emit scalar frequencies!"

    st.divider()
    st.caption(f"Active Feedback Memory: **{len(st.session_state.feedback_dataset)} Samples**")
    st.session_state.show_learned_insights = st.checkbox(
        "🧪 Show experimental learned-model insights",
        value=st.session_state.show_learned_insights,
        help="Displays what the feedback-trained model would predict, alongside the main verdict. Purely informational - it never changes the main verdict, truth index, or confidence scores above."
    )

st.markdown("<h1 style='color:#f8fafc; margin-bottom:5px;'>VeriFact AI Command Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>Real-Time Live Web Grounding & Media Authenticator Engine</p>", unsafe_allow_html=True)

if analysis_mode == "📰 Text / Article Fact-Checker":
    
    user_input = st.text_area(
        "Enter News Claim, Article Paragraph, or Viral Post:",
        value=st.session_state.get('test_claim', ''),
        height=140,
        placeholder="Paste headline or paragraph to verify..."
    )

    search_lang = st.selectbox(
        "Search language (searches live news in this language):",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,
        help="Defaults to English, matching prior behavior exactly. Pick another language to search Google News in that language instead - useful for claims in Hindi, Tamil, etc. that English-only search would miss."
    )
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        run_btn = st.button("🔍 Run Deep Fact Check", type="primary", use_container_width=True)
        
    if run_btn and user_input.strip():
        with st.spinner(f"Analyzing claim nouns/entities, querying live news ({search_lang}) + fact-check feeds..."):
            
            # Step 1: Query Extraction & Multi-Source Search (general news +
            # dedicated fact-check RSS feeds - see fetch_all_sources)
            queries = extract_search_queries(user_input)
            all_articles = []
            for q in queries:
                fetched = fetch_all_sources(q, lang=search_lang)
                all_articles.extend(fetched)
                
            # Deduplicate Articles
            seen = set()
            unique_articles = []
            for a in all_articles:
                if a['title'] not in seen:
                    seen.add(a['title'])
                    unique_articles.append(a)
                    
            # Step 2: Single-Article Noun/Main Word Overlap Engine
            overlap_ratio, raw_max_sim, best_match, matched_words, total_words = calculate_entity_and_vector_match(user_input, unique_articles)
            corroboration_pct = int(overlap_ratio * 100)
            
            # Step 3: Linguistic Risk Scanner
            sensationalism_score, journalistic_score = analyze_linguistic_risk(user_input)

            # Step 4: Debunk/Negation Signal Check
            # Word overlap alone can't tell "X announced" from "X denies" - both
            # share the same key nouns - so check the matched article's own
            # language for explicit debunk/denial signals before trusting overlap.
            # Combines the original keyword check with an optional local NLI
            # contradiction score (falls back to keyword-only if transformers/
            # torch aren't installed - see get_debunk_assessment).
            debunk_flag, nli_score, debunk_method = get_debunk_assessment(user_input, best_match)
            is_factcheck_source = bool(best_match and best_match.get('source_type') == 'factcheck')

            # Step 4b: Known Recurring Hoax Check
            # Some claims are well-documented, repeatedly fact-checked hoaxes
            # that live search can't reliably catch (old debunk articles don't
            # surface; unrelated real news with the same entity names can
            # falsely inflate word overlap). These take priority over the
            # search-based signals below.
            known_hoax_flag = matches_known_hoax(user_input)
            
            # Step 5: Re-calibrated Decision Matrix (UNCHANGED from the
            # previously-tuned version - this is still what runs by default
            # for every claim, regardless of whether a learned model exists)
            if known_hoax_flag:
                verdict = "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT"
                status_class = "badge-fake"
                truth_index = 5
                summary = "This matches a well-documented, recurring misinformation pattern that has been repeatedly fact-checked and debunked by official sources (e.g. PIB Fact Check)."
            elif debunk_flag and overlap_ratio >= 0.20:
                verdict = "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT"
                status_class = "badge-fake"
                truth_index = max(100 - int(overlap_ratio * 100) - 20, 5)
                summary = f"A matching report from '{best_match['source'] if best_match else 'a news source'}' explicitly identifies this claim as false, denied, or debunked" + (f" (detected via {debunk_method})." if debunk_method != "none" else ".")
            elif overlap_ratio >= 0.35 or (overlap_ratio >= 0.20 and raw_max_sim >= 0.15) or (overlap_ratio >= 0.22 and journalistic_score >= 20):
                verdict = "🟢 VERIFIED REAL / HIGHLY LIKELY"
                status_class = "badge-real"
                truth_index = min(int(max(overlap_ratio, raw_max_sim) * 100 + 40), 98)
                summary = f"Matches live coverage from '{best_match['source'] if best_match else 'Global News'}'. Key claim nouns ({len(matched_words)} matched) confirmed in live news reports."
            elif sensationalism_score >= 40 and overlap_ratio < 0.25:
                verdict = "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT"
                status_class = "badge-fake"
                truth_index = max(100 - sensationalism_score, 10)
                summary = "Exhibits heavy clickbait language and key claim nouns failed to match together in verified news reports."
            else:
                verdict = "⚠️ UNVERIFIED / PROBABLE FAKE NEWS"
                status_class = "badge-warning"
                truth_index = 35
                summary = f"Key nouns/main terms were not found together in any single verified live news report ({len(matched_words)}/{total_words} words matched)."

            # Step 6: Hybrid switch. Rule-based verdict above is ALWAYS
            # computed. It only gets replaced as the displayed primary verdict
            # once the learned model has earned that trust (see
            # pick_primary_verdict) - so with no/little feedback yet, this is
            # a no-op and today's calibrated accuracy is exactly preserved.
            feature_vector = {
                'overlap_ratio': round(overlap_ratio, 4),
                'raw_max_sim': round(raw_max_sim, 4),
                'sensationalism_score': sensationalism_score,
                'journalistic_score': journalistic_score,
                'debunk_flag': int(debunk_flag),
                'known_hoax_flag': int(known_hoax_flag),
                'nli_contradiction_score': round(nli_score, 4) if nli_score is not None else 0.0,
                'is_factcheck_source': int(is_factcheck_source),
            }
            final_verdict, final_truth_index, verdict_source, learned_pred, learned_confidence = pick_primary_verdict(
                verdict, truth_index, feature_vector, TEXT_FEATURE_KEYS,
                st.session_state.text_learned_model, st.session_state.text_model_meta
            )
            if verdict_source == "learned-model":
                status_class = "badge-real" if ("REAL" in str(final_verdict) or "VERIFIED" in str(final_verdict)) else ("badge-fake" if "FAKE" in str(final_verdict) or "DEBUNKED" in str(final_verdict) else "badge-warning")
                summary = f"Predicted by the feedback-trained model ({model_confidence_pct:=round(learned_confidence*100)}% confidence, trained on {st.session_state.text_model_meta.get('n_samples')} samples, {st.session_state.text_model_meta.get('accuracy')}% held-out accuracy)."
                
            # Store Last Analyzed Claim for Active Learning Pipeline.
            # 'features' is captured so future feedback on THIS claim can be
            # used to (re)train the model - and 'rule_based_verdict' is kept
            # too so feedback always has ground truth for the ORIGINAL
            # rule-based prediction, even on turns where the learned model
            # was the one actually displayed.
            st.session_state.last_analyzed_claim = {
                'type': 'Text Claim',
                'content': user_input,
                'predicted_verdict': final_verdict,
                'rule_based_verdict': verdict,
                'verdict_source': verdict_source,
                'features': feature_vector
            }

            # Log Session History
            st.session_state.verification_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'claim': user_input[:60] + "...",
                'verdict': final_verdict,
                'truth_index': f"{final_truth_index}%",
                'corroboration': f"{corroboration_pct}%"
            })
            
            # Dashboard Verdict Header
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="command-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="{status_class}">{final_verdict}</span>
                        <h2 style="color:#ffffff; margin-top:12px; margin-bottom:4px;">Truth Index: {final_truth_index}%</h2>
                        <p style="color:#cbd5e1; font-size:0.95rem;">{summary}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            source_note = (
                f"🧠 Verdict source: **Learned model** (trained on {st.session_state.text_model_meta.get('n_samples')} samples, {st.session_state.text_model_meta.get('accuracy')}% held-out accuracy - it has earned primacy over the rule-based engine)."
                if verdict_source == "learned-model"
                else "⚙️ Verdict source: **Rule-based engine** (the learned model hasn't yet reached the sample/accuracy bar to take over - see the Feedback tab for progress)."
            )
            st.caption(source_note)
            if learned_pred is not None and verdict_source == "rule-based":
                agree = "✅ agrees" if learned_pred == verdict else "⚠️ disagrees"
                st.caption(f"For comparison, the learned model currently predicts **{learned_pred}** ({learned_confidence*100:.0f}% confidence) - {agree} with the rule-based verdict above.")
            
            # Metric Columns
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{final_truth_index}%</div><div class="metric-label">Truth Index</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{corroboration_pct}%</div><div class="metric-label">Single Article Match</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{journalistic_score}%</div><div class="metric-label">Journalistic Tone</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{sensationalism_score}%</div><div class="metric-label">Sensationalism Score</div></div>""", unsafe_allow_html=True)

            if st.session_state.show_learned_insights:
                with st.expander("🔬 Rule-based vs. learned-model detail"):
                    st.markdown(f"- **Rule-based verdict:** {verdict} (Truth Index {truth_index}%)")
                    if learned_pred is not None:
                        st.markdown(f"- **Learned-model prediction:** {learned_pred} ({learned_confidence*100:.0f}% confidence)")
                    else:
                        st.markdown("- **Learned-model prediction:** not available yet (train it in the Feedback tab).")
                    st.markdown(f"- **NLI contradiction score:** {f'{nli_score:.2f}' if nli_score is not None else 'N/A (sentence-transformers not installed, or no matched article)'}")
                    st.markdown(f"- **Fact-check source matched:** {'Yes' if is_factcheck_source else 'No'}")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📲 WhatsApp Debunk Card", "🟢 Live News & Source Authority", "📊 Analytics Details"])
            
            with tab1:
                st.markdown("#### Ready-to-Share WhatsApp Fact-Check Briefing")
                debunk_text = f"""*🛡️ VERIFACT AI FACT CHECK ALERT*
----------------------------------
*Claim:* "{user_input[:100]}..."
*Verdict:* {final_verdict}
*Truth Index:* {final_truth_index}%
*Single Article Word Overlap:* {corroboration_pct}%

*Summary:* {summary}
*Verified via VeriFact AI Command Center*"""
                st.code(debunk_text, language="markdown")
                
            with tab2:
                st.markdown("#### Top Matching News Articles Found")
                if unique_articles:
                    for art in unique_articles[:4]:
                        src_info = evaluate_source_authority(art['source'])
                        badge_color = src_info['badge_color']
                        tier_label = src_info['tier_label']
                        is_fc = art.get('source_type') == 'factcheck'
                        
                        st.markdown(f"""
                        <div style="background:rgba(15,23,42,0.6); padding:12px; border-radius:8px; margin-bottom:8px; border:1px solid rgba(255,255,255,0.05);">
                            <a href="{art['link']}" target="_blank" style="color:#38bdf8; font-weight:bold; text-decoration:none;">{art['title']}</a><br>
                            <span style="color:#94a3b8; font-size:0.8rem;">Source: {art['source']} | </span>
                            <span style="color:{badge_color}; font-weight:bold; font-size:0.8rem;">{tier_label}</span>
                            {' <span style="color:#a78bfa; font-weight:bold; font-size:0.8rem;"> | 🔍 Dedicated Fact-Check Source</span>' if is_fc else ''}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No direct corroborating headlines found on live news feeds.")
                    
            with tab3:
                st.json({
                    "final_verdict": final_verdict,
                    "verdict_source": verdict_source,
                    "rule_based_verdict": verdict,
                    "learned_model_prediction": learned_pred,
                    "single_article_word_overlap_ratio": overlap_ratio,
                    "matched_key_words": matched_words,
                    "total_key_words": total_words,
                    "vector_cosine_similarity": raw_max_sim,
                    "sensationalism_score": sensationalism_score,
                    "debunk_signal_detected": debunk_flag,
                    "debunk_detection_method": debunk_method,
                    "nli_contradiction_score": nli_score,
                    "nli_available": HAS_NLI,
                    "known_hoax_pattern_matched": known_hoax_flag,
                    "factcheck_source_matched": is_factcheck_source,
                    "extracted_queries": queries,
                    "duckduckgo_fallback_enabled": HAS_DDG
                })

elif analysis_mode == "📷 Image & Video Authenticator":
    st.markdown("### 📷 Image & Video Authenticator Engine")
    st.markdown("Upload a video (`.mp4`, `.mov`) or news screenshot (`.jpg`, `.png`) to evaluate authenticity against deepfake signals, synthetic audio, and live news grounding.")
    st.caption("⚠️ This combines lightweight forensic heuristics (EXIF metadata, Error Level Analysis, frame-sharpness consistency) with a pretrained deepfake-detection classifier where available. None of this is proof of authenticity or manipulation. A 2025 benchmark (Deepfake-Eval-2024) found open-source deepfake detectors lose roughly half their claimed accuracy on real in-the-wild content versus the curated datasets they're normally scored on - treat every result here as a lead for further checking, never a final verdict.")
    
    uploaded_media = st.file_uploader("Choose Video or Image File:", type=["mp4", "mov", "avi", "jpg", "jpeg", "png", "webp"])
    media_context = st.text_input("Associated Claim Context (Optional):", placeholder="e.g. 'Viral video claiming official statement announced today'")
    
    if uploaded_media is not None:
        file_type = uploaded_media.type
        is_video = "video" in file_type
        
        col_med1, col_med2 = st.columns([1, 1])
        with col_med1:
            if is_video:
                st.video(uploaded_media)
            else:
                st.image(uploaded_media, use_container_width=True)

                # OCR: read any text baked into the image (screenshots of fake
                # posts/notices/tweets are one of the most common real-world
                # formats this kind of misinformation actually takes). This
                # runs independently of, and never modifies, the forensic
                # pipeline/verdict below - purely additive.
                if HAS_OCR:
                    ocr_lang_choice = st.selectbox(
                        "OCR language (for text inside the image):",
                        options=["English", "Hindi", "English + Hindi"],
                        index=0,
                        key="ocr_lang_select",
                        help="Requires the matching Tesseract language pack on the server - defaults to English."
                    )
                    ocr_lang_map = {"English": "eng", "Hindi": "hin", "English + Hindi": "eng+hin"}
                    try:
                        from PIL import Image as _PILImage
                        import io as _io
                        _pil_img_for_ocr = _PILImage.open(_io.BytesIO(uploaded_media.getvalue()))
                        ocr_text = extract_text_from_image(_pil_img_for_ocr, lang=ocr_lang_map[ocr_lang_choice])
                    except Exception:
                        ocr_text = None

                    if ocr_text:
                        with st.expander("📝 Text detected in image (OCR)", expanded=True):
                            st.text_area("Extracted text:", value=ocr_text, height=100, key="ocr_extracted_text_display", disabled=True)
                            st.caption("This is raw OCR output and may contain errors - review before relying on it.")
                            if st.button("🔍 Send this text to the Fact-Checker"):
                                st.session_state.test_claim = ocr_text
                                st.session_state.analysis_mode_radio = "📰 Text / Article Fact-Checker"
                                st.rerun()
                else:
                    st.caption("💡 OCR not available on this deployment (pytesseract/tesseract-ocr not installed) - install to detect text baked into screenshots.")
                
        with col_med2:
            st.markdown("#### Media Verification Pipeline")
            if st.button("⚡ Authenticate Media File", type="primary"):
                with st.spinner("Analyzing file metadata, compression artifacts, and search grounding..."):

                    media_bytes = uploaded_media.getvalue()
                    filename = uploaded_media.name.lower()

                    # Text-context corroboration is now only ONE input among several,
                    # never the sole gate - so an upload with no caption still gets a
                    # real, file-based analysis instead of defaulting to "fake".
                    corroborated = False
                    if media_context.strip():
                        q = extract_search_queries(media_context)[0]
                        articles = fetch_live_news_with_fallback(q)
                        overlap, sim, _, _, _ = calculate_entity_and_vector_match(media_context, articles)
                        if overlap >= 0.50 or sim >= 0.15:
                            corroborated = True

                    synthetic_keywords = ["sora", "runway", "deepfake", "pika", "midjourney", "synth", "elevenlabs"]
                    has_synthetic_filename_tag = any(kw in filename or kw in media_context.lower() for kw in synthetic_keywords)

                    forensics = {}
                    ai_score = 40
                    manipulation_score = 40
                    forensic_notes = []

                    if is_video:
                        forensics = analyze_video_forensics(media_bytes, filename_hint=filename)
                        if forensics.get('parse_error'):
                            forensic_notes.append(f"Video could not be fully parsed for forensic analysis ({forensics['parse_error']}).")
                        else:
                            if forensics.get('encoder_tag'):
                                forensic_notes.append(f"Container encoder tag: {forensics['encoder_tag']}")
                            sharp_cv = forensics.get('frame_sharpness_std')
                            if sharp_cv is not None:
                                # Higher coefficient of variation in frame sharpness = more
                                # inconsistency across sampled frames.
                                manipulation_score = min(int(sharp_cv * 180), 90)
                                ai_score = manipulation_score
                                forensic_notes.append(f"Frame sharpness consistency variance: {sharp_cv} (sampled {forensics.get('frame_count_sampled', 0)} frames)")
                            else:
                                forensic_notes.append("Not enough frames could be sampled for a frame-consistency check.")
                            ml_score = forensics.get('ml_deepfake_score')
                            if ml_score is not None:
                                # Strongest single signal when available - takes over ai_score,
                                # but caveat prominently (see module-level note): even the best
                                # open-source deepfake detectors lose roughly half their claimed
                                # accuracy on real in-the-wild content per 2025 benchmarking.
                                ai_score = ml_score
                                forensic_notes.append(f"Pretrained deepfake classifier: {ml_score}% fake-probability (averaged over {forensics.get('ml_frames_sampled', 0)} sampled frames) - treat as evidence, not proof; real-world accuracy is meaningfully lower than academic benchmarks.")
                            elif not HAS_DEEPFAKE_CLF:
                                forensic_notes.append("Pretrained deepfake classifier not available (transformers not installed) - falling back to the frame-sharpness heuristic only.")
                    else:
                        forensics = analyze_image_forensics(media_bytes)
                        if forensics.get('parse_error'):
                            forensic_notes.append(f"Image could not be fully parsed for forensic analysis ({forensics['parse_error']}).")
                        else:
                            ela = forensics.get('ela_score')
                            if ela is not None:
                                manipulation_score = ela
                                forensic_notes.append(f"Error Level Analysis score: {ela}% (mean recompression error {forensics.get('ela_mean_error')}, high-error pixel ratio {forensics.get('ela_high_error_ratio')})")
                            if not forensics.get('has_camera_exif'):
                                forensic_notes.append("No camera EXIF metadata (Make/Model/GPS/timestamp) found - consistent with, but not proof of, AI generation or a screenshot.")
                            ai_score = 30 if forensics.get('has_camera_exif') else 55
                            ml_score = forensics.get('ml_deepfake_score')
                            if ml_score is not None:
                                ai_score = ml_score
                                forensic_notes.append(f"Pretrained deepfake classifier: {ml_score}% fake-probability - treat as evidence, not proof; real-world accuracy is meaningfully lower than academic benchmarks.")
                            elif not HAS_DEEPFAKE_CLF:
                                forensic_notes.append("Pretrained deepfake classifier not available (transformers not installed) - falling back to EXIF/ELA heuristics only.")

                    ai_signature_found = forensics.get('ai_signature_found') if not is_video else None
                    ml_score = forensics.get('ml_deepfake_score')

                    # --- Decision combination ---
                    # Priority order: explicit AI-generator metadata signature (strongest,
                    # near-definitive) > filename/context synthetic keyword hit > pretrained
                    # ML classifier's own confident read of the actual pixels > news
                    # corroboration > forensic heuristic scores alone.
                    if ai_signature_found:
                        media_verdict = "🚨 FAKE AI GENERATED VIDEO" if is_video else "🚨 FAKE AI GENERATED IMAGE"
                        badge_style = "badge-fake"
                        confidence = 95
                        summary_msg = f"Metadata explicitly identifies this file as generated by '{ai_signature_found}'."
                        ai_score = 95
                        manipulation_score = max(manipulation_score, 80)
                    elif has_synthetic_filename_tag:
                        media_verdict = "🚨 FAKE AI GENERATED VIDEO" if is_video else "🚨 FAKE AI GENERATED IMAGE"
                        badge_style = "badge-fake"
                        confidence = 80
                        summary_msg = "Filename or claim context references a known generative-AI tool."
                        ai_score = max(ai_score, 85)
                        manipulation_score = max(manipulation_score, 75)
                    elif ml_score is not None and ml_score >= 65:
                        media_verdict = "🚨 FAKE AI GENERATED VIDEO" if is_video else "🚨 FAKE AI GENERATED IMAGE"
                        badge_style = "badge-fake"
                        confidence = ml_score
                        summary_msg = f"The pretrained deepfake classifier flagged this as {ml_score}% likely AI-generated/manipulated - this is the strongest available signal here, but still an estimate (see the caveat above), not proof."
                        ai_score = ml_score
                    elif corroborated:
                        media_verdict = "🟢 REAL VIDEO" if is_video else "🟢 REAL IMAGE / GRAPHIC"
                        badge_style = "badge-real"
                        confidence = 85
                        summary_msg = "Corroborated by live news grounding feeds, and file-level forensic checks found no strong manipulation signal."
                        ai_score = min(ai_score, 20)
                        manipulation_score = min(manipulation_score, 25)
                    elif ml_score is not None and ml_score <= 20 and manipulation_score < 55:
                        media_verdict = "✅ NO MANIPULATION SIGNALS DETECTED"
                        badge_style = "badge-real"
                        confidence = 100 - ml_score
                        summary_msg = f"The pretrained deepfake classifier scored this only {ml_score}% likely fake, and forensic checks found no red flags. This is a positive signal, not independent confirmation of authenticity - it means nothing here looks wrong, not that the file's origin is verified."
                        ai_score = ml_score
                    elif manipulation_score >= 55:
                        media_verdict = "⚠️ SIGNS OF MANIPULATION DETECTED" 
                        badge_style = "badge-warning"
                        confidence = 60
                        summary_msg = "File-level forensic checks (" + "; ".join(forensic_notes[:2]) + ") flagged inconsistencies worth a closer manual look. This is not a confirmed fake."
                    else:
                        media_verdict = "⚠️ UNVERIFIED — NO STRONG SIGNAL EITHER WAY"
                        badge_style = "badge-warning"
                        confidence = 45
                        summary_msg = "No corroborating news coverage and no strong forensic red flags. Insufficient evidence to call this real or fake with confidence."

                    st.session_state.last_analyzed_claim = {
                        'type': 'Media File',
                        'content': media_context if media_context else filename,
                        'predicted_verdict': media_verdict,
                        'rule_based_verdict': media_verdict,
                        'features': {
                            'ai_score': int(ai_score),
                            'manipulation_score': int(manipulation_score),
                            'corroborated': int(corroborated),
                            'ai_signature_found': int(bool(ai_signature_found)),
                            'is_video': int(is_video),
                            'ml_deepfake_score_raw': int(forensics.get('ml_deepfake_score')) if forensics.get('ml_deepfake_score') is not None else -1,
                        }
                    }

                    # Hybrid switch (same mechanism as the text checker): the
                    # learned model only becomes primary once it has enough
                    # samples AND genuine held-out accuracy - see
                    # pick_primary_verdict. With no/little media feedback yet,
                    # this is a no-op and the forensic-based verdict above is
                    # exactly what gets shown, unchanged.
                    media_feature_vector = st.session_state.last_analyzed_claim['features']
                    final_media_verdict, final_media_confidence, media_verdict_source, media_learned_pred, media_learned_confidence = pick_primary_verdict(
                        media_verdict, confidence, media_feature_vector, MEDIA_FEATURE_KEYS,
                        st.session_state.media_learned_model, st.session_state.media_model_meta
                    )
                    if media_verdict_source == "learned-model":
                        badge_style = "badge-real" if "REAL" in str(final_media_verdict) else ("badge-fake" if "FAKE" in str(final_media_verdict) else "badge-warning")
                        summary_msg = f"Predicted by the feedback-trained media model ({round(media_learned_confidence*100)}% confidence, trained on {st.session_state.media_model_meta.get('n_samples')} samples, {st.session_state.media_model_meta.get('accuracy')}% held-out accuracy)."
                        media_verdict = final_media_verdict
                        confidence = final_media_confidence
                    st.session_state.last_analyzed_claim['predicted_verdict'] = media_verdict
                    st.session_state.last_analyzed_claim['verdict_source'] = media_verdict_source

                    st.session_state.verification_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'claim': f"[{'Video' if is_video else 'Image'}] " + (media_context[:40] if media_context else filename),
                        'verdict': media_verdict,
                        'truth_index': f"{100 - ai_score}%",
                        'corroboration': "Verified" if corroborated else "Unverified"
                    })

                    st.markdown(f"""
                    <div class="command-card">
                        <span class="{badge_style}">{media_verdict}</span>
                        <h3 style="color:#ffffff; margin-top:12px;">Confidence Rating: {confidence}%</h3>
                        <p style="color:#cbd5e1; font-size:0.9rem;">{summary_msg}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    media_source_note = (
                        f"🧠 Verdict source: **Learned model** (trained on {st.session_state.media_model_meta.get('n_samples')} samples, {st.session_state.media_model_meta.get('accuracy')}% held-out accuracy)."
                        if media_verdict_source == "learned-model"
                        else "⚙️ Verdict source: **Forensic rule-based engine** (learned model hasn't yet reached the sample/accuracy bar to take over)."
                    )
                    st.caption(media_source_note)

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""<div class="metric-box"><div class="metric-value">{ai_score}%</div><div class="metric-label">AI Generation Probability</div></div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class="metric-box"><div class="metric-value">{manipulation_score}%</div><div class="metric-label">Manipulation Signal Score</div></div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class="metric-box"><div class="metric-value">{"HIGH" if corroborated else "LOW"}</div><div class="metric-label">Live Corroboration</div></div>""", unsafe_allow_html=True)

                    if st.session_state.show_learned_insights:
                        with st.expander("🔬 Forensic-rule vs. learned-model detail"):
                            if media_learned_pred is not None:
                                st.markdown(f"- **Learned-model prediction:** {media_learned_pred} ({media_learned_confidence*100:.0f}% confidence)")
                            else:
                                st.markdown("- **Learned-model prediction:** not available yet (train it in the Feedback tab).")

                    if forensic_notes:
                        with st.expander("🔬 Forensic analysis details"):
                            for note in forensic_notes:
                                st.markdown(f"- {note}")

else:
    st.markdown("### 🧠 Model Feedback & Active Learning Hub")
    st.markdown("Provide feedback on predictions, submit ground-truth corrections, and retrain the model memory to improve prediction accuracy.")

    total_feedback = len(st.session_state.feedback_dataset)
    correct_count = sum(1 for item in st.session_state.feedback_dataset if item.get('is_correct') == 'Yes 👍')
    accuracy_rate = int((correct_count / max(total_feedback, 1)) * 100)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{total_feedback}</div><div class="metric-label">Feedback Samples</div></div>""", unsafe_allow_html=True)
    with f2:
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{accuracy_rate}%</div><div class="metric-label">User Accuracy Rate</div></div>""", unsafe_allow_html=True)
    with f3:
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{correct_count}</div><div class="metric-label">Verified Correct</div></div>""", unsafe_allow_html=True)
    with f4:
        st.markdown(f"""<div class="metric-box"><div class="metric-value">{total_feedback - correct_count}</div><div class="metric-label">Corrections Logged</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_fb1, col_fb2 = st.columns([1, 1])

    with col_fb1:
        st.markdown("#### 💬 Submit Accuracy Feedback")
        
        claim_type = st.session_state.last_analyzed_claim['type'] if st.session_state.last_analyzed_claim else 'Text Claim'

        if st.session_state.last_analyzed_claim:
            last_text = st.session_state.last_analyzed_claim['content']
            last_verdict = st.session_state.last_analyzed_claim['predicted_verdict']
            last_features = st.session_state.last_analyzed_claim.get('features')
            st.info(f"**Last Analyzed {claim_type}:** {last_text}\n\n**Predicted Verdict:** {last_verdict}")
        else:
            st.info("No claim analyzed in current session yet. Run a Text or Media check first, then come back here to give feedback on it.")
            last_text = ""
            last_verdict = "🟢 VERIFIED REAL / HIGHLY LIKELY"
            last_features = None

        claim_to_feedback = st.text_area("Claim / Content for Training:", value=last_text, height=90)
        is_accurate = st.radio("Was the prediction accurate?", ["Yes 👍", "No 👎"], horizontal=True)

        if claim_type == 'Media File':
            label_options = [
                "🟢 REAL VIDEO", "🟢 REAL IMAGE / GRAPHIC", "✅ NO MANIPULATION SIGNALS DETECTED",
                "🚨 FAKE AI GENERATED VIDEO", "🚨 FAKE AI GENERATED IMAGE",
                "⚠️ SIGNS OF MANIPULATION DETECTED", "⚠️ UNVERIFIED — NO STRONG SIGNAL EITHER WAY"
            ]
        else:
            label_options = [
                "🟢 VERIFIED REAL / HIGHLY LIKELY",
                "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT",
                "⚠️ UNVERIFIED / PROBABLE FAKE NEWS"
            ]
        corrected_verdict = st.selectbox("Select Correct Ground-Truth Label:", label_options)

        if st.button("💾 Submit Feedback (saved to disk)", type="primary"):
            if claim_to_feedback.strip():
                entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'type': claim_type,
                    'content': claim_to_feedback,
                    'predicted_verdict': last_verdict,
                    'is_correct': is_accurate,
                    'corrected_label': corrected_verdict if is_accurate == "No 👎" else last_verdict,
                    'features': last_features
                }
                st.session_state.feedback_dataset.append(entry)
                saved_ok = append_feedback_entry(entry)
                if saved_ok:
                    st.success("Feedback saved to disk - it will persist across app restarts.")
                else:
                    st.warning("Feedback recorded for this session, but couldn't be written to disk (filesystem may be read-only in this deployment) - it won't survive a restart.")
                if last_features is None:
                    st.caption("Note: this claim was analyzed before the feature-capture update, or corrections were entered manually - it has no feature vector, so it won't be usable for training until you analyze it fresh via the checker.")
                st.rerun()
            else:
                st.warning("Please enter or select a claim before submitting feedback.")

    with col_fb2:
        st.markdown("#### ⚡ Retrain & Calibrate Model Weights")
        st.markdown(f"""
        This trains a real `LogisticRegression` classifier on your feedback's stored
        numeric signals (word overlap, TF-IDF similarity, sensationalism score, NLI
        contradiction score, fact-check source flag, etc. for text; forensic scores
        for media) against your corrected labels.

        **Hybrid switch:** the learned model only becomes the PRIMARY verdict source
        once it earns it - at least **{MIN_SAMPLES_FOR_PRIMARY} samples** and a genuine
        **held-out accuracy ≥ {MIN_HELDOUT_ACCURACY_FOR_PRIMARY:.0f}%**. Below that bar, it's
        shown only as an advisory comparison and the rule-based/forensic engine stays
        primary - so accuracy can only ever go up relative to today, never down.
        """)

        if st.button("🔄 Trigger Real Retraining Cycle"):
            with st.spinner("Training logistic regression models on stored feedback..."):
                all_entries = st.session_state.feedback_dataset
                text_entries = [e for e in all_entries if e.get('type') == 'Text Claim']
                media_entries = [e for e in all_entries if e.get('type') == 'Media File']

                text_result = train_model_from_feedback(text_entries, TEXT_FEATURE_KEYS)
                media_result = train_model_from_feedback(media_entries, MEDIA_FEATURE_KEYS)

                if text_result['model'] is not None:
                    text_meta = {'n_samples': text_result['n_samples'], 'accuracy': text_result['accuracy'], 'accuracy_type': text_result['accuracy_type']}
                    st.session_state.text_learned_model = text_result['model']
                    st.session_state.text_model_meta = text_meta
                    save_model(text_result['model'], TEXT_MODEL_PATH, meta=text_meta)
                    earned = text_meta['n_samples'] >= MIN_SAMPLES_FOR_PRIMARY and text_meta['accuracy_type'] == 'held-out test split' and (text_meta['accuracy'] or 0) >= MIN_HELDOUT_ACCURACY_FOR_PRIMARY
                    st.success(f"Text model trained on {text_result['n_samples']} samples across {text_result['n_classes']} labels. Accuracy: {text_result['accuracy']}% ({text_result['accuracy_type']}).")
                    st.caption(f"Label distribution: {text_result['class_counts']}")
                    st.markdown("✅ **This model has earned PRIMARY status** - it will now drive the Text Fact-Checker's verdicts." if earned else f"⏳ Not primary yet - needs ≥{MIN_SAMPLES_FOR_PRIMARY} samples with held-out accuracy ≥{MIN_HELDOUT_ACCURACY_FOR_PRIMARY:.0f}%. Still shown as an advisory comparison.")
                else:
                    st.info(f"Text model not (re)trained: {text_result['message']}")

                if media_result['model'] is not None:
                    media_meta = {'n_samples': media_result['n_samples'], 'accuracy': media_result['accuracy'], 'accuracy_type': media_result['accuracy_type']}
                    st.session_state.media_learned_model = media_result['model']
                    st.session_state.media_model_meta = media_meta
                    save_model(media_result['model'], MEDIA_MODEL_PATH, meta=media_meta)
                    earned = media_meta['n_samples'] >= MIN_SAMPLES_FOR_PRIMARY and media_meta['accuracy_type'] == 'held-out test split' and (media_meta['accuracy'] or 0) >= MIN_HELDOUT_ACCURACY_FOR_PRIMARY
                    st.success(f"Media model trained on {media_result['n_samples']} samples across {media_result['n_classes']} labels. Accuracy: {media_result['accuracy']}% ({media_result['accuracy_type']}).")
                    st.caption(f"Label distribution: {media_result['class_counts']}")
                    st.markdown("✅ **This model has earned PRIMARY status** - it will now drive the Media Authenticator's verdicts." if earned else f"⏳ Not primary yet - needs ≥{MIN_SAMPLES_FOR_PRIMARY} samples with held-out accuracy ≥{MIN_HELDOUT_ACCURACY_FOR_PRIMARY:.0f}%. Still shown as an advisory comparison.")
                else:
                    st.info(f"Media model not (re)trained: {media_result['message']}")

                if text_result['model'] is None and media_result['model'] is None:
                    st.warning("Enable '🧪 Show experimental learned-model insights' in the sidebar once a model trains successfully, to see its predictions alongside future checks.")

    st.divider()

    st.markdown("### 📊 Collected Model Training & Feedback Dataset")
    if st.session_state.feedback_dataset:
        df_feedback = pd.DataFrame([{k: v for k, v in e.items() if k != 'features'} for e in st.session_state.feedback_dataset])
        st.dataframe(df_feedback, use_container_width=True)

        fb_csv = df_feedback.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Feedback Training Dataset (CSV)",
            data=fb_csv,
            file_name="verifact_training_feedback.csv",
            mime="text/csv"
        )
    else:
        st.info("No feedback samples recorded yet.")

st.divider()
if st.session_state.verification_history:
    st.markdown("### 📜 Session Verification Audit Log")
    df_history = pd.DataFrame(st.session_state.verification_history)
    st.dataframe(df_history, use_container_width=True)
    
    csv_data = df_history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Verification Audit Log (CSV)",
        data=csv_data,
        file_name="verifact_audit_history.csv",
        mime="text/csv"
    )
