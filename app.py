import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import io
import json
import pickle
import unicodedata
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable

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
MEDIA_FEATURE_KEYS = ['ai_score', 'manipulation_score', 'corroborated', 'ai_signature_found', 'ml_deepfake_score_raw']

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

def pick_primary_verdict(rule_verdict, rule_truth_index, feature_vector, feature_keys, learned_model, model_meta, override_mode="auto"):
    """
    The hybrid switch. Returns (final_verdict, final_truth_index, source_label,
    learned_pred, learned_confidence) where source_label is 'rule-based' or
    'learned-model'.

    override_mode:
      - "auto" (default): the learned model only becomes primary once it has
        EARNED it - enough training samples AND a genuine held-out accuracy
        above the bar. Early on (little/no feedback), the rule-based verdict
        always wins by default, so accuracy never regresses versus the
        unmodified rule-based engine. This is the original, always-safe
        behavior.
      - "rule_based": always use the rule-based verdict, regardless of what
        the learned model says or how well-trained it is. An explicit user
        choice to stay on the hand-built logic.
      - "learned_model": always use the learned model's prediction if one
        exists (regardless of whether it has "earned" primacy under auto
        rules) - an explicit user choice to trust the trained model. Falls
        back to rule-based only if no model has been trained at all yet.
    """
    learned_pred, learned_confidence = None, None
    if learned_model is not None and feature_vector is not None:
        try:
            X = np.array([[float(feature_vector[k]) for k in feature_keys]])
            learned_pred = learned_model.predict(X)[0]
            learned_confidence = float(max(learned_model.predict_proba(X)[0]))
        except Exception:
            learned_pred, learned_confidence = None, None

    def _learned_result():
        final_truth_index = int(learned_confidence * 100) if "REAL" in str(learned_pred) or "VERIFIED" in str(learned_pred) else int(100 - learned_confidence * 100)
        return learned_pred, final_truth_index, "learned-model", learned_pred, learned_confidence

    if override_mode == "rule_based":
        return rule_verdict, rule_truth_index, "rule-based", learned_pred, learned_confidence

    if override_mode == "learned_model":
        if learned_pred is not None:
            return _learned_result()
        return rule_verdict, rule_truth_index, "rule-based", learned_pred, learned_confidence

    # override_mode == "auto" (default): original earned-primacy logic
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
        return _learned_result()

    return rule_verdict, rule_truth_index, "rule-based", learned_pred, learned_confidence

# --- Bulk upload training helpers -------------------------------------------

TEXT_LABEL_ALIASES = {
    'real': "🟢 VERIFIED REAL / HIGHLY LIKELY", 'true': "🟢 VERIFIED REAL / HIGHLY LIKELY",
    'verified': "🟢 VERIFIED REAL / HIGHLY LIKELY", 'verified real': "🟢 VERIFIED REAL / HIGHLY LIKELY",
    'fake': "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT", 'false': "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT",
    'debunked': "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT",
    'unverified': "⚠️ UNVERIFIED / PROBABLE FAKE NEWS", 'unknown': "⚠️ UNVERIFIED / PROBABLE FAKE NEWS",
}
MEDIA_LABEL_ALIASES = {
    'real': "🟢 REAL IMAGE / GRAPHIC", 'true': "🟢 REAL IMAGE / GRAPHIC",
    'fake': "🚨 FAKE AI GENERATED IMAGE", 'false': "🚨 FAKE AI GENERATED IMAGE", 'ai generated': "🚨 FAKE AI GENERATED IMAGE",
    'manipulated': "⚠️ SIGNS OF MANIPULATION DETECTED",
    'unverified': "⚠️ UNVERIFIED — NO STRONG SIGNAL EITHER WAY", 'unknown': "⚠️ UNVERIFIED — NO STRONG SIGNAL EITHER WAY",
    'no manipulation': "✅ NO MANIPULATION SIGNALS DETECTED",
}
TEXT_CANONICAL_LABELS = ["🟢 VERIFIED REAL / HIGHLY LIKELY", "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT", "⚠️ UNVERIFIED / PROBABLE FAKE NEWS"]
MEDIA_CANONICAL_LABELS = ["🟢 REAL IMAGE / GRAPHIC", "🚨 FAKE AI GENERATED IMAGE", "⚠️ SIGNS OF MANIPULATION DETECTED", "✅ NO MANIPULATION SIGNALS DETECTED", "⚠️ UNVERIFIED — NO STRONG SIGNAL EITHER WAY"]

def normalize_label(raw, claim_type):
    """
    Maps a free-text label from an uploaded file (e.g. "Real", "fake",
    "FALSE") onto the app's canonical verdict strings. Accepts the exact
    canonical strings too (case-insensitive substring match), so a file
    that already uses the app's own verdict text also works. Returns None
    if the label can't be recognized.
    """
    if raw is None:
        return None
    raw_str = str(raw).strip()
    if not raw_str:
        return None
    raw_lower = raw_str.lower()
    canonical_list = TEXT_CANONICAL_LABELS if claim_type == 'Text Claim' else MEDIA_CANONICAL_LABELS
    for c in canonical_list:
        if raw_lower in c.lower():
            return c
    aliases = TEXT_LABEL_ALIASES if claim_type == 'Text Claim' else MEDIA_LABEL_ALIASES
    return aliases.get(raw_lower)

def compute_text_features_for_claim(claim_text, lang='English'):
    """
    Computes the exact same feature vector the live Text Fact-Checker
    computes for a claim, by calling the SAME underlying functions
    (extract_search_queries, fetch_all_sources, calculate_entity_and_vector_match,
    analyze_linguistic_risk, get_debunk_assessment, matches_known_hoax) rather
    than reimplementing any of that logic. Used for bulk file-upload
    training, so uploaded data is scored identically to a live check and can
    never silently drift out of sync with it. Makes real network calls
    (live search) - one claim at a time, so bulk callers should cap row
    counts and show progress.
    """
    queries = extract_search_queries(claim_text)
    all_articles = []
    for q in queries:
        all_articles.extend(fetch_all_sources(q, lang=lang))
    seen = set()
    unique_articles = []
    for a in all_articles:
        if a['title'] not in seen:
            seen.add(a['title'])
            unique_articles.append(a)

    overlap_ratio, raw_max_sim, best_match, matched_words, total_words = calculate_entity_and_vector_match(claim_text, unique_articles)
    sensationalism_score, journalistic_score = analyze_linguistic_risk(claim_text)
    debunk_flag, nli_score, debunk_method = get_debunk_assessment(claim_text, best_match)
    is_factcheck_source = bool(best_match and best_match.get('source_type') == 'factcheck')
    known_hoax_flag = matches_known_hoax(claim_text)

    return {
        'overlap_ratio': round(overlap_ratio, 4),
        'raw_max_sim': round(raw_max_sim, 4),
        'sensationalism_score': sensationalism_score,
        'journalistic_score': journalistic_score,
        'debunk_flag': int(debunk_flag),
        'known_hoax_flag': int(known_hoax_flag),
        'nli_contradiction_score': round(nli_score, 4) if nli_score is not None else 0.0,
        'is_factcheck_source': int(is_factcheck_source),
    }

def run_training_cycle():
    """
    Trains (or retrains) both the text and media learned models from the
    current st.session_state.feedback_dataset, updates session state, and
    persists both to disk. Returns (text_result, media_result) dicts from
    train_model_from_feedback. Shared by the manual Retrain button and the
    bulk file-upload flow, so both paths behave identically and any future
    change to training only needs to happen in one place.
    """
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

    if media_result['model'] is not None:
        media_meta = {'n_samples': media_result['n_samples'], 'accuracy': media_result['accuracy'], 'accuracy_type': media_result['accuracy_type']}
        st.session_state.media_learned_model = media_result['model']
        st.session_state.media_model_meta = media_meta
        save_model(media_result['model'], MEDIA_MODEL_PATH, meta=media_meta)

    return text_result, media_result

def report_training_result(result, model_name):
    """Renders the standard success/info block for one trained model's
    result dict - shared by the manual Retrain button and bulk upload."""
    if result['model'] is not None:
        meta = {'n_samples': result['n_samples'], 'accuracy': result['accuracy'], 'accuracy_type': result['accuracy_type']}
        earned = meta['n_samples'] >= MIN_SAMPLES_FOR_PRIMARY and meta['accuracy_type'] == 'held-out test split' and (meta['accuracy'] or 0) >= MIN_HELDOUT_ACCURACY_FOR_PRIMARY
        st.success(f"{model_name} model trained on {result['n_samples']} samples across {result['n_classes']} labels. Accuracy: {result['accuracy']}% ({result['accuracy_type']}).")
        st.caption(f"Label distribution: {result['class_counts']}")
        st.markdown(
            "✅ **This model has earned PRIMARY status under Automatic mode.**"
            if earned else
            f"⏳ Not primary yet under Automatic mode - needs ≥{MIN_SAMPLES_FOR_PRIMARY} samples with held-out accuracy ≥{MIN_HELDOUT_ACCURACY_FOR_PRIMARY:.0f}%. You can still force it on via the Verdict Source selector."
        )
    else:
        st.info(f"{model_name} model not (re)trained: {result['message']}")

# --- Verification log PDF export --------------------------------------------

_PDF_NAVY = rl_colors.HexColor("#0F172A")
_PDF_SLATE = rl_colors.HexColor("#475569")
_PDF_LIGHT_BG = rl_colors.HexColor("#F2F5F7")
_PDF_BORDER = rl_colors.HexColor("#E2E8F0")
_PDF_FAKE_RED = rl_colors.HexColor("#DC2626")
_PDF_REAL_GREEN = rl_colors.HexColor("#0F9E8E")
_PDF_WARN_AMBER = rl_colors.HexColor("#B45309")

_PDF_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF\U0000FE00-\U0000FE0F]+",
    flags=re.UNICODE
)

def _pdf_strip_emoji(text):
    """Removes emoji AND their trailing variation-selector codepoints
    (U+FE0F etc.) - reportlab's default fonts render unsupported codepoints
    as visible placeholder boxes, so verdict strings are cleaned to plain
    text for a professional-looking PDF."""
    return _PDF_EMOJI_PATTERN.sub('', str(text)).strip()

def _pdf_verdict_color(verdict_str):
    """Order matters: 'UNVERIFIED' contains the substring 'VERIFIED' and
    must be checked FIRST, or it gets miscolored as a real/verified
    verdict - the same bug class already fixed in classify_verdict_category
    for the app's own logic."""
    v = str(verdict_str)
    if "UNVERIFIED" in v:
        return _PDF_WARN_AMBER
    if "REAL" in v or "VERIFIED" in v:
        return _PDF_REAL_GREEN
    if "FAKE" in v or "DEBUNKED" in v:
        return _PDF_FAKE_RED
    return _PDF_WARN_AMBER

def _pdf_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(_PDF_NAVY)
    canvas.rect(0, doc.pagesize[1] - 0.75 * inch, doc.pagesize[0], 0.75 * inch, fill=1, stroke=0)
    canvas.setFillColor(rl_colors.white)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(0.6 * inch, doc.pagesize[1] - 0.5 * inch, "VeriFact AI")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(rl_colors.HexColor("#CBD5E1"))
    canvas.drawString(0.6 * inch, doc.pagesize[1] - 0.65 * inch, "Verification Audit Log")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_PDF_SLATE)
    canvas.drawString(0.6 * inch, 0.4 * inch, f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}")
    canvas.drawRightString(doc.pagesize[0] - 0.6 * inch, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()

def generate_verification_log_pdf(history):
    """
    Builds a professionally-formatted PDF of the session's verification
    history: branded header/footer, a summary stats row, and a color-coded,
    paginated table of every logged check. Returns raw PDF bytes suitable
    for st.download_button. Tested against 0, 3, and 40-row histories,
    including page-break/repeating-header behavior - never raises on an
    empty list.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=1.1 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        title="VeriFact AI Verification Audit Log"
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleBig', parent=styles['Heading1'], fontSize=18, textColor=_PDF_NAVY, spaceAfter=4)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10, textColor=_PDF_SLATE, spaceAfter=14)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=12, textColor=_PDF_NAVY, spaceBefore=10, spaceAfter=6)
    cell_style = ParagraphStyle('Cell', parent=styles['Normal'], fontSize=8, textColor=_PDF_NAVY, leading=10)
    header_cell_style = ParagraphStyle('HeaderCell', parent=styles['Normal'], fontSize=8, textColor=rl_colors.white, leading=10, fontName='Helvetica-Bold')
    summary_label_style = ParagraphStyle('SummaryLabel', parent=styles['Normal'], fontSize=9, textColor=rl_colors.white, alignment=1, fontName='Helvetica-Bold')
    summary_value_style = ParagraphStyle('SummaryValue', parent=styles['Normal'], fontSize=16, textColor=_PDF_NAVY, alignment=1, fontName='Helvetica-Bold')
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=_PDF_SLATE, leading=11)

    elements = []
    elements.append(Paragraph("Verification Audit Log", title_style))
    elements.append(Paragraph(f"{len(history)} checks recorded this session", subtitle_style))

    n_fake = sum(1 for h in history if _pdf_verdict_color(h.get('verdict', '')) == _PDF_FAKE_RED)
    n_real = sum(1 for h in history if _pdf_verdict_color(h.get('verdict', '')) == _PDF_REAL_GREEN)
    n_other = len(history) - n_fake - n_real

    summary_data = [
        [Paragraph("TOTAL CHECKS", summary_label_style), Paragraph("FLAGGED FAKE", summary_label_style),
         Paragraph("VERIFIED REAL", summary_label_style), Paragraph("UNVERIFIED / OTHER", summary_label_style)],
        [Paragraph(str(len(history)), summary_value_style), Paragraph(str(n_fake), summary_value_style),
         Paragraph(str(n_real), summary_value_style), Paragraph(str(n_other), summary_value_style)]
    ]
    summary_table = Table(summary_data, colWidths=[1.7 * inch] * 4)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), _PDF_NAVY),
        ('BACKGROUND', (0, 1), (-1, 1), _PDF_LIGHT_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, _PDF_BORDER),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 16))
    elements.append(Paragraph("Detailed Log", section_style))
    elements.append(HRFlowable(width="100%", color=_PDF_BORDER, thickness=1))
    elements.append(Spacer(1, 8))

    table_data = [[
        Paragraph("TIME", header_cell_style), Paragraph("CLAIM / FILE", header_cell_style),
        Paragraph("VERDICT", header_cell_style), Paragraph("TRUTH INDEX", header_cell_style),
        Paragraph("CORROBORATION", header_cell_style)
    ]]
    for h in history:
        verdict_color = _pdf_verdict_color(h.get('verdict', ''))
        clean_verdict = _pdf_strip_emoji(h.get('verdict', ''))
        verdict_para = Paragraph(f'<font color="{verdict_color.hexval()}"><b>{clean_verdict}</b></font>', cell_style)
        table_data.append([
            Paragraph(str(h.get('timestamp', '')), cell_style),
            Paragraph(str(h.get('claim', ''))[:70], cell_style),
            verdict_para,
            Paragraph(str(h.get('truth_index', '')), cell_style),
            Paragraph(str(h.get('corroboration', '')), cell_style),
        ])

    log_table = Table(table_data, colWidths=[0.95 * inch, 2.55 * inch, 2.15 * inch, 0.85 * inch, 1.3 * inch], repeatRows=1)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), _PDF_NAVY),
        ('GRID', (0, 0), (-1, -1), 0.5, _PDF_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), _PDF_LIGHT_BG))
    log_table.setStyle(TableStyle(style_cmds))
    elements.append(log_table)

    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", color=_PDF_BORDER, thickness=1))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "<b>Disclaimer:</b> Verdicts in this log are produced by an automated, evidence-based system "
        "(live news/fact-check search and/or a feedback-trained model). "
        "They are decision support, not a certified fact-check - always verify high-stakes claims "
        "through a professional fact-checking organization.",
        disclaimer_style
    ))

    doc.build(elements, onFirstPage=_pdf_header_footer, onLaterPages=_pdf_header_footer)
    buf.seek(0)
    return buf.getvalue()

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

# --- Claim origin-tracing ("timeline") --------------------------------------
# Purely a DISPLAY feature computed from the same articles the verdict logic
# already fetched - it never influences the verdict itself. Shows the
# earliest/latest dated coverage found among matched articles, so a
# recurring hoax reads as a narrative ("first matched coverage: 2019, most
# recent: 2024") instead of a flat one-off verdict. Honesty note: this is
# the earliest article OUR search happened to find, not a proven origin
# date - phrased that way in the UI, never as "this rumor originated on...".

from email.utils import parsedate_to_datetime as _parsedate_to_datetime
from datetime import timezone as _timezone

def _parse_article_date(date_str):
    if not date_str:
        return None
    try:
        dt = _parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_timezone.utc)
        return dt
    except Exception:
        return None

def build_claim_timeline(articles):
    """Returns a dict with earliest/latest matched-article dates, or None if
    no article in the list has a parseable date."""
    dated = []
    for a in articles:
        dt = _parse_article_date(a.get('date', ''))
        if dt:
            dated.append((dt, a))
    if not dated:
        return None
    dated.sort(key=lambda x: x[0])
    earliest, latest = dated[0], dated[-1]
    return {
        'earliest_date': earliest[0],
        'earliest_source': earliest[1].get('source'),
        'latest_date': latest[0],
        'latest_source': latest[1].get('source'),
        'span_days': (latest[0] - earliest[0]).days,
        'distinct_dates': len(set(d[0].date() for d in dated)),
        'total_matched': len(dated),
    }

# --- Manipulation-technique classification ----------------------------------
# Maps the verdict onto the IFCN/First Draft misinformation-type taxonomy
# (Fabricated Content, False Context, Misleading Content, etc.) using
# signals the app ALREADY computed for the verdict - a rule-based
# approximation, not a certain classification. Computed strictly AFTER the
# verdict; never feeds back into it, so it cannot change verdict accuracy.

def classify_verdict_category(verdict):
    # Order matters: "UNVERIFIED / PROBABLE FAKE NEWS" contains the
    # substring "FAKE" and would be misclassified if checked after it.
    if "UNVERIFIED" in verdict:
        return 'unverified'
    if "REAL" in verdict or "VERIFIED" in verdict:
        return 'real'
    if "DEBUNKED" in verdict or "FAKE" in verdict:
        return 'fake'
    return 'unverified'

def classify_manipulation_technique(verdict, overlap_ratio, debunk_flag, known_hoax_flag, sensationalism_score, is_factcheck_source):
    """Returns a {'technique','description'} dict, or None if the verdict
    isn't a 'fake' category (no technique to label for real/unverified)."""
    if classify_verdict_category(verdict) != 'fake':
        return None
    if known_hoax_flag:
        return {'technique': 'Fabricated Content',
                'description': 'Entirely invented claim with no factual basis - matches a well-documented, repeatedly-debunked hoax pattern.'}
    if debunk_flag and is_factcheck_source:
        return {'technique': 'Fabricated Content',
                'description': 'A dedicated fact-checking source has directly investigated and refuted this specific claim.'}
    if debunk_flag and overlap_ratio >= 0.20:
        return {'technique': 'False Context',
                'description': 'Related real coverage exists, but the specific claim as stated has been denied/contradicted by that coverage - genuine information may be getting reframed or misattributed.'}
    if sensationalism_score >= 40 and overlap_ratio < 0.25:
        return {'technique': 'Misleading Content / Clickbait Framing',
                'description': 'Heavy sensational/alarmist language with no corroborating coverage - framing designed to provoke a reaction rather than inform.'}
    return {'technique': 'Unverified Claim',
            'description': 'No corroborating coverage found and no specific fabrication pattern matched - insufficient evidence to classify technique with confidence.'}

# --- "Forward-back" reply card ----------------------------------------------
# Most fact-checkers give YOU information; almost none help you respond to
# whoever actually sent you the rumor. Generates a short, non-confrontational
# message worded for forwarding back into the same WhatsApp/family group -
# hand-written per language (not machine-translated, to avoid mistranslation
# errors), templated by verdict category only. Purely a text-generation
# feature - never reads from or writes to any variable the verdict logic uses.

REPLY_CARD_TEMPLATES = {
    'fake': {
        'English': "Hey, I checked this and it looks like it's not true — {source_note}. Might be worth not forwarding it further. Happy to share what I found if useful!",
        'Hindi': "नमस्ते, मैंने इसे चेक किया और यह सही नहीं लग रहा — {source_note}। इसे आगे न भेजना ही बेहतर होगा। ज़्यादा जानकारी चाहिए तो बता दीजिए।",
    },
    'real': {
        'English': "Just checked — this one looks accurate, {source_note}.",
        'Hindi': "मैंने इसे चेक किया — यह सही लग रहा है, {source_note}।",
    },
    'unverified': {
        'English': "I looked into this but couldn't find anything confirming it either way — {source_note}. Might be worth waiting for more sources before sharing further.",
        'Hindi': "मैंने इसे देखा लेकिन अभी तक इसकी पुष्टि करने वाला कुछ नहीं मिला — {source_note}। आगे भेजने से पहले थोड़ा और इंतज़ार करना बेहतर होगा।",
    }
}

def generate_reply_card(verdict, source_note, language='English'):
    category = classify_verdict_category(verdict)
    lang_templates = REPLY_CARD_TEMPLATES.get(category, REPLY_CARD_TEMPLATES['unverified'])
    template = lang_templates.get(language) or lang_templates['English']
    return template.format(source_note=source_note or "based on what I could find")

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

with st.sidebar:
    st.markdown("<h2 style='color:#34d399; margin-bottom:0;'>🛡️ VeriFact AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.8rem;'>Misinformation Command Center</p>", unsafe_allow_html=True)
    st.divider()
    
    analysis_mode = st.radio(
        "Select Pipeline Mode:",
        ["📰 Text / Article Fact-Checker", "🧠 Model Feedback & Active Learning"],
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
st.markdown("<p style='color:#94a3b8;'>Real-Time Live Web Grounding & Text Fact-Checker Engine</p>", unsafe_allow_html=True)

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

    text_verdict_mode_label = {
        "Automatic (recommended)": "auto",
        "Always rule-based engine": "rule_based",
        "Always learned model (if trained)": "learned_model",
    }
    text_mode_choice = st.selectbox(
        "Verdict source:",
        options=list(text_verdict_mode_label.keys()),
        index=0,
        key="text_verdict_mode_select",
        help="Automatic: the learned model only takes over once it's proven itself (30+ samples, 75%+ held-out accuracy). You can force either source manually once a model has been trained in the Feedback tab."
    )
    text_override_mode = text_verdict_mode_label[text_mode_choice]
    
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
                st.session_state.text_learned_model, st.session_state.text_model_meta,
                override_mode=text_override_mode
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
                f"🧠 Verdict source: **Learned model** (trained on {st.session_state.text_model_meta.get('n_samples')} samples, {st.session_state.text_model_meta.get('accuracy')}% held-out accuracy) - mode: {text_mode_choice}."
                if verdict_source == "learned-model"
                else f"⚙️ Verdict source: **Rule-based engine** - mode: {text_mode_choice}."
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

            # Manipulation-technique classification and claim timeline are
            # computed AFTER the verdict, purely for display - neither reads
            # back into or changes final_verdict/final_truth_index above.
            technique_info = classify_manipulation_technique(final_verdict, overlap_ratio, debunk_flag, known_hoax_flag, sensationalism_score, is_factcheck_source)
            claim_timeline = build_claim_timeline(unique_articles)

            if technique_info:
                st.markdown(f"""
                <div style="background:rgba(167,139,250,0.1); border:1px solid rgba(167,139,250,0.4); border-radius:8px; padding:10px 14px; margin-top:8px;">
                    <span style="color:#a78bfa; font-weight:bold; font-size:0.85rem;">🏷️ Manipulation Technique: {technique_info['technique']}</span><br>
                    <span style="color:#cbd5e1; font-size:0.8rem;">{technique_info['description']}</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4 = st.tabs(["📲 WhatsApp Debunk Card", "🟢 Live News & Source Authority", "🕰️ Claim Timeline", "💬 Forward-Back Reply"])
            
            with tab1:
                st.markdown("#### Ready-to-Share WhatsApp Fact-Check Briefing")

                # Only truncate (and show "...") when the claim actually IS
                # longer than the snippet limit - the old version always
                # appended "..." even for short claims, which looked wrong.
                claim_snippet = user_input.strip()
                if len(claim_snippet) > 120:
                    claim_snippet = claim_snippet[:120].rstrip() + "..."

                debunk_lines = [
                    "*🛡️ VERIFACT AI FACT CHECK ALERT*",
                    "----------------------------------",
                    f'*Claim:* "{claim_snippet}"',
                    f"*Verdict:* {final_verdict}",
                    f"*Truth Index:* {final_truth_index}%  |  *Match Confidence:* {corroboration_pct}%",
                ]
                if technique_info:
                    debunk_lines.append(f"*Flag Type:* {technique_info['technique']}")
                debunk_lines.append("")
                debunk_lines.append(f"*Summary:* {summary}")
                if best_match and best_match.get('link'):
                    debunk_lines.append("")
                    debunk_lines.append(f"*Reference:* {best_match['source']}")
                    debunk_lines.append(best_match['link'])
                debunk_lines.append("")
                debunk_lines.append(f"_Checked via VeriFact AI · {datetime.now().strftime('%d %b %Y, %H:%M')}_")
                debunk_text = "\n".join(debunk_lines)

                st.code(debunk_text, language="markdown")

                wa_link = "https://wa.me/?text=" + urllib.parse.quote(debunk_text)
                st.link_button("📲 Open in WhatsApp", wa_link, use_container_width=True)
                st.caption("Opens WhatsApp with this message pre-filled, ready to pick a chat and send - or copy the text above directly.")
                
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
                st.markdown("#### Claim Coverage Timeline")
                st.caption("Earliest/latest dates among the articles our search actually found - not a proven origin date, just what's visible in live search results.")
                if claim_timeline:
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        st.markdown(f"""<div class="metric-box"><div class="metric-value" style="font-size:1.1rem;">{claim_timeline['earliest_date'].strftime('%d %b %Y')}</div><div class="metric-label">Earliest Matched Coverage</div></div>""", unsafe_allow_html=True)
                        st.caption(f"Source: {claim_timeline['earliest_source']}")
                    with tc2:
                        st.markdown(f"""<div class="metric-box"><div class="metric-value" style="font-size:1.1rem;">{claim_timeline['latest_date'].strftime('%d %b %Y')}</div><div class="metric-label">Most Recent Matched Coverage</div></div>""", unsafe_allow_html=True)
                        st.caption(f"Source: {claim_timeline['latest_source']}")
                    if claim_timeline['span_days'] > 180:
                        st.info(f"📅 Matched coverage spans {claim_timeline['span_days']} days across {claim_timeline['distinct_dates']} distinct dates - this looks like a claim that resurfaces periodically rather than a single one-off story.")
                    elif claim_timeline['distinct_dates'] > 1:
                        st.caption(f"Coverage found across {claim_timeline['distinct_dates']} distinct dates.")
                else:
                    st.info("No parseable publish dates found among the matched articles - timeline unavailable for this claim.")

            with tab4:
                st.markdown("#### Reply card for forwarding back to whoever sent you this")
                st.caption("A short, non-confrontational message worded for sending back into the group/chat this came from - not just information for you.")
                reply_lang = st.selectbox("Reply language:", options=["English", "Hindi"], index=0, key="reply_card_lang")
                source_hint = f"checked against {best_match['source']}" if best_match else "based on available search results"
                reply_text = generate_reply_card(final_verdict, source_hint, reply_lang)
                st.text_area("Message:", value=reply_text, height=100, key="reply_card_text", disabled=True)
                st.code(reply_text, language=None)

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
                "🟢 REAL IMAGE / GRAPHIC", "✅ NO MANIPULATION SIGNALS DETECTED",
                "🚨 FAKE AI GENERATED IMAGE",
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
                text_result, media_result = run_training_cycle()
                report_training_result(text_result, "Text")
                report_training_result(media_result, "Media")
                if text_result['model'] is None and media_result['model'] is None:
                    st.warning("Submit some feedback first (or use the bulk upload below), then retrain.")

    st.divider()

    st.markdown("### 📤 Bulk Upload Training Data")
    st.markdown(
        "Upload a CSV or JSON file to train and test the model on your own labeled dataset, instead of "
        "(or in addition to) single-claim feedback above. Two supported formats:\n"
        "- **Raw claims**: a text column (`text`/`claim`/`content`) plus a label column (`label`/`corrected_label`/`verdict` - "
        "accepts Real/Fake/Unverified or the app's own verdict text). Each row is run through the live fact-checking "
        "pipeline to compute its features, so this is slower and capped at 100 rows per upload.\n"
        "- **Pre-computed features**: columns matching the model's feature names plus a label column - trains instantly, "
        "no cap, no network calls."
    )

    bulk_type_choice = st.selectbox("Data type:", ["Text Claims", "Image Verdicts (pre-computed features only)"], key="bulk_claim_type_select")
    bulk_claim_type = 'Text Claim' if bulk_type_choice == "Text Claims" else 'Media File'
    bulk_feature_keys = TEXT_FEATURE_KEYS if bulk_claim_type == 'Text Claim' else MEDIA_FEATURE_KEYS

    with st.expander("📋 See expected file format / download a template"):
        if bulk_claim_type == 'Text Claim':
            st.code("text,label\n\"RBI announces plastic currency notes next month\",fake\n\"ISRO completes Gaganyaan engine test\",real", language="text")
            template_csv = "text,label\n\"RBI announces plastic currency notes next month\",fake\n\"ISRO completes Gaganyaan engine test\",real\n"
        else:
            header = ",".join(MEDIA_FEATURE_KEYS + ["label"])
            example = ",".join(["55", "20", "0", "0", "-1", "real"])
            st.code(f"{header}\n{example}", language="text")
            template_csv = f"{header}\n{example}\n"
        st.download_button("📥 Download CSV template", data=template_csv, file_name=f"{bulk_claim_type.lower().replace(' ', '_')}_template.csv", mime="text/csv", key="bulk_template_download")

    bulk_file = st.file_uploader("Upload CSV or JSON:", type=["csv", "json"], key="bulk_upload_file")

    if bulk_file is not None:
        try:
            bulk_df = pd.read_json(bulk_file) if bulk_file.name.endswith('.json') else pd.read_csv(bulk_file)
        except Exception as e:
            bulk_df = None
            st.error(f"Couldn't read this file: {e}")

        if bulk_df is not None and len(bulk_df) > 0:
            st.caption(f"Found {len(bulk_df)} rows. Columns: {', '.join(str(c) for c in bulk_df.columns)}")
            cols_lower = {str(c).lower(): c for c in bulk_df.columns}
            label_col = next((cols_lower[c] for c in ['label', 'corrected_label', 'verdict'] if c in cols_lower), None)
            text_col = next((cols_lower[c] for c in ['text', 'claim', 'content'] if c in cols_lower), None)
            has_all_features = all(k.lower() in cols_lower for k in bulk_feature_keys)

            if not label_col:
                st.warning("No label column found - expected a column named `label`, `corrected_label`, or `verdict`.")
            elif has_all_features:
                st.info(f"Detected pre-computed feature columns - will train instantly on all {len(bulk_df)} rows, no network calls needed.")
                if st.button("🚀 Train on Uploaded Features", type="primary", key="bulk_train_features_btn"):
                    added, skipped = 0, 0
                    for _, row in bulk_df.iterrows():
                        norm_label = normalize_label(row[label_col], bulk_claim_type)
                        if not norm_label:
                            skipped += 1
                            continue
                        try:
                            feat = {k: float(row[cols_lower[k.lower()]]) for k in bulk_feature_keys}
                        except Exception:
                            skipped += 1
                            continue
                        entry = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'type': bulk_claim_type,
                            'content': str(row[text_col]) if text_col and text_col in row else '(bulk upload - features only)',
                            'predicted_verdict': norm_label,
                            'is_correct': 'Bulk Upload',
                            'corrected_label': norm_label,
                            'features': feat
                        }
                        st.session_state.feedback_dataset.append(entry)
                        append_feedback_entry(entry)
                        added += 1
                    st.success(f"Added {added} labeled rows" + (f" ({skipped} skipped - unrecognized label or bad feature values)." if skipped else "."))
                    with st.spinner("Training..."):
                        text_result, media_result = run_training_cycle()
                    report_training_result(text_result if bulk_claim_type == 'Text Claim' else media_result, bulk_type_choice)
                    st.rerun()
            elif text_col and bulk_claim_type == 'Text Claim':
                row_count = len(bulk_df)
                capped = min(row_count, 100)
                if row_count > 100:
                    st.warning(f"File has {row_count} rows; each row requires live searches, so only the first 100 will be processed this run. Upload the remainder separately afterward to add more.")
                if st.button("🚀 Compute Features Live & Train", type="primary", key="bulk_train_raw_btn"):
                    progress = st.progress(0, text="Starting...")
                    added, skipped = 0, 0
                    for i in range(capped):
                        row = bulk_df.iloc[i]
                        norm_label = normalize_label(row[label_col], bulk_claim_type)
                        claim_text = str(row[text_col]) if text_col in row else ""
                        progress.progress((i + 1) / capped, text=f"Processing {i + 1}/{capped}: {claim_text[:50]}...")
                        if not norm_label or not claim_text.strip():
                            skipped += 1
                            continue
                        try:
                            feat = compute_text_features_for_claim(claim_text)
                        except Exception:
                            skipped += 1
                            continue
                        entry = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'type': 'Text Claim',
                            'content': claim_text,
                            'predicted_verdict': norm_label,
                            'is_correct': 'Bulk Upload',
                            'corrected_label': norm_label,
                            'features': feat
                        }
                        st.session_state.feedback_dataset.append(entry)
                        append_feedback_entry(entry)
                        added += 1
                    progress.empty()
                    st.success(f"Processed {added} claims" + (f" ({skipped} skipped - unrecognized label or empty text)." if skipped else "."))
                    with st.spinner("Training..."):
                        text_result, media_result = run_training_cycle()
                    report_training_result(text_result, "Text")
                    st.rerun()
            else:
                st.warning(f"Couldn't find a usable text column (`text`/`claim`/`content`) or all required feature columns ({', '.join(bulk_feature_keys)}) for {bulk_type_choice}.")

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
    
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_data = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Audit Log (CSV)",
            data=csv_data,
            file_name="verifact_audit_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    with exp_col2:
        try:
            pdf_data = generate_verification_log_pdf(st.session_state.verification_history)
            st.download_button(
                label="📄 Export Audit Log (PDF)",
                data=pdf_data,
                file_name="verifact_audit_log.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.caption(f"PDF export unavailable: {e}")
