import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import unicodedata
import joblib
import os
from pathlib import Path
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    .command-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
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
    return _PDF_EMOJI_PATTERN.sub('', str(text)).strip()

def _pdf_verdict_color(verdict_str):
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
        "(live news/fact-check search and evidence-based text analysis). "
        "They are decision support, not a certified fact-check - always verify high-stakes claims "
        "through a professional fact-checking organization.",
        disclaimer_style
    ))

    doc.build(elements, onFirstPage=_pdf_header_footer, onLaterPages=_pdf_header_footer)
    buf.seek(0)
    return buf.getvalue()

if 'verification_history' not in st.session_state:
    st.session_state.verification_history = []
QUERY_STOP_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'in', 'to', 'for', 'of', 'with',
    'that', 'this', 'it', 'from', 'by', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has',
    'had', 'will', 'would', 'says', 'said', 'according', 'announced', 'new', 'news', 'breaking'
}

def unicode_words(text):
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
        if (w.isupper() and len(w) >= 2) or re.search(r'\d', w) or (w[0].isupper() and len(w) >= 3):
            entities.append(w)
        elif len(lw) > 3:
            general_words.append(lw)

    entities = list(dict.fromkeys(entities))
    general_words = list(dict.fromkeys(general_words))

    remaining_slots = max(0, 5 - len(entities[:4]))
    q1_terms = entities[:4] + general_words[:remaining_slots]
    q1 = " ".join(q1_terms) if q1_terms else text[:60]

    q2 = " ".join(lead_sentence.split()[:8])

    return [q1, q2]

LANGUAGE_OPTIONS = {
    'English': {'hl': 'en-IN', 'gl': 'IN', 'ceid': 'IN:en'},
    'Hindi': {'hl': 'hi-IN', 'gl': 'IN', 'ceid': 'IN:hi'},
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
        articles = fetch_duckduckgo_news(query)
    return articles

FACTCHECK_RSS_FEEDS = [
    ("PIB Fact Check", "https://pib.gov.in/PressReleseDetail.aspx?rss=1"),
    ("AltNews", "https://www.altnews.in/feed/"),
    ("BOOM Live", "https://www.boomlive.in/feed"),
    ("Factly", "https://factly.in/feed/"),
]

def fetch_factcheck_rss(query):
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
    general = fetch_live_news_with_fallback(query, lang=lang)
    factcheck = fetch_factcheck_rss(query)
    return general + factcheck

NLI_LABELS = ['contradiction', 'entailment', 'neutral']

@st.cache_resource(show_spinner="Loading local NLI model (one-time)...")
def get_nli_model():
    if not HAS_NLI:
        return None
    try:
        return CrossEncoder('cross-encoder/nli-deberta-v3-xsmall')
    except Exception:
        return None

def compute_nli_contradiction_score(claim_text, article_snippet):
    if not article_snippet or not article_snippet.strip():
        return None
    model = get_nli_model()
    if model is None:
        return None
    try:
        logits = model.predict([(article_snippet[:512], claim_text[:300])])
        row = np.asarray(logits[0], dtype=np.float64)
        exp = np.exp(row - row.max())
        probs = exp / exp.sum()
        contradiction_idx = NLI_LABELS.index('contradiction')
        return float(probs[contradiction_idx])
    except Exception:
        return None

def get_debunk_assessment(claim_text, article):
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
            matched_words = [w for w in main_words if w in snippet_text]
            overlap_ratio = len(matched_words) / len(main_words) if main_words else 0.0

            try:
                vectorizer = TfidfVectorizer(stop_words='english').fit_transform([sentence, article['snippet']])
                vectors = vectorizer.toarray()
                sim_score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
            except Exception:
                sim_score = 0.0

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

@st.cache_resource
def load_trained_model():
    try:
        vectorizer = joblib.load("vectorizer.pkl")
        model = joblib.load("model (2).pkl")
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)


trained_model, trained_vectorizer, model_load_error = load_trained_model()


DATASET_DIR = Path("datasets")
FEVER_LOCAL_FILE = DATASET_DIR / "fever_train.parquet"
LIAR_LOCAL_FILES = {
    "train": DATASET_DIR / "liar_train.tsv",
    "valid": DATASET_DIR / "liar_valid.tsv",
    "test": DATASET_DIR / "liar_test.tsv",
}


def model_clean_text(text):
    text = str(text)
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?[A-Z]{2,}\s*\(AP\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()


def find_local_file(primary_path, patterns):
    if primary_path.exists():
        return primary_path
    if DATASET_DIR.exists():
        for pattern in patterns:
            matches = sorted(DATASET_DIR.glob(pattern))
            if matches:
                return matches[0]
    return None


def load_local_training_data():
    fever_file = find_local_file(
        FEVER_LOCAL_FILE,
        ["*fever*.parquet", "*fever*.csv"]
    )
    liar_train = find_local_file(LIAR_LOCAL_FILES["train"], ["*liar*train*.tsv"])
    liar_valid = find_local_file(LIAR_LOCAL_FILES["valid"], ["*liar*valid*.tsv"])
    liar_test = find_local_file(LIAR_LOCAL_FILES["test"], ["*liar*test*.tsv"])

    missing = []
    if fever_file is None:
        missing.append("FEVER training file")
    if liar_train is None:
        missing.append("LIAR train.tsv")
    if liar_valid is None:
        missing.append("LIAR valid.tsv")
    if liar_test is None:
        missing.append("LIAR test.tsv")
    if missing:
        raise FileNotFoundError(
            "Missing local dataset files: " + ", ".join(missing) +
            ". Put the downloaded files inside the 'datasets' folder."
        )

    if fever_file.suffix.lower() == ".parquet":
        fever_df = pd.read_parquet(fever_file)
    else:
        fever_df = pd.read_csv(fever_file)

    fever_df["label"] = fever_df["label"].map({
        "SUPPORTS": 0,
        "REFUTES": 1,
        "NOT ENOUGH INFO": 1,
    })
    fever_df = fever_df[["claim", "label"]].rename(columns={"claim": "text"})

    liar_columns = [
        "id", "label", "statement", "subject", "speaker", "job",
        "state", "party", "barely_true", "false", "half_true",
        "mostly_true", "pants_fire", "context"
    ]

    liar_frames = []
    for file_path in [liar_train, liar_valid, liar_test]:
        liar_frames.append(pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=liar_columns
        ))
    liar_df = pd.concat(liar_frames, ignore_index=True)

    liar_df["label"] = liar_df["label"].map({
        "true": 0,
        "mostly-true": 0,
        "false": 1,
        "pants-fire": 1,
        "half-true": 1,
        "barely-true": 1,
    })
    liar_df = liar_df[["statement", "label"]].rename(
        columns={"statement": "text"}
    )

    data = pd.concat([fever_df, liar_df], ignore_index=True).dropna()
    data["text"] = data["text"].apply(model_clean_text)
    data = data[data["text"].str.len() > 0]
    data = data.drop_duplicates(subset=["text"])
    return data


def train_local_model():
    data = load_local_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.20,
        stratify=data["label"],
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=250000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
        solver="liblinear",
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    test_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, test_pred)

    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(model, "model (1).pkl")

    return model, vectorizer, accuracy, len(data), y_test, test_pred


with st.sidebar:
    st.markdown("<h2 style='color:#34d399; margin-bottom:0;'>🛡️ VeriFact AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.8rem;'>Misinformation Command Center</p>", unsafe_allow_html=True)
    st.divider()
    
    analysis_mode = st.radio(
        "Analysis Mode",
        [
            "📰 Text / Article Fact-Checker",
            "🤖 Trained Model Predictor",
            "🛠️ Train / Update Model"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("### ⚡ Test Claim Benchmarks")
    if st.button("🔴 Fake: Plastic Currency Rumor"):
        st.session_state.test_claim = "The Reserve Bank of India has announced that all paper currency notes will be replaced with plastic bank notes next month."
    if st.button("🟢 Real: ISRO Gaganyaan Engine"):
        st.session_state.test_claim = "ISRO successfully completed core stage engine testing for the Gaganyaan human spaceflight mission."
    if st.button("🚨 Clickbait: 5G Scalar Waves"):
        st.session_state.test_claim = "BREAKING URGENT: Secret government plot leaked as 5G cell towers emit scalar frequencies!"



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

    text_mode_choice = "Rule-based engine"
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        run_btn = st.button("🔍 Run Deep Fact Check", type="primary", use_container_width=True)
        
    if run_btn and user_input.strip():
        with st.spinner(f"Analyzing claim nouns/entities, querying live news ({search_lang}) + fact-check feeds..."):
            queries = extract_search_queries(user_input)
            all_articles = []
            for q in queries:
                fetched = fetch_all_sources(q, lang=search_lang)
                all_articles.extend(fetched)
                
            seen = set()
            unique_articles = []
            for a in all_articles:
                if a['title'] not in seen:
                    seen.add(a['title'])
                    unique_articles.append(a)
                    
            overlap_ratio, raw_max_sim, best_match, matched_words, total_words = calculate_entity_and_vector_match(user_input, unique_articles)
            corroboration_pct = int(overlap_ratio * 100)
            
            sensationalism_score, journalistic_score = analyze_linguistic_risk(user_input)

            debunk_flag, nli_score, debunk_method = get_debunk_assessment(user_input, best_match)
            is_factcheck_source = bool(best_match and best_match.get('source_type') == 'factcheck')

            known_hoax_flag = matches_known_hoax(user_input)
            
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

            final_verdict = verdict
            final_truth_index = truth_index

            st.session_state.verification_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'claim': user_input[:60] + "...",
                'verdict': final_verdict,
                'truth_index': f"{final_truth_index}%",
                'corroboration': f"{corroboration_pct}%"
            })
            
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

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{final_truth_index}%</div><div class="metric-label">Truth Index</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{corroboration_pct}%</div><div class="metric-label">Single Article Match</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{journalistic_score}%</div><div class="metric-label">Journalistic Tone</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{sensationalism_score}%</div><div class="metric-label">Sensationalism Score</div></div>""", unsafe_allow_html=True)

            with st.expander("🔬 Evidence analysis details"):
                st.markdown(f"- **NLI contradiction score:** {f'{nli_score:.2f}' if nli_score is not None else 'N/A (sentence-transformers not installed, or no matched article)'}")
                st.markdown(f"- **Fact-check source matched:** {'Yes' if is_factcheck_source else 'No'}")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<h4>Top Matching News Articles Found</h4>")
            if unique_articles:
                for art in unique_articles[:4]:
                    is_fc = art.get('source_type') == 'factcheck'
                    st.markdown(f"""
                    <div style="background:rgba(15,23,42,0.6); padding:12px; border-radius:8px; margin-bottom:8px; border:1px solid rgba(255,255,255,0.05);">
                        <a href="{art['link']}" target="_blank" style="color:#38bdf8; font-weight:bold; text-decoration:none;">{art['title']}</a><br>
                        <span style="color:#94a3b8; font-size:0.8rem;">Source: {art['source']}</span>
                        {' <span style="color:#a78bfa; font-weight:bold; font-size:0.8rem;"> | 🔍 Dedicated Fact-Check Source</span>' if is_fc else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No direct corroborating headlines found on live news feeds.")

elif analysis_mode == "🛠️ Train / Update Model":
    st.markdown("<h3>🛠️ Train / Update Fake-News Model</h3>")
    st.markdown(
        "<p style='color:#94a3b8;'>Train the same TF-IDF + Logistic Regression model using downloaded FEVER and LIAR datasets stored locally in the <b>datasets</b> folder.</p>",
        unsafe_allow_html=True
    )

    st.info(
        "Dataset sources: FEVER — Sheffield NLP; LIAR — William Yang Wang / UCSB. "
        "The app uses local downloaded copies and does not download datasets during training."
    )

    st.markdown("<strong>Expected folder structure:</strong>")
    st.code(
        "datasets/\n"
        "├── fever_train.parquet\n"
        "├── liar_train.tsv\n"
        "├── liar_valid.tsv\n"
        "└── liar_test.tsv"
    )

    if st.button("🚀 Train / Update Model", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading local datasets, vectorizing text and training the model..."):
                new_model, new_vectorizer, accuracy, sample_count, y_test, test_pred = train_local_model()

            trained_model = new_model
            trained_vectorizer = new_vectorizer

            st.success("Model training completed and model files were saved.")
            st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
            st.caption(f"Training samples after cleaning/deduplication: {sample_count}")

            st.markdown("<h4>Classification Report</h4>")
            st.code(classification_report(
                y_test,
                test_pred,
                target_names=["REAL", "FAKE"]
            ))

            st.markdown("<h4>Saved Model Files</h4>")
            st.code("tfidf_vectorizer.pkl\nmodel (1).pkl")

        except Exception as e:
            st.error(f"Training failed: {e}")

elif analysis_mode == "🤖 Trained Model Predictor":
    st.markdown("<h3>🤖 Trained Model Predictor</h3>")
    st.markdown(
        "<p style='color:#94a3b8;'>TF-IDF + Logistic Regression prediction using your trained fake-news model. This mode does not perform live web searches.</p>",
        unsafe_allow_html=True
    )

    model_claim = st.text_area(
        "Enter news text or a claim to predict:",
        height=180,
        placeholder="Paste a headline, claim, or news article here..."
    )

    if trained_model is None or trained_vectorizer is None:
        st.error(
            "The trained model could not be loaded. Make sure these files are in the same folder as app.py: "
            "tfidf_vectorizer.pkl and model (1).pkl"
            + (f"\n\nLoading error: {model_load_error}" if model_load_error else "")
        )
    else:
        if st.button("🤖 Analyze News", type="primary", use_container_width=True):
            if not model_claim.strip():
                st.warning("Please enter some text before analyzing.")
            else:
                cleaned = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', model_claim, flags=re.IGNORECASE)
                cleaned = re.sub(r'^.*?[A-Z]{2,}\s*\(AP\)\s*-\s*', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
                cleaned = re.sub(r'<.*?>', '', cleaned)
                cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)
                cleaned = cleaned.lower().strip()

                if not cleaned:
                    st.warning("The entered text became empty after preprocessing. Please enter a longer news claim or article.")
                else:
                    vec_input = trained_vectorizer.transform([cleaned])
                    prediction = trained_model.predict(vec_input)[0]
                    proba = trained_model.predict_proba(vec_input)[0]

                    classes = list(trained_model.classes_)
                    probability_by_class = {cls: float(prob) for cls, prob in zip(classes, proba)}

                    real_prob = probability_by_class.get(0, probability_by_class.get('REAL', 0.0))
                    fake_prob = probability_by_class.get(1, probability_by_class.get('FAKE', 0.0))
                    top_confidence = max(real_prob, fake_prob)
                    truth_index = int(round(real_prob * 100))

                    UNCERTAIN_THRESHOLD = 0.60

                    if top_confidence < UNCERTAIN_THRESHOLD:
                        status_class = "badge-warning"
                        display_verdict = "⚠️ UNCERTAIN"
                        summary = (
                            f"The model does not have enough confidence to classify this as TRUE or FALSE. "
                            f"Its highest prediction confidence is {top_confidence * 100:.1f}%."
                        )
                    elif prediction == 1:
                        status_class = "badge-fake"
                        display_verdict = "🚨 FALSE"
                        summary = (
                            f"The trained model predicts this news as FALSE with "
                            f"{fake_prob * 100:.1f}% confidence."
                        )
                    else:
                        status_class = "badge-real"
                        display_verdict = "🟢 TRUE"
                        summary = (
                            f"The trained model predicts this news as TRUE with "
                            f"{real_prob * 100:.1f}% confidence."
                        )

                    st.markdown(f"""
                    <div style="
                        background:rgba(30,41,59,0.82);
                        padding:36px;
                        border-radius:22px;
                        margin-top:18px;
                        border:1px solid rgba(148,163,184,0.16);
                        box-shadow:0 10px 30px rgba(0,0,0,0.12);
                    ">
                        <span class="{status_class}">{display_verdict}</span>
                        <h1 style="color:#f8fafc; font-size:3rem; margin:35px 0 28px 0;">
                            Truth Index: {truth_index}%
                        </h1>
                        <p style="color:#cbd5e1; font-size:1.05rem; line-height:1.8; margin:0;">
                            {summary}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.caption(
                        "Truth Index represents the model's estimated probability that the text is REAL. "
                        "UNCERTAIN is shown when neither REAL nor FAKE reaches 60% confidence."
                    )
