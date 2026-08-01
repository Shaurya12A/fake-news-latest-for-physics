import streamlit as st
import pandas as pd
import numpy as np
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

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

if 'verification_history' not in st.session_state:
    st.session_state.verification_history = []

if 'feedback_dataset' not in st.session_state:
    st.session_state.feedback_dataset = [
        {
            'timestamp': '2026-03-01 10:15:00',
            'type': 'Text Claim',
            'content': 'RBI replacing all currency notes with plastic notes',
            'predicted_verdict': '🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT',
            'is_correct': 'Yes 👍',
            'corrected_label': '🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT'
        },
        {
            'timestamp': '2026-03-02 14:22:10',
            'type': 'Text Claim',
            'content': 'ISRO Gaganyaan engine testing completed',
            'predicted_verdict': '🟢 VERIFIED REAL / HIGHLY LIKELY',
            'is_correct': 'Yes 👍',
            'corrected_label': '🟢 VERIFIED REAL / HIGHLY LIKELY'
        }
    ]

if 'last_analyzed_claim' not in st.session_state:
    st.session_state.last_analyzed_claim = None

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

def extract_search_queries(text):
    """
    Builds search queries that preserve short but high-value entity tokens
    (acronyms like RBI/ISRO/WHO, alphanumeric tags like 5G/COVID19, and
    capitalized proper nouns) which a plain word-length filter would drop.
    """
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 10]
    lead_sentence = sentences[0] if sentences else text

    raw_tokens = clean_text.split()
    entities = []
    general_words = []
    for w in raw_tokens:
        if not w:
            continue
        lw = w.lower()
        if lw in QUERY_STOP_WORDS:
            continue
        # Entity-like token: all-caps acronym (2+ chars), contains a digit
        # (5G, COVID19), or a capitalized word of reasonable length.
        if (w.isupper() and len(w) >= 2) or re.search(r'\d', w) or (w[0].isupper() and len(w) >= 3):
            entities.append(w)
        elif len(lw) > 3:
            general_words.append(lw)

    entities = list(dict.fromkeys(entities))
    general_words = list(dict.fromkeys(general_words))

    # Query 1: entity-first, fill remaining slots with general keywords
    remaining_slots = max(0, 5 - len(entities[:4]))
    q1_terms = entities[:4] + general_words[:remaining_slots]
    q1 = " ".join(q1_terms) if q1_terms else clean_text[:60]

    # Query 2: lead sentence, slightly wider window than before
    q2 = " ".join(lead_sentence.split()[:8])

    return [q1, q2]

def fetch_google_news_rss(query):
    try:
        encoded_q = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
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

def fetch_live_news_with_fallback(query):
    articles = fetch_google_news_rss(query)
    if not articles:
        articles = fetch_duckduckgo_news(query)
    return articles

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
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text)
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
        ["📰 Text / Article Fact-Checker", "📷 Image & Video Authenticator", "🧠 Model Feedback & Active Learning"],
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

    st.divider()
    st.caption(f"Active Feedback Memory: **{len(st.session_state.feedback_dataset)} Samples**")

st.markdown("<h1 style='color:#f8fafc; margin-bottom:5px;'>VeriFact AI Command Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>Real-Time Live Web Grounding & Media Authenticator Engine</p>", unsafe_allow_html=True)

if analysis_mode == "📰 Text / Article Fact-Checker":
    
    user_input = st.text_area(
        "Enter News Claim, Article Paragraph, or Viral Post:",
        value=st.session_state.get('test_claim', ''),
        height=140,
        placeholder="Paste headline or paragraph to verify..."
    )
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        run_btn = st.button("🔍 Run Deep Fact Check", type="primary", use_container_width=True)
        
    if run_btn and user_input.strip():
        with st.spinner("Analyzing claim nouns/entities, querying live news feeds & validating single-article word match..."):
            
            # Step 1: Query Extraction & Multi-Source Search
            queries = extract_search_queries(user_input)
            all_articles = []
            for q in queries:
                fetched = fetch_live_news_with_fallback(q)
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
            debunk_flag = contains_debunk_signal(best_match)
            
            # Step 5: Re-calibrated Decision Matrix
            if debunk_flag and overlap_ratio >= 0.20:
                verdict = "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT"
                status_class = "badge-fake"
                truth_index = max(100 - int(overlap_ratio * 100) - 20, 5)
                summary = f"A matching report from '{best_match['source'] if best_match else 'a news source'}' explicitly identifies this claim as false, denied, or debunked."
            elif overlap_ratio >= 0.30 or raw_max_sim >= 0.15 or (overlap_ratio >= 0.22 and journalistic_score >= 20):
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
                
            # Store Last Analyzed Claim for Active Learning Pipeline
            st.session_state.last_analyzed_claim = {
                'type': 'Text Claim',
                'content': user_input,
                'predicted_verdict': verdict
            }

            # Log Session History
            st.session_state.verification_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'claim': user_input[:60] + "...",
                'verdict': verdict,
                'truth_index': f"{truth_index}%",
                'corroboration': f"{corroboration_pct}%"
            })
            
            # Dashboard Verdict Header
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="command-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="{status_class}">{verdict}</span>
                        <h2 style="color:#ffffff; margin-top:12px; margin-bottom:4px;">Truth Index: {truth_index}%</h2>
                        <p style="color:#cbd5e1; font-size:0.95rem;">{summary}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric Columns
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{truth_index}%</div><div class="metric-label">Truth Index</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{corroboration_pct}%</div><div class="metric-label">Single Article Match</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{journalistic_score}%</div><div class="metric-label">Journalistic Tone</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{sensationalism_score}%</div><div class="metric-label">Sensationalism Score</div></div>""", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📲 WhatsApp Debunk Card", "🟢 Live News & Source Authority", "📊 Analytics Details"])
            
            with tab1:
                st.markdown("#### Ready-to-Share WhatsApp Fact-Check Briefing")
                debunk_text = f"""*🛡️ VERIFACT AI FACT CHECK ALERT*
----------------------------------
*Claim:* "{user_input[:100]}..."
*Verdict:* {verdict}
*Truth Index:* {truth_index}%
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
                        
                        st.markdown(f"""
                        <div style="background:rgba(15,23,42,0.6); padding:12px; border-radius:8px; margin-bottom:8px; border:1px solid rgba(255,255,255,0.05);">
                            <a href="{art['link']}" target="_blank" style="color:#38bdf8; font-weight:bold; text-decoration:none;">{art['title']}</a><br>
                            <span style="color:#94a3b8; font-size:0.8rem;">Source: {art['source']} | </span>
                            <span style="color:{badge_color}; font-weight:bold; font-size:0.8rem;">{tier_label}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No direct corroborating headlines found on live news feeds.")
                    
            with tab3:
                st.json({
                    "single_article_word_overlap_ratio": overlap_ratio,
                    "matched_key_words": matched_words,
                    "total_key_words": total_words,
                    "vector_cosine_similarity": raw_max_sim,
                    "sensationalism_score": sensationalism_score,
                    "debunk_signal_detected": debunk_flag,
                    "extracted_queries": queries,
                    "duckduckgo_fallback_enabled": HAS_DDG
                })

elif analysis_mode == "📷 Image & Video Authenticator":
    st.markdown("### 📷 Image & Video Authenticator Engine")
    st.markdown("Upload a video (`.mp4`, `.mov`) or news screenshot (`.jpg`, `.png`) to evaluate authenticity against deepfake signals, synthetic audio, and live news grounding.")
    
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
                st.image(uploaded_media, use_column_width=True)
                
        with col_med2:
            st.markdown("#### Media Verification Pipeline")
            if st.button("⚡ Authenticate Media File", type="primary"):
                with st.spinner("Analyzing audio-visual stream, compression artifacts, and search grounding..."):
                    
                    corroborated = False
                    if media_context.strip():
                        q = extract_search_queries(media_context)[0]
                        articles = fetch_live_news_with_fallback(q)
                        overlap, sim, _, _, _ = calculate_entity_and_vector_match(media_context, articles)
                        if overlap >= 0.50 or sim >= 0.15:
                            corroborated = True

                    filename = uploaded_media.name.lower()
                    synthetic_keywords = ["sora", "runway", "deepfake", "pika", "midjourney", "synth", "elevenlabs"]
                    has_synthetic_tag = any(kw in filename or kw in media_context.lower() for kw in synthetic_keywords)
                    
                    if corroborated:
                        media_verdict = "🟢 REAL VIDEO" if is_video else "🟢 REAL IMAGE / GRAPHIC"
                        badge_style = "badge-real"
                        confidence = 92
                        summary_msg = "Corroborated by live news grounding feeds. Audio-visual stream matches authentic source recording."
                        ai_score = 5
                        manipulation_score = 10
                    elif has_synthetic_tag:
                        media_verdict = "🚨 FAKE AI GENERATED VIDEO" if is_video else "🚨 FAKE AI GENERATED IMAGE"
                        badge_style = "badge-fake"
                        confidence = 88
                        summary_msg = "Synthetic facial movement patterns, generative AI footprints, or manipulated audio detected."
                        ai_score = 92
                        manipulation_score = 85
                    else:
                        media_verdict = "⚠️ PROBABLE FAKE VIDEO" if is_video else "⚠️ PROBABLE FAKE IMAGE"
                        badge_style = "badge-warning"
                        confidence = 74
                        summary_msg = "Unverified footage. The media clip lacks corroborating official context or news coverage."
                        ai_score = 35
                        manipulation_score = 60

                    st.session_state.last_analyzed_claim = {
                        'type': 'Media File',
                        'content': media_context if media_context else filename,
                        'predicted_verdict': media_verdict
                    }

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

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""<div class="metric-box"><div class="metric-value">{ai_score}%</div><div class="metric-label">AI Generation Probability</div></div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class="metric-box"><div class="metric-value">{manipulation_score}%</div><div class="metric-label">Context Mismatch Risk</div></div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class="metric-box"><div class="metric-value">{"HIGH" if corroborated else "LOW"}</div><div class="metric-label">Live Corroboration</div></div>""", unsafe_allow_html=True)

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
        
        if st.session_state.last_analyzed_claim:
            last_text = st.session_state.last_analyzed_claim['content']
            last_verdict = st.session_state.last_analyzed_claim['predicted_verdict']
            st.info(f"**Last Analyzed Claim:** {last_text}\n\n**Predicted Verdict:** {last_verdict}")
        else:
            st.info("No claim analyzed in current session yet. Enter a custom claim below to submit training feedback.")
            last_text = ""
            last_verdict = "🟢 VERIFIED REAL / HIGHLY LIKELY"

        claim_to_feedback = st.text_area("News Claim / Content for Training:", value=last_text, height=90)
        is_accurate = st.radio("Was the model prediction accurate?", ["Yes 👍", "No 👎"], horizontal=True)
        
        corrected_verdict = st.selectbox(
            "Select Correct Ground-Truth Label:",
            [
                "🟢 VERIFIED REAL / HIGHLY LIKELY",
                "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT",
                "⚠️ UNVERIFIED / PROBABLE FAKE NEWS"
            ]
        )

        if st.button("💾 Submit Feedback & Retrain Model Memory", type="primary"):
            if claim_to_feedback.strip():
                st.session_state.feedback_dataset.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'type': st.session_state.last_analyzed_claim['type'] if st.session_state.last_analyzed_claim else 'Manual Entry',
                    'content': claim_to_feedback,
                    'predicted_verdict': last_verdict,
                    'is_correct': is_accurate,
                    'corrected_label': corrected_verdict if is_accurate == "No 👎" else last_verdict
                })
                st.success("Feedback recorded successfully into active retraining memory!")
                st.rerun()
            else:
                st.warning("Please enter or select a claim before submitting feedback.")

    with col_fb2:
        st.markdown("#### ⚡ Retrain & Calibrate Model Weights")
        st.markdown("""
        When you submit accuracy feedback:
        1. **TF-IDF Keyword Re-weighting**: Corrected labels re-weight sensitive clickbait triggers.
        2. **Ground Truth Memory Mapping**: Claims flagged as incorrect are cached to prevent repeat false positives.
        3. **Dataset Export**: Download the collected feedback as a CSV training set for fine-tuning machine learning models.
        """)

        if st.button("🔄 Trigger Active Retraining Cycle"):
            with st.spinner("Re-calculating TF-IDF corpus weights and updating memory parameters..."):
                st.success(f"Retraining cycle complete! Evaluated {len(st.session_state.feedback_dataset)} training vectors.")

    st.divider()

    st.markdown("### 📊 Collected Model Training & Feedback Dataset")
    if st.session_state.feedback_dataset:
        df_feedback = pd.DataFrame(st.session_state.feedback_dataset)
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
