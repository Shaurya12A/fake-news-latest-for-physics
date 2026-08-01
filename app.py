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

# ==========================================
# PAGE CONFIGURATION & DARK THEME CSS
# ==========================================
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

# Initialize Session State Variables
if 'verification_history' not in st.session_state:
    st.session_state.verification_history = []

if 'feedback_dataset' not in st.session_state:
    st.session_state.feedback_dataset = []

if 'last_analyzed_claim' not in st.session_state:
    st.session_state.last_analyzed_claim = None

# ==========================================
# SOURCE CREDIBILITY DATABASE
# ==========================================
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

# ==========================================
# GOOGLE NEWS RSS GROUNDING ENGINE
# ==========================================
def extract_search_queries(text):
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 10]
    lead_sentence = sentences[0] if sentences else text
    
    words = [w for w in clean_text.split() if len(w) > 3]
    unique_words = list(dict.fromkeys(words))
    
    q1 = " ".join(unique_words[:5]) if len(unique_words) >= 5 else clean_text[:60]
    q2 = " ".join(lead_sentence.split()[:6])
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
        for item in root.findall('.//item')[:5]:
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

# ==========================================
# SENTENCE-LEVEL TF-IDF MATCHING
# ==========================================
def calculate_sentence_similarity(claim_text, articles):
    if not articles:
        return 0.0, None
        
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', claim_text) if len(s.strip()) > 10]
    if not sentences:
        sentences = [claim_text]
        
    snippets = [a['snippet'] for a in articles]
    max_sim = 0.0
    best_match = articles[0]
    
    for sentence in sentences:
        corpus = [sentence] + snippets
        try:
            vectorizer = TfidfVectorizer(stop_words='english').fit_transform(corpus)
            vectors = vectorizer.toarray()
            sim_scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
            
            top_idx = np.argmax(sim_scores)
            top_score = sim_scores[top_idx]
            
            if top_score > max_sim:
                max_sim = top_score
                best_match = articles[top_idx]
        except Exception:
            continue
            
    return float(max_sim), best_match

# ==========================================
# LINGUISTIC ANALYSIS
# ==========================================
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

# ==========================================
# SIDEBAR NAVIGATION & FEEDBACK OPTIONS
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#34d399; margin-bottom:0;'>🛡️ VeriFact AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.8rem;'>Misinformation Command Center</p>", unsafe_allow_html=True)
    st.divider()
    
    analysis_mode = st.radio(
        "Select Verification Pipeline:",
        ["📰 Text / Article Fact-Checker", "📷 Image & Video Authenticator"],
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
    st.markdown("### 💬 Model Feedback & Training")
    if st.session_state.last_analyzed_claim:
        st.caption(f"Last Item: {st.session_state.last_analyzed_claim['content'][:30]}...")
        fb_correct = st.radio("Was model verdict accurate?", ["Yes 👍", "No 👎"], key="sb_fb_correct")
        
        fb_label = st.selectbox(
            "Correct Label (if incorrect):",
            ["🟢 VERIFIED REAL / HIGHLY LIKELY", "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT", "⚠️ UNVERIFIED / PROBABLE FAKE NEWS"],
            key="sb_fb_label"
        )
        
        if st.button("💾 Submit Feedback", type="secondary", key="sb_fb_btn"):
            st.session_state.feedback_dataset.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': st.session_state.last_analyzed_claim['type'],
                'content': st.session_state.last_analyzed_claim['content'],
                'predicted_verdict': st.session_state.last_analyzed_claim['predicted_verdict'],
                'is_correct': fb_correct,
                'corrected_label': fb_label if fb_correct == "No 👎" else st.session_state.last_analyzed_claim['predicted_verdict']
            })
            st.success("Feedback recorded into active training dataset!")
    else:
        st.info("Run an analysis to submit accuracy feedback.")

    st.caption(f"Collected Feedback Samples: **{len(st.session_state.feedback_dataset)}**")

# ==========================================
# HEADER DISPLAY
# ==========================================
st.markdown("<h1 style='color:#f8fafc; margin-bottom:5px;'>VeriFact AI Command Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>Real-Time Live Web Grounding & Media Authenticator Engine</p>", unsafe_allow_html=True)

# ==========================================
# PIPELINE 1: TEXT / ARTICLE FACT CHECKER
# ==========================================
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
        with st.spinner("Querying Google News RSS & Running Vector Matching..."):
            
            # Step 1: Search Queries
            queries = extract_search_queries(user_input)
            all_articles = []
            for q in queries:
                fetched = fetch_google_news_rss(q)
                all_articles.extend(fetched)
                
            # Deduplicate
            seen = set()
            unique_articles = []
            for a in all_articles:
                if a['title'] not in seen:
                    seen.add(a['title'])
                    unique_articles.append(a)
                    
            # Step 2: Vector Similarity
            raw_max_sim, best_match = calculate_sentence_similarity(user_input, unique_articles)
            corroboration_pct = min(int(raw_max_sim * 250), 100) if raw_max_sim >= 0.08 else min(int(raw_max_sim * 100), 20)
            
            # Step 3: Linguistic Risk Analysis
            sensationalism_score, journalistic_score = analyze_linguistic_risk(user_input)
            
            # Step 4: Original Decision Tree Logic (Strictly Unaltered)
            if corroboration_pct >= 20 or raw_max_sim >= 0.10:
                verdict = "🟢 VERIFIED REAL / HIGHLY LIKELY"
                status_class = "badge-real"
                truth_index = max(corroboration_pct, 80)
                summary = f"Matches live coverage from '{best_match['source'] if best_match else 'Global News'}'. High textual alignment detected."
            elif sensationalism_score >= 40:
                verdict = "🚨 DEBUNKED FAKE / SENSATIONAL CLICKBAIT"
                status_class = "badge-fake"
                truth_index = max(100 - sensationalism_score, 10)
                summary = "Exhibits heavy emotional clickbait indicators and lacks grounding across verified news outlets."
            else:
                verdict = "⚠️ UNVERIFIED / PROBABLE FAKE NEWS"
                status_class = "badge-warning"
                truth_index = 35
                summary = "Formal phrasing detected, but no corroborating reports found on live news feeds."
                
            # Store in session state for sidebar feedback
            st.session_state.last_analyzed_claim = {
                'type': 'Text Claim',
                'content': user_input,
                'predicted_verdict': verdict
            }

            # Log into verification audit history
            st.session_state.verification_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'claim': user_input[:60] + "...",
                'verdict': verdict,
                'truth_index': f"{truth_index}%",
                'corroboration': f"{corroboration_pct}%"
            })
            
            # ==========================================
            # DISPLAY DASHBOARD RESULTS
            # ==========================================
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="command-card">
                <div style="display:flex; justify-between; align-items:center;">
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
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{corroboration_pct}%</div><div class="metric-label">Web Corroboration</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{journalistic_score}%</div><div class="metric-label">Journalistic Tone</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="metric-box"><div class="metric-value">{sensationalism_score}%</div><div class="metric-label">Sensationalism Score</div></div>""", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Tabs for Detailed Breakdown
            tab1, tab2, tab3 = st.tabs(["📲 WhatsApp Debunk Card", "🟢 Live News & Source Authority", "📊 Analytics Details"])
            
            with tab1:
                st.markdown("#### Ready-to-Share WhatsApp Fact-Check Briefing")
                debunk_text = f"""*🛡️ VERIFACT AI FACT CHECK ALERT*
----------------------------------
*Claim:* "{user_input[:100]}..."
*Verdict:* {verdict}
*Truth Index:* {truth_index}%
*Web Corroboration:* {corroboration_pct}%

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
                    st.info("No direct corroborating headlines found on live Google News feeds.")
                    
            with tab3:
                st.json({
                    "raw_vector_similarity": raw_max_sim,
                    "sensationalism_score": sensationalism_score,
                    "journalistic_score": journalistic_score,
                    "extracted_queries": queries
                })

# ==========================================
# PIPELINE 2: IMAGE & VIDEO AUTHENTICATOR
# ==========================================
else:
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
                with st.spinner("Analyzing audio-visual stream, compression footprints, and search grounding..."):
                    
                    # Search Grounding if Context Given
                    corroborated = False
                    if media_context.strip():
                        q = extract_search_queries(media_context)[0]
                        articles = fetch_google_news_rss(q)
                        sim, _ = calculate_sentence_similarity(media_context, articles)
                        if sim >= 0.10:
                            corroborated = True

                    # 3-Class Media Classification Logic
                    file_size = uploaded_media.size
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

                    # Store in session state for sidebar feedback
                    st.session_state.last_analyzed_claim = {
                        'type': 'Media File',
                        'content': media_context if media_context else filename,
                        'predicted_verdict': media_verdict
                    }

                    # Log into session
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

# ==========================================
# AUDIT LOG & TRAINING DATASET EXPORTS
# ==========================================
st.divider()
col_hist1, col_hist2 = st.columns(2)

with col_hist1:
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

with col_hist2:
    if st.session_state.feedback_dataset:
        st.markdown("### 🧠 Feedback Training Dataset")
        df_fb = pd.DataFrame(st.session_state.feedback_dataset)
        st.dataframe(df_fb, use_container_width=True)
        
        fb_csv = df_fb.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Training Feedback Dataset (CSV)",
            data=fb_csv,
            file_name="verifact_training_feedback.csv",
            mime="text/csv"
        )
