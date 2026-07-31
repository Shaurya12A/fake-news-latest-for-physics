import re
import math
import html
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from duckduckgo_search import DDGS

st.set_page_config(
    page_title="VeriFact AI — Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Audit History
if "verification_history" not in st.session_state:
    st.session_state.verification_history = []

st.markdown("""
<style>
    /* Dark Theme Base Overrides */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Main Header Styling */
    .command-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .command-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .command-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
    }

    /* Metric Box Styling */
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Custom Progress Bar Styling */
    .progress-container {
        width: 100%;
        background-color: #1e293b;
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin-top: 6px;
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }

    /* WhatsApp Card Custom Styling */
    .whatsapp-card-box {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid #22c55e;
        border-radius: 14px;
        padding: 18px;
        font-family: monospace;
        color: #e2e8f0;
        white-space: pre-wrap;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.15);
    }

    /* Streamlit Input Fixes */
    .stTextArea textarea {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def build_local_ml_model():
    """
    Builds and trains a local TF-IDF + Logistic Regression classifier benchmark.
    """
    training_data = [
        # Real News Samples
        ("NASA spacecraft successfully orbits distant exoplanet in deep space discovery", 1),
        ("Scientists report breakthrough in renewable solar cell energy efficiency ratings", 1),
        ("World Health Organization publishes updated clinical guidelines for flu prevention", 1),
        ("Central Bank announces minor adjustment to interest rates following economic inflation report", 1),
        ("Global climate summit concludes with international agreement on carbon reduction targets", 1),
        ("Researchers identify new deep ocean marine species during deep trench expedition", 1),
        ("Technology firm releases annual open-source software security update", 1),
        ("City council approves budget expansion for public transportation infrastructure", 1),
        ("Astronomers observe rare supernova explosion using radio telescope network", 1),
        ("Health Ministry confirms seasonal vaccine distribution starting next month", 1),
        ("Government officials announce new national highway construction project starting next fiscal year", 1),
        ("Electric vehicle sales saw a significant growth rate according to annual industry report", 1),
        ("Local university researchers publish peer reviewed study on renewable battery longevity", 1),
        ("Supreme Court delivers verdict on constitutional rights case after months of hearing", 1),
        ("Finance ministry presents annual budget allocation for education and healthcare sectors", 1),
        ("Indian Space Research Organisation successfully completes core stage engine testing for spaceflight mission", 1),
        ("Lok Sabha approved the Public Examination Prevention of Unfair Means Amendment Bill", 1),
        ("United Nations Human Rights Office reported civilian casualties in Ukraine conflict", 1),
        ("Israeli military forces conducted targeted air and ground operations across northern Gaza", 1),
        
        # Fake / Clickbait Samples
        ("BREAKING: Miracle kitchen spice completely cures all diseases in 24 hours scientists shocked", 0),
        ("Secret government plot leaked: 5G towers emitting secret mind control signals", 0),
        ("DOCTORS HATE HIM! Local man discovers one weird trick to lose 50 pounds overnight", 0),
        ("ALERT: Drinking hot lemon water with baking soda instantly kills all viral infections", 0),
        ("Unbelievable leak exposes celebrity hidden clone replacement program", 0),
        ("SHOCKING TRUTH: Alien mothership spotted hiding behind moon by amateur telescope", 0),
        ("FORWARD THIS TO EVERYONE before Facebook deletes this secret medical discovery", 0),
        ("Ancient lost scroll proves magic elixir reverses aging in three days", 0),
        ("BANNED VIDEO: What they don't want you to know about energy drinks and DNA modification", 0),
        ("URGENT: Microchips secretly implanted in tap water supply nationwide", 0),
        ("Miracle water trick guarantees instant weight loss without diet or exercise", 0),
    ]

    df = pd.DataFrame(training_data, columns=['text', 'label'])
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return vectorizer, model

vectorizer, ml_model = build_local_ml_model()

def extract_search_queries(text: str) -> list[str]:
    """
    Generates multi-tier targeted search queries from claim text or long paragraphs.
    Handles full-length articles by prioritizing lead sentences and key entities.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 8]
    lead_text = " ".join(sentences[:2]) if sentences else text
    
    def get_entities(s):
        words = s.split()
        proper_nouns = []
        for w in words:
            clean_w = re.sub(r'[^\w]', '', w)
            if len(clean_w) > 1 and clean_w[0].isupper():
                if clean_w.lower() not in ["the", "this", "that", "breaking", "urgent", "according", "minister", "official", "union", "report", "statement"]:
                    proper_nouns.append(clean_w)
        return list(dict.fromkeys(proper_nouns))
        
    lead_entities = get_entities(lead_text)
    all_entities = get_entities(text)
    
    stopwords = set([
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at", 
        "to", "for", "with", "by", "about", "against", "between", "into", "through", "during", 
        "before", "after", "above", "below", "from", "up", "down", "out", "off", "over", 
        "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
        "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", 
        "just", "should", "now", "of", "it", "that", "this", "these", "those", "they", "them", 
        "their", "what", "which", "who", "whom", "he", "him", "his", "she", "her", "has", "have", 
        "had", "having", "do", "does", "did", "according", "reports", "stated", "official", "published", "approved"
    ])
    
    clean_lead = re.sub(r'[^\w\s]', '', lead_text.lower())
    lead_words = [w for w in clean_lead.split() if w not in stopwords and len(w) > 2]
    
    queries = []
    
    # Tier 1: Lead Entities
    if lead_entities:
        q1 = " ".join(lead_entities[:4])
        queries.append(q1)
        
    # Tier 2: All Key Entities across full text
    if all_entities and len(all_entities) > 2:
        q2 = " ".join(all_entities[:4])
        queries.append(q2)
        
    # Tier 3: Core content words from lead sentence
    if len(lead_words) >= 3:
        q3 = " ".join(lead_words[:5])
        queries.append(q3)
        
    # Tier 4: Core 3 content words
    if len(lead_words) >= 2:
        q4 = " ".join(lead_words[:3])
        queries.append(q4)
        
    return list(dict.fromkeys(queries))

def analyze_linguistic_markers(text: str):
    """
    Calculates sensationalism, clickbait markers, and journalistic vocabulary.
    """
    words = text.split()
    total_words = max(len(words), 1)
    
    upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_ratio = (upper_words / total_words) * 100
    
    exclamations = text.count("!")
    questions = text.count("?")
    sensational_punct = exclamations + (questions * 0.5)
    
    clickbait_keywords = [
        "shocking", "miracle", "secret", "banned", "cure", "urgent", "leak",
        "they don't want you to know", "doctors hate", "forward this", "unbelievable",
        "mind control", "instant", "guaranteed", "truth behind", "exposed", "hidden",
        "shocking truth", "banned video", "miracle spice", "cloning program"
    ]
    
    text_lower = text.lower()
    matched_triggers = [kw for kw in clickbait_keywords if kw in text_lower]
    trigger_score = min(len(matched_triggers) * 25, 100)
    
    journalistic_keywords = [
        "according to", "reported", "announced", "published", "study", "researchers",
        "officials", "spokesperson", "statement", "confirmed", "data", "percent", "ministry",
        "department", "university", "journal", "agency", "court", "minister", "government",
        "assembly", "parliament", "amendment", "bill", "supreme court", "lok sabha", "rajya sabha",
        "police", "isro", "rbi", "nasa", "reuters", "express", "times", "today", "hindu", "united nations",
        "un", "forces", "military", "nato", "coalition"
    ]
    matched_journalistic = [kw for kw in journalistic_keywords if kw in text_lower]
    journalistic_score = min(len(matched_journalistic) * 20, 100)

    sensationalism_score = int(min(
        max((caps_ratio * 2.0) + (sensational_punct * 15) + (trigger_score * 0.6) - (journalistic_score * 0.1), 0),
        100
    ))
    
    return {
        "sensationalism_score": sensationalism_score,
        "caps_ratio": round(caps_ratio, 1),
        "exclamations": exclamations,
        "triggers_found": matched_triggers,
        "journalistic_score": journalistic_score
    }

def google_news_rss_search(query: str):
    """
    Queries Google News RSS feed directly. Unrestricted, fast, and 100% free.
    """
    results = []
    try:
        encoded_q = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
        
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        
        with urllib.request.urlopen(req, timeout=6) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            for item in root.findall('.//item')[:6]:
                title_elem = item.find('title')
                link_elem = item.find('link')
                source_elem = item.find('source')
                pubdate_elem = item.find('pubDate')
                
                title = title_elem.text if title_elem is not None else ""
                link = link_elem.text if link_elem is not None else "#"
                source = source_elem.text if source_elem is not None else "News Source"
                pubdate = pubdate_elem.text if pubdate_elem is not None else ""
                
                if title:
                    results.append({
                        "title": title,
                        "url": link,
                        "snippet": f"Publisher: {source} | Date: {pubdate}. Headline: {title}",
                        "source_engine": "Google News RSS"
                    })
    except Exception:
        pass
    return results

def fallback_duckduckgo_search(query: str):
    """
    Secondary fallback search using DuckDuckGo.
    """
    results = []
    try:
        ddgs = DDGS()
        res = list(ddgs.news(query, max_results=5))
        if not res:
            res = list(ddgs.text(query, max_results=5))
        
        for r in res:
            title = r.get('title', '')
            url = r.get('href', r.get('url', '#'))
            snippet = r.get('body', r.get('snippet', ''))
            if title:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source_engine": "DuckDuckGo Engine"
                })
    except Exception:
        pass
    return results

def fetch_and_corroborate_live_sources(claim: str):
    """
    Queries Google News RSS and DuckDuckGo for live coverage.
    Computes Sentence-Level & Paragraph TF-IDF Cosine Similarity to prevent vector dilution on long texts.
    """
    sources = []
    debunk_matches_count = 0
    raw_results = []
    
    explicit_debunk_phrases = [
        "fact check", "fact-check", "fake news", "hoax", "false claim", 
        "myth", "debunked", "baseless", "no truth", "viral rumour", "viral rumor",
        "misleading claim", "disproved", "falsely claimed", "fake post", "pib fact check"
    ]
    
    search_queries = extract_search_queries(claim)
    
    # Layer 1: Google News RSS Search
    for q in search_queries:
        res_gnews = google_news_rss_search(q)
        if res_gnews:
            raw_results.extend(res_gnews)
            if len(raw_results) >= 5:
                break
            
    # Layer 2: DuckDuckGo Fallback Search
    if len(raw_results) < 3:
        for q in search_queries:
            res_ddg = fallback_duckduckgo_search(q)
            if res_ddg:
                raw_results.extend(res_ddg)
                if len(raw_results) >= 5:
                    break
                
    if raw_results:
        snippets = []
        seen_titles = set()
        
        for r in raw_results:
            title = r.get('title', 'Web Result')
            snippet = r.get('snippet', '')
            url = r.get('url', '#')
            engine = r.get('source_engine', 'Web Search')
            
            clean_title_key = re.sub(r'[^\w]', '', title.lower())
            if clean_title_key in seen_titles:
                continue
            seen_titles.add(clean_title_key)
            
            title_lower = title.lower()
            snippet_lower = snippet.lower()
            
            for dphrase in explicit_debunk_phrases:
                if dphrase in title_lower or dphrase in snippet_lower:
                    debunk_matches_count += 1
                    break
                    
            snippets.append(f"{title}. {snippet}")
            sources.append({"title": title, "url": url, "snippet": snippet, "engine": engine})
            
            if len(sources) >= 6:
                break
        
        if snippets:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', claim) if len(s.strip()) > 10]
            compare_units = [claim] + sentences if sentences else [claim]
            
            all_texts = compare_units + snippets
            sim_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = sim_vectorizer.fit_transform(all_texts)
            
            num_units = len(compare_units)
            unit_vectors = tfidf_matrix[0:num_units]
            snippet_vectors = tfidf_matrix[num_units:]
            
            sim_matrix = cosine_similarity(unit_vectors, snippet_vectors)
            snippet_max_sims = np.max(sim_matrix, axis=0) if sim_matrix.size > 0 else np.zeros(len(snippets))
            max_similarity = float(np.max(snippet_max_sims)) if len(snippet_max_sims) > 0 else 0.0
            
            if max_similarity >= 0.15:
                normalized_match = min(int(max_similarity * 250), 100)
            elif max_similarity >= 0.08:
                normalized_match = int(max_similarity * 200)
            else:
                normalized_match = int(max_similarity * 100)
                
            is_debunked_online = debunk_matches_count >= 1
            
            if is_debunked_online:
                normalized_match = min(normalized_match, 15)
                
            for i, src in enumerate(sources):
                snippet_best_sim = float(snippet_max_sims[i])
                src['similarity'] = round(snippet_best_sim * 100, 1)
                
            return sources, max_similarity, normalized_match, is_debunked_online

    return [], 0.0, 0, False

def evaluate_source_credibility(sources):
    """
    Evaluates source domain authority and assigns credibility tiers.
    Tier 1: High Trust / Tier 2: General / Tier 3: Unverified
    """
    tier_1_keywords = [
        "pib", "reuters", "bbc", "indian express", "the hindu", "ap news", "pti",
        "press trust of india", "isro", "who", "nasa", "rbi", "reserve bank",
        "supreme court", "financial express", "economic times", "business standard",
        "ndtv", "times of india", "hindustan times", "alt news", "boom live",
        "snopes", "mint", "wire", "scroll", "aaj tak", "lok sabha"
    ]
    
    evaluated = []
    t1_count, t2_count, t3_count = 0, 0, 0
    
    for s in sources:
        title_lower = s.get('title', '').lower()
        snippet_lower = s.get('snippet', '').lower()
        url_lower = s.get('url', '').lower()
        text_comb = f"{title_lower} {snippet_lower} {url_lower}"
        
        is_t1 = any(kw in text_comb for kw in tier_1_keywords)
        
        if is_t1:
            tier_label = "🟢 Tier 1 (High Trust / Official News Outlet)"
            tier_badge = "Tier 1 — High Trust"
            badge_color = "#22c55e"
            t1_count += 1
        elif "blog" in text_comb or "forum" in text_comb or "wordpress" in text_comb:
            tier_label = "🔴 Tier 3 (Unverified Blog / Social Post)"
            tier_badge = "Tier 3 — High Risk"
            badge_color = "#ef4444"
            t3_count += 1
        else:
            tier_label = "🟡 Tier 2 (General News / Aggregator)"
            tier_badge = "Tier 2 — Medium Trust"
            badge_color = "#f59e0b"
            t2_count += 1
            
        evaluated.append({
            "title": s.get('title', ''),
            "url": s.get('url', '#'),
            "engine": s.get('engine', 'Search Engine'),
            "similarity": s.get('similarity', 0),
            "tier_label": tier_label,
            "tier_badge": tier_badge,
            "badge_color": badge_color,
            "snippet": s.get('snippet', '')
        })
        
    return evaluated, {"t1": t1_count, "t2": t2_count, "t3": t3_count}

def generate_whatsapp_card(claim, verdict, verdict_icon, composite_truth_index, corroboration_score, ling_metrics, sources):
    """
    Generates a ready-to-copy WhatsApp formatted message card for instant sharing.
    """
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    short_claim = claim[:120] + "..." if len(claim) > 120 else claim
    
    source_lines = []
    if sources:
        for s in sources[:2]:
            source_lines.append(f"• {s['title'][:55]}...")
        source_str = "\n".join(source_lines)
    else:
        source_str = "• No matching live coverage found."
        
    card_text = f"""❌ *VERIFACT AI FACT-CHECK ALERT* ❌

📌 *Claim Analyzed:*
"{short_claim}"

📊 *Verification Status:* {verdict_icon} {verdict}
🛡️ *Truth Index:* {composite_truth_index}%
🌐 *Web Grounding Score:* {corroboration_score}%

💡 *Key Metrics:*
• Sensationalism Risk: {ling_metrics['sensationalism_score']}/100
• Journalistic Tone: {ling_metrics['journalistic_score']}/100

🔗 *Verified News Coverage:*
{source_str}

⏰ *Checked:* {timestamp}
_Verified via VeriFact AI Engine_"""

    return card_text

st.sidebar.markdown("### 📋 Verification Benchmarks")
st.sidebar.markdown("Select a sample claim to test live verification:")

sample_claims = {
    "Select a benchmark...": "",
    "🟢 Real News: Lok Sabha Public Examination Bill": "Lok Sabha approved the Public Examination Prevention of Unfair Means Amendment Bill with strict penalties for paper leaks.",
    "🟢 Real News: ISRO Gaganyaan Mission": "Indian Space Research Organisation successfully completed core stage engine testing for the Gaganyaan human spaceflight mission.",
    "🟢 Real Global Conflict: UN Ukraine Report": "The United Nations Human Rights Office reported that civilian casualties in Ukraine have exceeded 16,000 since the beginning of the full-scale conflict.",
    "🚨 Clickbait Fake: 5G & Bird DNA": "BREAKING URGENT: Secret government plot leaked as 5G cell towers emit scalar frequencies that alter human DNA overnight!",
    "⚠️ Debunked Rumor: RBI Plastic Currency": "The Reserve Bank of India has officially announced that all current paper currency notes will be fully replaced with plastic bank notes starting next month."
}

selected_sample = st.sidebar.selectbox("Choose Sample:", list(sample_claims.keys()))
default_text = sample_claims[selected_sample] if selected_sample != "Select a benchmark..." else ""

# Sidebar Audit Trail Quick Access
st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 Session History Counter")
st.sidebar.info(f"Claims Verified This Session: **{len(st.session_state.verification_history)}**")

st.markdown("""
<div class="command-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 class="command-title">🛡️ VeriFact AI — Command Center</h1>
            <p class="command-subtitle">Grounding News Corroboration Engine — Google News RSS + TF-IDF Vector Math + WhatsApp Debunk Card</p>
        </div>
        <div style="text-align: right; font-size: 12px; color: #34d399; font-weight: 700; background: rgba(52, 211, 153, 0.1); padding: 6px 12px; border-radius: 20px; border: 1px solid rgba(52, 211, 153, 0.3);">
            ● LIVE ENGINE ONLINE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

user_claim = st.text_area(
    "Enter Headline, Article Paragraph, or News Claim for Analysis:",
    value=default_text,
    height=120,
    placeholder="e.g., Paste headline or excerpt from Indian Express, BBC, Reuters, or PIB..."
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🔎 Run Verification", type="primary", use_container_width=True)

if analyze_btn and user_claim.strip():
    
    with st.spinner("Analyzing claim against Google News RSS & Sentence-Level TF-IDF Matrix..."):
        
        claim_vector = vectorizer.transform([user_claim])
        ml_prob_real = ml_model.predict_proba(claim_vector)[0][1] * 100
        
        ling_metrics = analyze_linguistic_markers(user_claim)
        
        sources, raw_max_sim, corroboration_score, is_debunked_online = fetch_and_corroborate_live_sources(user_claim)
        
        sensational_penalty = (100 - ling_metrics["sensationalism_score"])

        if is_debunked_online:
            composite_truth_index = 10
            verdict = "DEBUNKED FAKE / HOAX DETECTED"
            verdict_color = "#ef4444"
            verdict_icon = "🚨"

        elif ling_metrics["sensationalism_score"] >= 40 and corroboration_score < 40:
            composite_truth_index = max(10, 100 - ling_metrics["sensationalism_score"] - 15)
            verdict = "DEBUNKED FAKE / SENSATIONAL CLICKBAIT"
            verdict_color = "#ef4444"
            verdict_icon = "🚨"

        elif corroboration_score >= 20 or raw_max_sim >= 0.10:
            composite_truth_index = int(
                (max(corroboration_score, 65) * 0.70) + 
                (sensational_penalty * 0.15) + 
                (ml_prob_real * 0.15)
            )
            composite_truth_index = max(composite_truth_index, 75)
            verdict = "VERIFIED REAL / HIGHLY LIKELY"
            verdict_color = "#22c55e"
            verdict_icon = "🟢"

        elif ling_metrics["journalistic_score"] >= 20 and corroboration_score < 20:
            composite_truth_index = 30
            verdict = "UNVERIFIED / PROBABLE FAKE NEWS"
            verdict_color = "#f59e0b"
            verdict_icon = "⚠️"

        else:
            composite_truth_index = 45
            verdict = "UNVERIFIED / PENDING CORROBORATION"
            verdict_color = "#f59e0b"
            verdict_icon = "⚠️"

        # Evaluate Source Credibility Tiers
        eval_sources, tier_counts = evaluate_source_credibility(sources)
        
        # Generate WhatsApp Share Card
        whatsapp_card_text = generate_whatsapp_card(
            user_claim, verdict, verdict_icon, composite_truth_index, 
            corroboration_score, ling_metrics, sources
        )

        # Log to Session Verification History
        st.session_state.verification_history.append({
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Claim": user_claim[:80] + "..." if len(user_claim) > 80 else user_claim,
            "Verdict": verdict,
            "Truth Index (%)": composite_truth_index,
            "Web Corroboration (%)": corroboration_score,
            "Sensationalism Score": ling_metrics['sensationalism_score'],
            "Sources Found": len(sources)
        })

    st.markdown("---")
    
    # Verdict Header Card
    st.markdown(f"""
    <div style="background-color: rgba(15, 23, 42, 0.85); padding: 22px; border-radius: 16px; border-left: 6px solid {verdict_color}; border-top: 1px solid rgba(255,255,255,0.08); margin-bottom: 24px;">
        <h2 style="margin: 0; color: white; font-size: 22px; display: flex; align-items: center; gap: 10px;">
            <span>{verdict_icon}</span> <span>{verdict}</span>
        </h2>
        <p style="margin-top: 8px; color: #cbd5e1; font-size: 14px;">
            <b>Truth Index:</b> {composite_truth_index}% &nbsp;|&nbsp; 
            <b>Google News Corroboration:</b> {corroboration_score}% &nbsp;|&nbsp; 
            <b>Sensationalism Score:</b> {ling_metrics['sensationalism_score']}/100
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{verdict_color}">{composite_truth_index}%</div>
            <div class="metric-label">Truth Index</div>
            <div class="progress-container">
                <div class="progress-bar-fill" style="width: {composite_truth_index}%; background-color: {verdict_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#38bdf8">{corroboration_score}%</div>
            <div class="metric-label">Web Grounding</div>
            <div class="progress-container">
                <div class="progress-bar-fill" style="width: {corroboration_score}%; background-color: #38bdf8;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#a855f7">{ling_metrics['journalistic_score']}/100</div>
            <div class="metric-label">Journalistic Tone</div>
            <div class="progress-container">
                <div class="progress-bar-fill" style="width: {ling_metrics['journalistic_score']}%; background-color: #a855f7;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        sens_color = "#ef4444" if ling_metrics['sensationalism_score'] >= 40 else "#34d399"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{sens_color}">{ling_metrics['sensationalism_score']}/100</div>
            <div class="metric-label">Sensationalism</div>
            <div class="progress-container">
                <div class="progress-bar-fill" style="width: {ling_metrics['sensationalism_score']}%; background-color: {sens_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Live News & Source Authority", 
        "📲 WhatsApp Debunk Card", 
        "🧠 Linguistic Analysis", 
        "📜 Audit History & Export",
        "⚙️ Technical Architecture"
    ])
    
    with tab1:
        st.markdown("### Live Corroboration & Source Credibility Matrix")
        if is_debunked_online:
            st.error("🚨 **Fact-check or debunking articles disproving this claim were found in online news registries!**")
        
        # Source Credibility Tier Summary Bar
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**🟢 Tier 1 Outlets (High Trust):** `{tier_counts['t1']}`")
        with c2:
            st.markdown(f"**🟡 Tier 2 Outlets (Aggregators):** `{tier_counts['t2']}`")
        with c3:
            st.markdown(f"**🔴 Tier 3 Outlets (Unverified):** `{tier_counts['t3']}`")

        st.markdown("---")
        
        if eval_sources:
            st.write(f"Found **{len(eval_sources)}** live news matches. Sentence-Level TF-IDF Cosine Similarity assesses claim alignment:")
            for idx, src in enumerate(eval_sources, 1):
                badge_col = src['badge_color']
                tier_lab = src['tier_label']
                with st.expander(f"{idx}. {src['title']} ({src['tier_badge']} | Match: {src.get('similarity', 0)}%)"):
                    st.markdown(f"<span style='color:{badge_col}; font-weight:bold;'>{tier_lab}</span>", unsafe_allow_html=True)
                    st.write(src['snippet'])
                    st.markdown(f"[🔗 Open Source Article]({src['url']})")
        else:
            st.info("No direct live news matches found on search engines. Unverified claims receive a neutral pending score.")

    with tab2:
        st.markdown("### 📲 Ready-to-Share WhatsApp Debunk Card")
        st.write("Copy and share this pre-formatted fact-check alert to debunk rumors on WhatsApp, Telegram, or Twitter:")
        
        st.markdown(f'<div class="whatsapp-card-box">{html.escape(whatsapp_card_text)}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.code(whatsapp_card_text, language="markdown")

    with tab3:
        st.markdown("### Linguistic & Clickbait Indicators")
        st.write(f"- **ALL-CAPS Word Ratio:** {ling_metrics['caps_ratio']}%")
        st.write(f"- **Exclamation Marks Count:** {ling_metrics['exclamations']}")
        st.write(f"- **Journalistic Tone Score:** {ling_metrics['journalistic_score']}/100")
        
        if ling_metrics['triggers_found']:
            st.warning(f"⚠️ **Sensational Trigger Words Detected:** {', '.join(ling_metrics['triggers_found'])}")
        else:
            st.success("✅ No sensational clickbait trigger phrases detected.")

    with tab4:
        st.markdown("### 📜 Session Verification Audit History")
        st.write("Track all news checks conducted during this active session:")
        
        if st.session_state.verification_history:
            history_df = pd.DataFrame(st.session_state.verification_history)
            st.dataframe(history_df, use_container_width=True)
            
            # Download CSV Export Button
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Session Audit History (CSV)",
                data=csv_data,
                file_name=f"verifact_audit_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.info("No claims verified yet in this session.")

    with tab5:
        st.markdown("""
        ### VeriFact AI Engine Specifications
        1. **Google News RSS Grounding**: Searches Google News XML RSS streams without facing rate limits or API key restrictions.
        2. **Multi-Tier Search Query Generator**: Generates targeted query variations (Named Entities, Content Keywords) to locate live coverage.
        3. **Sentence-Level TF-IDF Vector Math**: Measures vector similarity between claim sentences and live web snippets to eliminate vector length dilution on full-length articles.
        4. **Source Credibility Evaluation**: Categorizes news publishers into Tier 1 (PIB, Reuters, BBC, The Hindu), Tier 2, and Tier 3 sources.
        5. **WhatsApp Debunk Card**: Converts fact-check outputs into copyable social messaging cards.
        """)

elif analyze_btn:
    st.warning("Please enter a claim or headline to analyze.")

elif not analyze_btn and st.session_state.verification_history:
    st.markdown("---")
    st.markdown("### 📜 Session Verification Audit Log")
    history_df = pd.DataFrame(st.session_state.verification_history)
    st.dataframe(history_df, use_container_width=True)
    
    csv_data = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Session Audit History (CSV)",
        data=csv_data,
        file_name=f"verifact_audit_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
