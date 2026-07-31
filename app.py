import re
import math
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from duckduckgo_search import DDGS

st.set_page_config(
    page_title="Fake News Detector (Pure NLP & Web Corroboration)",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Algorithmic Fake News Detector & Fact-Checker")
st.markdown("""
*100% Free, Local NLP & Web-Grounded Verification Engine — No AI API Keys Required!*
This tool combines **TF-IDF Machine Learning**, **Cosine Similarity Corroboration**, and **Linguistic Bias Scoring**.
""")

@st.cache_resource
def build_local_ml_model():
    """
    Builds and trains a local TF-IDF + Logistic Regression model on expanded benchmark data.
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
    ]

    df = pd.DataFrame(training_data, columns=['text', 'label'])
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return vectorizer, model

vectorizer, ml_model = build_local_ml_model()

def analyze_linguistic_markers(text: str):
    """
    Calculates linguistic red flags (sensationalism, clickbait, ALL-CAPS)
    as well as positive journalistic tone markers.
    """
    words = text.split()
    total_words = max(len(words), 1)
    
    # 1. Capitalization ratio
    upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_ratio = (upper_words / total_words) * 100
    
    # 2. Punctuation density (!!!, ???)
    exclamations = text.count("!")
    questions = text.count("?")
    sensational_punct = exclamations + (questions * 0.5)
    
    # 3. Clickbait & sensational trigger words
    clickbait_keywords = [
        "shocking", "miracle", "secret", "banned", "cure", "urgent", "leak",
        "they don't want you to know", "doctors hate", "forward this", "unbelievable",
        "mind control", "instant", "guaranteed", "truth behind", "exposed", "hidden",
        "shocking truth", "banned video", "miracle spice", "cloning program"
    ]
    
    text_lower = text.lower()
    matched_triggers = [kw for kw in clickbait_keywords if kw in text_lower]
    trigger_score = min(len(matched_triggers) * 25, 100)
    
    # 4. Positive Journalistic Tone Markers
    journalistic_keywords = [
        "according to", "reported", "announced", "published", "study", "researchers",
        "officials", "spokesperson", "statement", "confirmed", "data", "percent", "ministry",
        "department", "university", "journal", "agency"
    ]
    matched_journalistic = [kw for kw in journalistic_keywords if kw in text_lower]
    journalistic_score = min(len(matched_journalistic) * 20, 100)

    # Compute overall sensationalism index (0 to 100)
    sensationalism_score = int(min(
        max((caps_ratio * 2.0) + (sensational_punct * 15) + (trigger_score * 0.6) - (journalistic_score * 0.3), 0),
        100
    ))
    
    return {
        "sensationalism_score": sensationalism_score,
        "caps_ratio": round(caps_ratio, 1),
        "exclamations": exclamations,
        "triggers_found": matched_triggers,
        "journalistic_score": journalistic_score
    }

def fetch_and_corroborate_live_sources(claim: str):
    """
    Searches DuckDuckGo directly for matching news coverage and computes
    normalized TF-IDF Cosine Similarity between claim and live headlines.
    """
    sources = []
    
    try:
        ddgs = DDGS()
        clean_query = re.sub(r'[^\w\s]', '', claim).strip()[:100]
        
        # Direct query for broader news coverage
        results = list(ddgs.text(clean_query, max_results=6))
        
        if results:
            snippets = []
            for r in results:
                title = r.get('title', 'Web Result')
                body = r.get('body', '')
                url = r.get('href', '#')
                snippet_text = f"{title}. {body}"
                snippets.append(snippet_text)
                sources.append({"title": title, "url": url, "snippet": body})
            
            # Compute TF-IDF Cosine Similarity
            all_texts = [claim] + snippets
            sim_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = sim_vectorizer.fit_transform(all_texts)
            
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
            avg_similarity = float(np.mean(similarities)) if len(similarities) > 0 else 0.0
            
            # Non-linear normalization: TF-IDF cosine similarity > 0.15 is strong for short vs long texts
            # Scale max_similarity so 0.20 similarity yields ~75% match, 0.35+ yields ~95%
            normalized_match = min(int((max_similarity ** 0.6) * 100 * 1.8), 100)
            
            for i, src in enumerate(sources):
                src['similarity'] = round(float(similarities[i]) * 100, 1)
                
            return sources, max_similarity, normalized_match
    except Exception as e:
        pass
        
    return [], 0.0, 0

st.sidebar.header("📋 Benchmark Examples")
st.sidebar.markdown("Select a sample claim to test the verification pipeline instantly:")

sample_claims = {
    "Select a benchmark...": "",
    "🚨 Fake: 5G & Bird DNA Leak": "BREAKING URGENT: Internal leak proves 5G cell towers emit scalar frequencies that alter bird DNA, causing hundreds to fall dead!",
    "🟢 Real: NASA Webb Discovery": "NASA's James Webb Telescope has detected water vapor and atmospheric signatures on a rocky exoplanet orbiting in a star's habitable zone.",
    "⚠️ Misleading: Miracle Health Cure": "Drinking warm lemon water with baking soda every morning completely cures diabetes and eliminates 100% of viral infections instantly!",
    "🎭 Satire: Plant Conversation Record": "Local man sets national record for longest continuous conversation with a household snake plant."
}

selected_sample = st.sidebar.selectbox("Choose Sample:", list(sample_claims.keys()))

default_text = sample_claims[selected_sample] if selected_sample != "Select a benchmark..." else ""

user_claim = st.text_area(
    "Enter a news headline, article paragraph, or claim to evaluate:",
    value=default_text,
    height=120,
    placeholder="e.g., NASA confirmed a new asteroid will pass Earth safely this weekend..."
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🔎 Analyze Claim", type="primary", use_container_width=True)

if analyze_btn and user_claim.strip():
    
    with st.spinner("Processing local ML classification & scraping live web corroboration..."):
        
        # Step 1: Local ML Classifier Prediction
        claim_vector = vectorizer.transform([user_claim])
        ml_prob_real = ml_model.predict_proba(claim_vector)[0][1] * 100
        
        # Step 2: Linguistic & Sensationalism Analysis
        ling_metrics = analyze_linguistic_markers(user_claim)
        
        # Step 3: Live Web Corroboration & Cosine Similarity
        sources, raw_max_sim, corroboration_score = fetch_and_corroborate_live_sources(user_claim)
        
        # Step 4: Robust Composite Truth Index Calculation
        sensational_penalty = (100 - ling_metrics["sensationalism_score"])
        journalistic_boost = ling_metrics["journalistic_score"] * 0.15

        if len(sources) > 0 and corroboration_score > 30:
            # Strong web corroboration found
            composite_truth_index = int(
                (corroboration_score * 0.55) + 
                (sensational_penalty * 0.30) + 
                (ml_prob_real * 0.15)
            )
        else:
            # Fallback for novel/unmatched news based on tone and sensationalism
            base_score = 60 + journalistic_boost - (ling_metrics["sensationalism_score"] * 0.45)
            composite_truth_index = int(np.clip(base_score, 15, 85))

        # Determine Final Verdict
        if composite_truth_index >= 68:
            verdict = "VERIFIED REAL / HIGHLY LIKELY"
            verdict_color = "#22c55e"
            verdict_icon = "🟢"
        elif composite_truth_index >= 45:
            verdict = "MISLEADING / PARTIALLY UNVERIFIED"
            verdict_color = "#f59e0b"
            verdict_icon = "⚠️"
        else:
            verdict = "DEBUNKED FAKE / HIGH SENSATIONALISM"
            verdict_color = "#ef4444"
            verdict_icon = "🚨"

    st.markdown("---")
    st.subheader("📋 Verification Dashboard")
    
    # Verdict Header Box
    st.markdown(f"""
    <div style="background-color: rgba(30, 41, 59, 0.8); padding: 20px; border-radius: 12px; border-left: 6px solid {verdict_color}; margin-bottom: 20px;">
        <h2 style="margin: 0; color: white;">{verdict_icon} {verdict}</h2>
        <p style="margin-top: 8px; color: #cbd5e1; font-size: 15px;">
            <b>Composite Truth Index:</b> {composite_truth_index}% &nbsp;|&nbsp; 
            <b>Live Corroboration Match:</b> {corroboration_score}% &nbsp;|&nbsp; 
            <b>Sensationalism Index:</b> {ling_metrics['sensationalism_score']}/100
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metric Gauges
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Truth Index Score", f"{composite_truth_index}%")
    with col_m2:
        st.metric("Web Corroboration", f"{corroboration_score}%")
    with col_m3:
        st.metric("Journalistic Tone", f"{ling_metrics['journalistic_score']}/100")
    with col_m4:
        st.metric("Sensationalism Score", f"{ling_metrics['sensationalism_score']}/100")

    tab1, tab2, tab3 = st.tabs(["🌐 Live Search Corroboration", "🧠 Linguistic & Bias Analysis", "⚙️ How This Engine Works"])
    
    with tab1:
        st.markdown("### Live Web Search Matches (DuckDuckGo)")
        if sources:
            st.write(f"Found **{len(sources)}** relevant web results. Cosine similarity evaluates match confidence.")
            for idx, src in enumerate(sources, 1):
                with st.expander(f"{idx}. {src['title']} (TF-IDF Match: {src['similarity']}%)"):
                    st.write(src['snippet'])
                    st.markdown(f"[🔗 Read Full Source Article]({src['url']})")
        else:
            st.info("No direct live news matches found. Unverified or novel claims receive lower corroboration confidence.")

    with tab2:
        st.markdown("### Linguistic & Clickbait Red Flags")
        st.write(f"- **ALL-CAPS Word Ratio:** {ling_metrics['caps_ratio']}%")
        st.write(f"- **Exclamation Marks Count:** {ling_metrics['exclamations']}")
        st.write(f"- **Journalistic Vocabulary Boost:** +{ling_metrics['journalistic_score']} points")
        
        if ling_metrics['triggers_found']:
            st.warning(f"⚠️ **Sensational Trigger Words Detected:** {', '.join(ling_metrics['triggers_found'])}")
        else:
            st.success("✅ No obvious sensational clickbait trigger phrases detected.")

    with tab3:
        st.markdown("""
        ### Technical Architecture (100% API-Free)
        1. **Live DuckDuckGo Search**: Scrapes headlines matching your claim directly without API keys.
        2. **Normalized TF-IDF Cosine Similarity**: Converts the claim and search snippets into vector space and applies non-linear scaling to evaluate real-world text matches accurately.
        3. **Journalistic & Sensational Heuristics**: Evaluates reporting style, attributions, ALL-CAPS ratios, and clickbait words.
        4. **Composite Truth Index**: Combines live corroboration, ML classification, and tone analysis into a balanced verdict.
        """)

elif analyze_btn:
    st.warning("Please enter a claim or headline to analyze.")
