import streamlit as st
from google import genai
from duckduckgo_search import DDGS

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Live Fact-Checker & Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Live AI Fact-Checker & Fake News Detector")
st.markdown("Verify news headlines and claims against live web sources in real time.")

# Sidebar for API Key configuration
st.sidebar.header("⚙️ Configuration")

api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"].strip()
    st.sidebar.success("API Key loaded from Streamlit Secrets!")
else:
    user_input = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    if user_input:
        api_key = user_input.strip()

if not api_key:
    st.info("💡 **Getting Started:** Enter your API key in the sidebar, or set `GEMINI_API_KEY` in Streamlit Cloud secrets to run automatically.")
    st.stop()

# Initialize GenAI Client
@st.cache_resource
def get_genai_client(key):
    return genai.Client(api_key=key)

try:
    client = get_genai_client(api_key)
except Exception as e:
    st.error(f"Error initializing client: {str(e)}")
    st.stop()

def fetch_live_search_results(query: str):
    """Searches DuckDuckGo for top live news/web results without using Gemini Search Quota."""
    search_context = []
    sources = []
    try:
        ddgs = DDGS()
        results = list(ddgs.text(f"fact check news: {query}", max_results=5))
        for idx, r in enumerate(results, 1):
            title = r.get('title', 'Web Result')
            body = r.get('body', '')
            url = r.get('href', '#')
            search_context.append(f"Source [{idx}] ({title}):\n{body}\nURL: {url}")
            sources.append({"title": title, "url": url})
    except Exception as e:
        search_context.append("Web search temporarily unavailable.")
    return "\n\n".join(search_context), sources

# Input Section
user_claim = st.text_area(
    "Enter a headline, article excerpt, or claim to fact-check:",
    height=120,
    placeholder="e.g., NASA confirmed a new asteroid will hit Earth next week..."
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("🔎 Fact-Check Claim", type="primary")

if analyze_btn and user_claim.strip():
    with st.spinner("Searching the live web for context..."):
        # Step 1: Fetch live web search snippets via DuckDuckGo
        search_text, sources = fetch_live_search_results(user_claim)

    with st.spinner("Analyzing facts and evaluating authenticity..."):
        # Step 2: Feed user claim + real-time search context to Gemini
        prompt = f"""
        You are an expert investigative fact-checker.
        Evaluate the authenticity of the following claim using the provided live search results.

        CLAIM TO FACT-CHECK:
        "{user_claim}"

        LIVE WEB SEARCH CONTEXT:
        {search_text}

        Provide your analysis in this structured format:
        1. **VERDICT**: State clearly as one of [VERIFIED REAL, MOSTLY TRUE, MISLEADING, DEBUNKED FAKE, UNVERIFIED / SATIRE].
        2. **TRUTH RATING**: A confidence score out of 100%.
        3. **EXECUTIVE SUMMARY**: A concise 2-3 sentence overview of the fact-check findings based on the search context.
        4. **KEY FACTS & EVIDENCE**: Bullet points detailing real-world evidence, official sources, or timeline details found in the search context.
        5. **LINGUISTIC & CONTEXT ANALYSIS**: Mention any sensationalism, clickbait phrasing, or logical fallacies detected.
        """

        try:
            # Simple text call - no heavy tool search limits!
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )

            st.markdown("---")
            st.subheader("📋 Fact-Check Results")
            st.markdown(response.text)

            # Display Sources
            if sources:
                st.markdown("---")
                st.subheader("🌐 Live Web Sources Referenced")
                for idx, src in enumerate(sources, 1):
                    st.markdown(f"{idx}. [{src['title']}]({src['url']})")

        except Exception as e:
            st.error(f"Error during AI analysis: {str(e)}")

elif analyze_btn:
    st.warning("Please enter a claim or news headline before searching.")
