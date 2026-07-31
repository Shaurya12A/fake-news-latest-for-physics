import os
import streamlit as st
from google import genai
from google.genai import types

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Live Fact-Checker & Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Live AI Fact-Checker & Fake News Detector")
st.markdown("Verify news headlines, viral claims, or full articles against live web sources in real time.")

# Sidebar for API Key configuration
st.sidebar.header("⚙️ Configuration")

# Check Streamlit Secrets first, otherwise fallback to text input
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("API Key loaded from Streamlit Secrets!")
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get a free API key from Google AI Studio")

if not api_key:
    st.info("💡 **Getting Started:** Enter your API key in the sidebar, or set `GEMINI_API_KEY` in Streamlit Cloud secrets to run automatically.")
    st.stop()

# Initialize GenAI Client
@st.cache_resource
def get_genai_client(key):
    return genai.Client(api_key=key)

client = get_genai_client(api_key)

# Input Section
user_claim = st.text_area(
    "Enter a headline, claim, or news article text to fact-check:",
    height=130,
    placeholder="e.g., A major solar flare is predicted to knock out all global satellite communications next Tuesday..."
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("🔎 Fact-Check Claim", type="primary")

if analyze_btn and user_claim.strip():
    with st.spinner("Searching the live web and cross-referencing credible sources..."):
        try:
            prompt = f"""
            You are an expert investigative fact-checker. 
            Analyze the following claim or news text by searching the live web for verified facts, credible news reports, and official statements.
            
            Claim to check:
            "{user_claim}"
            
            Provide your analysis in the following structured format:
            1. **VERDICT**: State clearly as one of [VERIFIED REAL, MOSTLY TRUE, MISLEADING, DEBUNKED FAKE, UNVERIFIED / SATIRE].
            2. **TRUTH RATING**: A confidence percentage rating (e.g., 85%).
            3. **EXECUTIVE SUMMARY**: A concise 2-3 sentence overview of the fact-check findings.
            4. **KEY FACTS & EVIDENCE**: Bullet points detailing real-world evidence, official sources, or timeline details.
            5. **LINGUISTIC & CONTEXT ANALYSIS**: Note any clickbait phrasing, emotional manipulation, or sensationalism detected.
            """

            # Call Gemini with Google Search Grounding enabled
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                )
            )

            st.markdown("---")
            st.subheader("📋 Fact-Check Report")
            st.markdown(response.text)

            # Display citations and sources if grounding metadata exists
            grounding_metadata = getattr(response.candidates[0], "grounding_metadata", None) if response.candidates else None
            if grounding_metadata and getattr(grounding_metadata, "grounding_chunks", None):
                st.markdown("---")
                st.subheader("🌐 Web Sources & Citations")
                chunks = grounding_metadata.grounding_chunks
                for idx, chunk in enumerate(chunks, start=1):
                    web = getattr(chunk, "web", None)
                    if web:
                        title = getattr(web, "title", "Web Source")
                        uri = getattr(web, "uri", "#")
                        st.markdown(f"{idx}. [{title}]({uri})")

        except Exception as e:
            st.error(f"An error occurred during verification: {str(e)}")

elif analyze_btn:
    st.warning("Please enter a claim or headline before running the check.")
