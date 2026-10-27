import streamlit as st  # Web UI framework for the dashboard
import pandas as pd  # Data manipulation and table handling
import numpy as np  # Numerical array operations
import re  # Regular expressions for string cleaning
import joblib  # Saving and loading serialized ML models/objects
from pathlib import Path  # Object-oriented filesystem path management
from sklearn.feature_extraction.text import TfidfVectorizer  # Text to numerical TF-IDF feature matrix
from sklearn.metrics.pairwise import cosine_similarity  # Compute similarity between vectors (imported for utility)
from sklearn.model_selection import train_test_split  # Split data into training and testing partitions
from sklearn.linear_model import LogisticRegression  # Binary classification ML algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Performance evaluation metrics

st.set_page_config(  # Configure Streamlit page attributes
    page_title="VeriFact AI — Misinformation Command Center",  # Browser tab title
    page_icon="🛡️",  # Browser tab favicon
    layout="wide",  # Utilize full screen width layout
    initial_sidebar_state="expanded"  # Keep sidebar expanded by default
)

# Configure the dashboard appearance with custom CSS
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
""", unsafe_allow_html=True)  # Inject custom CSS styles directly into Streamlit

@st.cache_resource  # Cache trained model in memory across reruns
def load_trained_model():  # Function to load model artifacts from disk
    try:  # Handle errors if files are absent or corrupt
        vectorizer = joblib.load("vectorizer.pkl")  # Load pre-trained TF-IDF vectorizer
        model = joblib.load("model (2).pkl")  # Load trained classification model
        return model, vectorizer, None  # Return model and vectorizer with no error
    except Exception as e:  # Catch disk I/O or deserialization errors
        return None, None, str(e)  # Return None along with the error description

trained_model, trained_vectorizer, model_load_error = load_trained_model()  # Execute cached model loader

DATASET_DIR = Path("datasets")  # Define root directory for dataset storage
FEVER_LOCAL_FILE = DATASET_DIR / "fever_train.parquet"  # Path to FEVER training dataset
LIAR_LOCAL_FILES = {  # Paths to LIAR dataset splits
    "train": DATASET_DIR / "liar_train.tsv",  # LIAR training split path
    "valid": DATASET_DIR / "liar_valid.tsv",  # LIAR validation split path
    "test": DATASET_DIR / "liar_test.tsv",  # LIAR test split path
}  # Close LIAR file path dictionary

def model_clean_text(text):  # Preprocessing function for raw text
    text = str(text)  # Ensure text input is string type
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)  # Strip Reuters news wire datelines
    text = re.sub(r'^.*?[A-Z]{2,}\s*\(AP\)\s*-\s*', '', text, flags=re.IGNORECASE)  # Strip Associated Press datelines
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Strip web links and URLs
    text = re.sub(r'<.*?>', '', text)  # Strip embedded HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Strip digits, punctuation, and symbols
    return text.lower().strip()  # Normalize to lowercase and trim spaces

def find_local_file(primary_path, patterns):  # Locate file with fallback glob patterns
    if primary_path.exists():  # Check if primary expected path exists
        return primary_path  # Return existing path immediately
    if DATASET_DIR.exists():  # Ensure dataset folder is available to search
        for pattern in patterns:  # Iterate through allowed filename patterns
            matches = sorted(DATASET_DIR.glob(pattern))  # Find matching filenames sorted alphabetically
            if matches:  # If at least one match was found
                return matches[0]  # Pick the first matching file
    return None  # Return None if no matching file exists

def load_local_training_data():  # Load and prepare datasets for model training
    fever_file = find_local_file(  # Locate FEVER dataset file
        FEVER_LOCAL_FILE,  # Primary path
        ["*fever*.parquet", "*fever*.csv"]  # Match parquet or CSV variations
    )  # End FEVER search
    liar_train = find_local_file(LIAR_LOCAL_FILES["train"], ["*liar*train*.tsv"])  # Locate LIAR train TSV
    liar_valid = find_local_file(LIAR_LOCAL_FILES["valid"], ["*liar*valid*.tsv"])  # Locate LIAR validation TSV
    liar_test = find_local_file(LIAR_LOCAL_FILES["test"], ["*liar*test*.tsv"])  # Locate LIAR test TSV

    missing = []  # List to record missing dataset names
    if fever_file is None:  # Check if FEVER file is missing
        missing.append("FEVER training file")  # Log missing FEVER
    if liar_train is None:  # Check if LIAR train file is missing
        missing.append("LIAR train.tsv")  # Log missing LIAR train
    if liar_valid is None:  # Check if LIAR valid file is missing
        missing.append("LIAR valid.tsv")  # Log missing LIAR valid
    if liar_test is None:  # Check if LIAR test file is missing
        missing.append("LIAR test.tsv")  # Log missing LIAR test
    if missing:  # If any file was not found
        raise FileNotFoundError(  # Raise explicit exception with setup guidance
            "Missing local dataset files: " + ", ".join(missing) +
            ". Put the downloaded files inside the 'datasets' folder."
        )

    if fever_file.suffix.lower() == ".parquet":  # Check if FEVER is in parquet format
        fever_df = pd.read_parquet(fever_file)  # Load parquet file into DataFrame
    else:  # Fallback for CSV format
        fever_df = pd.read_csv(fever_file)  # Load CSV file into DataFrame

    fever_df["label"] = fever_df["label"].map({  # Remap FEVER labels to binary targets (0: Real, 1: Fake)
        "SUPPORTS": 0,  # Factual support mapped to 0 (Real)
        "REFUTES": 1,  # Factual refutation mapped to 1 (Fake)
        "NOT ENOUGH INFO": 1,  # Unverified claim mapped to 1 (Fake/Unreliable)
    })  # End FEVER label mapping
    fever_df = fever_df[["claim", "label"]].rename(columns={"claim": "text"})  # Standardize schema to text and label

    liar_columns = [  # Define official column header names for LIAR TSV
        "id", "label", "statement", "subject", "speaker", "job",
        "state", "party", "barely_true", "false", "half_true",
        "mostly_true", "pants_fire", "context"
    ]

    liar_frames = []  # Container to hold individual LIAR data splits
    for file_path in [liar_train, liar_valid, liar_test]:  # Loop through all 3 LIAR splits
        liar_frames.append(pd.read_csv(  # Read TSV and append to list
            file_path,  # Path of the split file
            sep="\t",  # Tab delimiter
            header=None,  # No headers present in raw LIAR TSV
            names=liar_columns  # Apply column names list
        ))  # End TSV load
    liar_df = pd.concat(liar_frames, ignore_index=True)  # Merge train, valid, and test into one table

    liar_df["label"] = liar_df["label"].map({  # Remap LIAR labels to binary targets (0: Real, 1: Fake)
        "true": 0,  # Completely true statement mapped to 0 (Real)
        "mostly-true": 0,  # Mostly true statement mapped to 0 (Real)
        "false": 1,  # False statement mapped to 1 (Fake)
        "pants-fire": 1,  # Grossly false statement mapped to 1 (Fake)
        "half-true": 1,  # Half true statement mapped to 1 (Fake/Unreliable)
        "barely-true": 1,  # Barely true statement mapped to 1 (Fake/Unreliable)
    })  # End LIAR label mapping
    liar_df = liar_df[["statement", "label"]].rename(  # Retain necessary fields and match schema
        columns={"statement": "text"}  # Rename 'statement' column to 'text'
    )

    data = pd.concat([fever_df, liar_df], ignore_index=True).dropna()  # Combine both datasets and drop nulls
    data["text"] = data["text"].apply(model_clean_text)  # Clean and preprocess every text sample
    data = data[data["text"].str.len() > 0]  # Filter out empty strings after cleaning
    data = data.drop_duplicates(subset=["text"])  # Deduplicate identical news texts
    return data  # Return cleaned training dataset

def train_local_model():  # Function to train and serialize the ML model
    data = load_local_training_data()  # Load combined and cleaned dataset

    X_train, X_test, y_train, y_test = train_test_split(  # Split dataset into train/test splits
        data["text"],  # Feature column (cleaned text)
        data["label"],  # Target labels (0 or 1)
        test_size=0.20,  # Reserve 20% of data for model testing
        stratify=data["label"],  # Maintain identical class distribution across splits
        random_state=42  # Seed for reproducible random splitting
    )

    vectorizer = TfidfVectorizer(  # Initialize text vectorizer
        ngram_range=(1, 2),  # Extract unigrams and bigrams
        min_df=2,  # Ignore words that appear fewer than 2 times
        max_df=0.98,  # Ignore corpus-wide stopwords appearing in >98% docs
        sublinear_tf=True,  # Use sublinear term frequency scaling (1 + log(tf))
        max_features=250000  # Cap vocabulary size to top 250k features
    )

    X_train_vec = vectorizer.fit_transform(X_train)  # Fit vectorizer on training text and transform it
    X_test_vec = vectorizer.transform(X_test)  # Transform test text using fitted vocabulary

    model = LogisticRegression(  # Initialize linear classification model
        max_iter=2000,  # Maximize iteration ceiling for optimization convergence
        class_weight="balanced",  # Adjust weights inversely proportional to class frequencies
        C=2.0,  # Regularization inverse strength (moderate regularization)
        solver="liblinear",  # Fast optimization algorithm for high-dimensional sparse text
        random_state=42  # Seed for deterministic optimization results
    )
    model.fit(X_train_vec, y_train)  # Train Logistic Regression on vectorized data

    test_pred = model.predict(X_test_vec)  # Predict classes on held-out test split
    accuracy = accuracy_score(y_test, test_pred)  # Calculate classification accuracy score

    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")  # Serialize fitted TF-IDF vectorizer to disk
    joblib.dump(model, "model (1).pkl")  # Serialize trained classification model to disk

    return model, vectorizer, accuracy, len(data), y_test, test_pred  # Return artifacts and performance stats

# Sidebar contains only sample news buttons. Clicking one places its text in the prediction box.
EXAMPLE_NEWS = {
    "Example 1 — Science": "The Earth revolves around the Sun, completing one orbit in about 365 days.",
    "Example 2 — Health Claim": "A viral message claims that drinking bleach can safely cure common illnesses.",
    "Example 3 — Unverified Report": "A social media post claims that all schools in a city will be closed tomorrow, but it does not cite an official announcement.",
}

def choose_example(example_text):  # Put the selected sample into the text area before Streamlit reruns
    st.session_state["model_claim"] = example_text

with st.sidebar:  # Keep the sidebar focused only on example news
    st.markdown("### 📰 Example News")  # Sidebar heading
    st.caption("Select an example to fill the prediction box.")  # Brief usage hint
    for example_title, example_text in EXAMPLE_NEWS.items():  # Create one button for each sample
        st.button(
            example_title,  # Button label
            key=f"example_{example_title}",  # Unique key for each button
            use_container_width=True,  # Make the button fit the sidebar width
            on_click=choose_example,  # Fill the prediction box when clicked
            args=(example_text,)  # Pass the sample text to the callback
        )

# Main page heading and prediction interface
st.markdown("<h1 style='color:#f8fafc; margin-bottom:5px;'>VeriFact AI Command Center</h1>", unsafe_allow_html=True)  # Main view heading
st.markdown("<p style='color:#94a3b8;'>Local Fake-News Prediction</p>", unsafe_allow_html=True)  # Main view subtitle
st.markdown("### 🤖 Trained Model Predictor")  # Predictor section heading
st.markdown(  # Explain the prediction mode
    "<p style='color:#94a3b8;'>TF-IDF + Logistic Regression prediction using your trained fake-news model. This mode does not perform live web searches.</p>",
    unsafe_allow_html=True
)

model_claim = st.text_area(  # Multi-line text input field for news text
    "Enter news text or a claim to predict:",  # Input field label
    height=180,  # Box height in pixels
    placeholder="Paste a headline, claim, or news article here...",  # Placeholder helper text
    key="model_claim"  # Shared state lets sidebar buttons populate this field
)

if trained_model is None or trained_vectorizer is None:  # Verify model and vectorizer exist
    st.error(  # Display error alert if artifacts are missing
        "The trained model could not be loaded. Make sure the model and vectorizer files are in the same folder as app.py. "
        + (f"\n\nLoading error: {model_load_error}" if model_load_error else "")
    )
else:  # Model files loaded successfully
    if st.button("🤖 Analyze News", type="primary", use_container_width=True):  # Trigger prediction action
        if not model_claim.strip():  # Check for blank or whitespace input
            st.warning("Please enter some text before analyzing.")  # Warn user about missing text
        else:  # Text input provided
            cleaned = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', model_claim, flags=re.IGNORECASE)  # Strip Reuters wire prefixes
            cleaned = re.sub(r'^.*?[A-Z]{2,}\s*\(AP\)\s*-\s*', '', cleaned, flags=re.IGNORECASE)  # Strip AP wire prefixes
            cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)  # Remove web addresses
            cleaned = re.sub(r'<.*?>', '', cleaned)  # Remove HTML tags
            cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)  # Keep letters and whitespace only
            cleaned = cleaned.lower().strip()  # Normalize case and trim whitespace

            if not cleaned:  # Check whether preprocessing removed all text
                st.warning("The entered text became empty after preprocessing. Please enter a longer news claim or article.")
            else:
                vec_input = trained_vectorizer.transform([cleaned])  # Convert text into TF-IDF features
                prediction = trained_model.predict(vec_input)[0]  # Predict the class label
                proba = trained_model.predict_proba(vec_input)[0]  # Get estimated class probabilities

                classes = list(trained_model.classes_)  # Read model class labels
                probability_by_class = {cls: float(prob) for cls, prob in zip(classes, proba)}  # Pair labels with probabilities

                real_prob = probability_by_class.get(0, probability_by_class.get('REAL', 0.0))  # Probability for REAL class
                fake_prob = probability_by_class.get(1, probability_by_class.get('FAKE', 0.0))  # Probability for FAKE class
                top_confidence = max(real_prob, fake_prob)  # Highest estimated class confidence
                truth_index = int(round(real_prob * 100))  # Display REAL probability as the Truth Index

                UNCERTAIN_THRESHOLD = 0.60  # Minimum confidence required to show a definite verdict

                if top_confidence < UNCERTAIN_THRESHOLD:  # Handle predictions below the confidence threshold
                    status_class = "badge-warning"
                    display_verdict = "⚠️ UNCERTAIN"
                    summary = (
                        f"The model does not have enough confidence to classify this as TRUE or FALSE. "
                        f"Its highest prediction confidence is {top_confidence * 100:.1f}%."
                    )
                elif prediction == 1:  # Display the FAKE class result
                    status_class = "badge-fake"
                    display_verdict = "🚨 FALSE"
                    summary = (
                        f"The trained model predicts this news as FALSE with "
                        f"{fake_prob * 100:.1f}% confidence."
                    )
                else:  # Display the REAL class result
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
                """, unsafe_allow_html=True)  # Render verdict and confidence card

                st.caption(  # Explain the displayed score
                    "Truth Index represents the model's estimated probability that the text is REAL. "
                    "UNCERTAIN is shown when neither REAL nor FAKE reaches 60% confidence."
                )
