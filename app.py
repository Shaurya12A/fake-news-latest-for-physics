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
    layout="wide"  # Utilize full screen width layout
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

# Show the app title and mode selector on the main page instead of a sidebar
st.markdown("<h2 style='color:#34d399; margin-bottom:0;'>🛡️ VeriFact AI</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:0.8rem;'>Misinformation Command Center</p>", unsafe_allow_html=True)

analysis_mode = st.radio(  # Main-page selector for choosing the app mode
    "Analysis Mode",
    [
        "🤖 Trained Model Predictor",  # Prediction mode
        "🛠️ Train / Update Model"  # Training mode
    ],
    index=0,  # Default to prediction mode
    horizontal=True  # Display options horizontally without a sidebar
)

st.markdown("<h1 style='color:#f8fafc; margin-bottom:5px;'>VeriFact AI Command Center</h1>", unsafe_allow_html=True)  # Main view heading
st.markdown("<p style='color:#94a3b8;'>Local Fake-News Prediction & Model Training</p>", unsafe_allow_html=True)  # Main view subtitle

if analysis_mode == "🛠️ Train / Update Model":  # View block for model training mode
    st.markdown("<h3>🛠️ Train / Update Fake-News Model</h3>")  # Section title
    st.markdown(  # Mode explanation
        "<p style='color:#94a3b8;'>Train the same TF-IDF + Logistic Regression model using downloaded FEVER and LIAR datasets stored locally in the <b>datasets</b> folder.</p>",
        unsafe_allow_html=True
    )

    st.info(  # Informational callout box
        "Dataset sources: FEVER — Sheffield NLP; LIAR — William Yang Wang / UCSB. "
        "The app uses local downloaded copies and does not download datasets during training."
    )

    st.markdown("<strong>Expected folder structure:</strong>")  # Directory requirements label
    st.code(  # Display expected folder hierarchy in code block
        "datasets/\n"
        "├── fever_train.parquet\n"
        "├── liar_train.tsv\n"
        "├── liar_valid.tsv\n"
        "└── liar_test.tsv"
    )

    if st.button("🚀 Train / Update Model", type="primary", use_container_width=True):  # Trigger model training button
        try:  # Handle runtime training exceptions
            with st.spinner("Loading local datasets, vectorizing text and training the model..."):  # Show loading spinner
                new_model, new_vectorizer, accuracy, sample_count, y_test, test_pred = train_local_model()  # Run training pipeline

            trained_model = new_model  # Update active in-memory model
            trained_vectorizer = new_vectorizer  # Update active in-memory vectorizer

            st.success("Model training completed and model files were saved.")  # Display success feedback
            st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")  # Display model accuracy metric
            st.caption(f"Training samples after cleaning/deduplication: {sample_count}")  # Display total cleaned dataset count

            st.markdown("<h4>Classification Report</h4>")  # Section label for detailed metrics
            st.code(classification_report(  # Display precision, recall, and F1-score report
                y_test,  # Ground truth test labels
                test_pred,  # Model test predictions
                target_names=["REAL", "FAKE"]  # Class label names
            ))

            st.markdown("<h4>Saved Model Files</h4>")  # Section label for generated files
            st.code("tfidf_vectorizer.pkl\nmodel (1).pkl")  # List exported file names

        except Exception as e:  # Catch and display training errors
            st.error(f"Training failed: {e}")  # Render training error alert

elif analysis_mode == "🤖 Trained Model Predictor":  # View block for prediction inference mode
    st.markdown("<h3>🤖 Trained Model Predictor</h3>")  # Section title
    st.markdown(  # Mode description
        "<p style='color:#94a3b8;'>TF-IDF + Logistic Regression prediction using your trained fake-news model. This mode does not perform live web searches.</p>",
        unsafe_allow_html=True
    )

    model_claim = st.text_area(  # Multi-line text input field for news text
        "Enter news text or a claim to predict:",  # Input field label
        height=180,  # Box height in pixels
        placeholder="Paste a headline, claim, or news article here..."  # Placeholder helper text
    )

    if trained_model is None or trained_vectorizer is None:  # Verify model and vectorizer exist
        st.error(  # Display error alert if artifacts are missing
            "The trained model could not be loaded. Make sure these files are in the same folder as app.py: "
            "tfidf_vectorizer.pkl and model (1).pkl"
            + (f"\n\nLoading error: {model_load_error}" if model_load_error else "")
        )
    else:  # Model files loaded successfully
        if st.button("🤖 Analyze News", type="primary", use_container_width=True):  # Trigger prediction action
            if not model_claim.strip():  # Check for blank or whitespace input
                st.warning("Please enter some text before analyzing.")  # Warn user about missing text
            else:  # Text input provided
                cleaned = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', model_claim, flags=re.IGNORECASE)  # Strip Reuters wire prefixes
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
