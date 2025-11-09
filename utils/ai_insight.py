import pandas as pd
import streamlit as st
import re
import sys
import os
# Allow model downloads for first initialization
import torch
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

def clean_text_for_pdf(text):
    """Clean text by replacing problematic characters with PDF-safe alternatives."""
    if not isinstance(text, str):
        text = str(text)
    
    replacements = {
        "—": "-", "–": "-", "'": "'", "'": "'", """: '"', """: '"', 
        "•": "-", "…": "...", "⚡": "*", "✨": "*", "🔥": "*",
        "📊": "[Chart]", "📈": "[Graph]", "📉": "[Graph]",
        "✅": "[OK]", "❌": "[X]", "⚠️": "[Warning]",
        "\n": "\n  ",  # Add some indentation for newlines
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# Initialize global variable for the summarizer
summarizer = None

# Try to import and initialize transformers
try:
    import os
    from transformers import pipeline

    # Check if running on Streamlit Cloud
    if os.getenv("STREAMLIT_SERVER_PORT"):
        print("🌐 Running on Streamlit Cloud — using ultra-light model")
        from transformers import pipeline
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",  # smaller, still works on CPU
            device=-1
        )

    else:
        print("Running locally - loading full AI model...")
        # Set up torch configuration
        torch.set_grad_enabled(False)  # Disable gradients for inference
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        
        # Initialize with a smaller model for better compatibility
        model_name = "sshleifer/distilbart-cnn-6-6"
        try:
            print("Loading model and tokenizer...")
            # Create the pipeline with the lightweight model
            summarizer = pipeline(
                "summarization",
                model=model_name,
                device=device
            )
        except Exception as model_error:
            # Handle model loading errors locally; show a Streamlit message and fallback to None
            st.error(f"""
            ⚠️ Error loading the AI model: {str(model_error)}
            
            Please try:
            1. Check your internet connection
            2. Clear your browser cache
            3. Restart the application
            """)
            summarizer = None
except Exception as model_error:
    st.error(f"""
    ⚠️ Error loading the AI model: {str(model_error)}
    
    Please try:
    1. Check your internet connection
    2. Clear your browser cache
    3. Restart the application
    """)
    # Fallback to basic summarization
    summarizer = None
        
except ImportError:
    st.error("""
    ⚠️ Required packages are not installed correctly.
    
    Please run these commands in your terminal:
    ```
    D:/DataVista/.venv/Scripts/pip.exe install --upgrade pip
    D:/DataVista/.venv/Scripts/pip.exe install torch==2.1.0 transformers==4.34.0 sentencepiece --no-cache-dir
    ```
    """)
    sys.exit(1)
except Exception as e:
    st.error(f"""
    ⚠️ Unexpected error: {str(e)}
    
    Please try:
    1. Make sure you have enough disk space
    2. Check your internet connection
    3. Try restarting the application
    """)
    sys.exit(1)

def generate_data_summary(df: pd.DataFrame) -> str:
    """Generate an AI-enhanced narrative summary of the dataset."""
    # Extract basic statistics
    rows, cols = df.shape
    missing_values = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    missing_ratio = round((missing_values / (rows * cols)) * 100, 2) if rows * cols > 0 else 0

    # Analyze column types
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

    # Correlation analysis
    corr_text = "No strong correlations detected."
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        corr_values = corr.unstack().sort_values(ascending=False)
        corr_values = corr_values[(corr_values < 0.999) & (corr_values > 0.4)]
        if not corr_values.empty:
            top_pair = corr_values.index[0]
            corr_value = round(corr_values.iloc[0], 2)
            corr_text = f"The strongest relationship (r = {corr_value}) is between `{top_pair[0]}` and `{top_pair[1]}`."

    # Generate narrative base
    narrative = f"""
    Dataset Overview:
    - The dataset has {rows} rows and {cols} columns.
    - Contains {missing_values} missing values ({missing_ratio}% of data) and {duplicates} duplicate rows.

    Numeric Insights:
    - Numeric columns include {', '.join(num_cols[:5]) + (' and more' if len(num_cols) > 5 else '')}.
    - {corr_text}

    Categorical Insights:
    - Key categorical columns are {', '.join(cat_cols[:5]) + (' and more' if len(cat_cols) > 5 else '')}.

    Data Health:
    - {'No duplicates detected.' if duplicates == 0 else f"{duplicates} duplicate rows found."}
    - {missing_values} missing values need to be handled ({missing_ratio}% of total data points).
    """

    # Use AI to make it human-friendly if available
    if summarizer is not None:
        try:
            # Set up maximum retries for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ai_summary = summarizer(
                        narrative,
                        max_length=160,
                        min_length=80,
                        do_sample=False
                    )[0]['summary_text']
                    return ai_summary
                except Exception as retry_error:
                    if attempt == max_retries - 1:
                        raise retry_error
                    st.warning(f"Retrying AI summarization (attempt {attempt + 2}/{max_retries})...")
                    continue
        except Exception as e:
            st.warning("⚠️ AI summarization failed. Falling back to standard narrative.")
            return narrative
    else:
        st.info("ℹ️ Using standard narrative (AI model not available)")
        return narrative
        return narrative

def display_ai_summary(summary_text: str, df=None):
    """Display AI-generated insights in a narrative format with professional styling."""
    st.markdown("## ✨ AI Insight Report")

    # Enhanced styling for narrative presentation
    st.markdown("""
        <style>
            .insight-box {
                background-color: #101820;
                color: #fafafa;
                padding: 25px;
                border-radius: 16px;
                border: 1px solid #00bfa6;
                font-size: 17px;
                line-height: 1.8;
                box-shadow: 0px 0px 10px rgba(0,191,166,0.4);
                margin: 20px 0;
            }
            .insight-section {
                background: rgba(255,255,255,0.05);
                padding: 20px;
                border-radius: 12px;
                margin: 15px 0;
                border-left: 4px solid;
            }
            .overview-section { border-color: #3498db; }
            .numeric-section { border-color: #2ecc71; }
            .categorical-section { border-color: #e67e22; }
            .health-section { border-color: #e74c3c; }
            .section-title {
                font-size: 1.2em;
                color: #00bfa6;
                margin-bottom: 10px;
                font-weight: 600;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #00bfa6;
            }
            .metric-label {
                color: #95a5a6;
                font-size: 0.9em;
            }
            .info-tag {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.9em;
                margin: 2px 4px;
            }
            .tag-good { background: rgba(46,204,113,0.2); color: #2ecc71; }
            .tag-warning { background: rgba(241,196,15,0.2); color: #f1c40f; }
            .tag-alert { background: rgba(231,76,60,0.2); color: #e74c3c; }
        </style>
    """, unsafe_allow_html=True)

    # Extract key metrics from DataFrame
    total_rows = df.shape[0] if df is not None else 0
    total_cols = df.shape[1] if df is not None else 0
    
    # Parse summary sections
    overview_pattern = r"Dataset Overview:(.+?)(?=Numeric Insights:|$)"
    numeric_pattern = r"Numeric Insights:(.+?)(?=Categorical Insights:|$)"
    categorical_pattern = r"Categorical Insights:(.+?)(?=Data Health:|$)"
    health_pattern = r"Data Health:(.+?)$"

    def extract_section(pattern, text):
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    overview = extract_section(overview_pattern, summary_text)
    numeric = extract_section(numeric_pattern, summary_text)
    categorical = extract_section(categorical_pattern, summary_text)
    health = extract_section(health_pattern, summary_text)

    # Display sections in a narrative format
    st.markdown(f"""
        <div class="insight-box">
            <div class="insight-section overview-section">
                <div class="section-title">📘 Dataset Overview</div>
                {overview.replace('-', '•')}
            </div>
            
            <div class="insight-section numeric-section">
                <div class="section-title">📊 Numeric Insights</div>
                {numeric.replace('-', '•')}
            </div>
            
            <div class="insight-section categorical-section">
                <div class="section-title">🧩 Categorical Insights</div>
                {categorical.replace('-', '•')}
            </div>
            
            <div class="insight-section health-section">
                <div class="section-title">🏥 Data Health</div>
                {health.replace('-', '•')}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Key Metrics Display
    col1, col2, col3 = st.columns(3)
    
    # Extract metrics using regex
    missing_match = re.search(r"(\d+)\s+missing", summary_text)
    corr_match = re.search(r"r\s*=\s*(0\.\d+)", summary_text)
    
    missing_values = int(missing_match.group(1)) if missing_match else 0
    correlation_value = float(corr_match.group(1)) if corr_match else 0.0
    
    # Calculate data quality score
    missing_ratio = missing_values / (total_rows * total_cols) if total_rows * total_cols > 0 else 0
    data_quality_score = round((1 - missing_ratio) * 100, 1)

    with col1:
        st.metric("Total Records", f"{total_rows:,}", "Dataset Size")
    with col2:
        st.metric("Data Quality", f"{data_quality_score}%", 
                 "Good" if data_quality_score > 95 else "Needs Attention")
    with col3:
        st.metric("Correlation Strength", f"{correlation_value * 100:.1f}%",
                 "Strong" if correlation_value > 0.7 else "Moderate" if correlation_value > 0.4 else "Weak")

    # Detailed Analysis Expander
    with st.expander("🔬 View Detailed Analysis"):
        if df is not None:
            st.write("### 📈 Statistical Overview")
            st.dataframe(df.describe())
            
            if len(df.select_dtypes(include=['number']).columns) >= 2:
                st.write("### 🔗 Correlation Matrix")
                corr_matrix = df.select_dtypes(include=['number']).corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlBu', vmin=-1, vmax=1))

    st.caption("⚙️ Generated using Hugging Face DistilBART — 100% Free and Offline")
    

