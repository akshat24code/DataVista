import pandas as pd
import streamlit as st
import requests
import re
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()


# ----------------------------------------------------------
# NVIDIA API Summarization
# ----------------------------------------------------------
def generate_ai_summary_via_nvidia(narrative: str, api_key: str) -> str:
    """Use NVIDIA LLM API (Llama-4 Maverick) to summarize dataset description."""
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    stream = False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if stream else "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert data analyst. Summarize dataset statistics in a human-readable, structured format."
            },
            {
                "role": "user",
                "content": narrative
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": stream
    }

    try:
        response = requests.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        st.warning(f"⚠️ NVIDIA API summarization failed: {e}")
        return narrative


# ----------------------------------------------------------
# Dataset Summary Generator
# ----------------------------------------------------------
def generate_data_summary(df: pd.DataFrame) -> str:
    """Generate summary stats and send to NVIDIA LLM API for AI enhancement."""
    # Ask for API key only once from sidebar
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        st.error("⚠️ NVIDIA API key not found. Please add it to your .env file as NVIDIA_API_KEY.")
        st.stop()


    rows, cols = df.shape
    missing_values = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    missing_ratio = round((missing_values / (rows * cols)) * 100, 2) if rows * cols > 0 else 0

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

    corr_text = "No strong correlations detected."
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        corr_values = corr.unstack().sort_values(ascending=False)
        corr_values = corr_values[(corr_values < 0.999) & (corr_values > 0.4)]
        if not corr_values.empty:
            top_pair = corr_values.index[0]
            corr_value = round(corr_values.iloc[0], 2)
            corr_text = f"The strongest correlation (r = {corr_value}) is between `{top_pair[0]}` and `{top_pair[1]}`."

    # Construct dataset narrative
    narrative = f"""
    Dataset Overview:
    - {rows} rows, {cols} columns
    - {missing_values} missing values ({missing_ratio}%)
    - {duplicates} duplicate rows

    Numeric Insights:
    - Numeric columns: {', '.join(num_cols[:5]) + (' and more' if len(num_cols) > 5 else '')}
    - {corr_text}

    Categorical Insights:
    - Key categorical columns: {', '.join(cat_cols[:5]) + (' and more' if len(cat_cols) > 5 else '')}

    Data Health:
    - {'No duplicates detected.' if duplicates == 0 else f"{duplicates} duplicates found."}
    - {missing_values} missing values need attention.
    """

    # Use NVIDIA API if key provided
    if api_key:
        return generate_ai_summary_via_nvidia(narrative, api_key)
    else:
        st.info("🔑 Please provide your NVIDIA API key in the sidebar to enable AI summarization.")
        return narrative


# ----------------------------------------------------------
# AI Summary Display (UI Styling)
# ----------------------------------------------------------
def display_ai_summary(summary_text: str, df=None):
    """Beautifully display AI insights with color sections."""
    st.markdown("## ✨ AI Insight Summary")

    # Style definitions
    st.markdown("""
        <style>
            .insight-card {
                background-color: #0f1a25;
                color: #f1f1f1;
                padding: 25px;
                border-radius: 16px;
                border: 1px solid #00bfa6;
                box-shadow: 0px 0px 10px rgba(0,191,166,0.4);
                margin-bottom: 20px;
                line-height: 1.8;
                font-size: 17px;
            }
            .section-title {
                font-size: 22px;
                font-weight: 600;
                color: #00bfa6;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .metric {
                margin-left: 20px;
                color: #cfd8dc;
            }
        </style>
    """, unsafe_allow_html=True)

    # 🧩 Split the NVIDIA text by section headers (###)
    sections = re.split(r"###\s*", summary_text)
    formatted_html = ""

    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split("\n")
        title = lines[0].replace("**", "").strip()
        content = "<br>".join(lines[1:])
        formatted_html += f"""
        <div class="insight-card">
            <div class="section-title">📘 {title}</div>
            <div class="metric">{content}</div>
        </div>
        """

    st.markdown(formatted_html, unsafe_allow_html=True)

    if df is not None:
        with st.expander("🔬 View Dataset Overview"):
            st.dataframe(df.describe())

    st.caption("⚙️ Generated using NVIDIA Llama-4 Maverick LLM API")
