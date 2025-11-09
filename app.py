import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.ai_insight import generate_data_summary, display_ai_summary
from utils.data_cleaner import create_eda_report
from utils.report_generator import generate_combined_report



def display_legacy_summary(summary, df):
    """Display the AI-generated summary in a structured way."""
    if isinstance(summary, dict):
        # Display key metrics in columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", summary.get('total_rows', 0))
            st.metric("Missing Values", summary.get('missing_values', 0))
        with col2:
            st.metric("Total Columns", summary.get('total_columns', 0))
            st.metric("Duplicate Rows", summary.get('duplicate_rows', 0))
        
        # Show column types
        if summary.get('numeric_columns'):
            st.write("📊 **Numeric Columns:**", ", ".join(summary['numeric_columns']))
        if summary.get('categorical_columns'):
            st.write("📝 **Categorical Columns:**", ", ".join(summary['categorical_columns']))
    else:
        # If summary is a string, display it directly
        st.write(summary)


st.set_page_config(page_title="DataVista", layout="wide")

st.title("📊 DataVista — AI Insight Generator")
st.caption("Upload data → choose report type → get insights")

# Sidebar navigation
st.sidebar.header("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])

report_type = st.sidebar.selectbox(
    "Choose the type of report you want:",
    ["Data Summary", "Correlation Report", "Visual Analysis"]
)

# Initialize df as None
df = None

# Check if file is uploaded and process it
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload only CSV or Excel files.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()
else:
    st.info("⬅️ Please upload a file to begin.")
    st.stop()

# -------------------------
# 1️⃣ DATA SUMMARY
# -------------------------
if report_type == "Data Summary":
    st.subheader("📘 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isna().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.write("### 🧱 Data Types:")
    st.write(df.dtypes)

    st.write("### 🔍 Missing Values by Column:")
    st.write(df.isna().sum())

    st.write("### 📊 Statistical Summary:")
    st.write(df.describe())

    st.write("### 🤖 AI-Generated Summary:")
    try:
        with st.spinner("Generating insights..."):
            summary = generate_data_summary(df)
            display_ai_summary(summary, df)  # Using enhanced version from ai_insight.py
            
            # Add full AI report generation button
            if st.button("📄 Generate Full AI Report (PDF)"):
                with st.spinner("Generating detailed AI Report..."):
                    file_path = generate_combined_report(summary, df)
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Full Report (PDF)",
                            data=f,
                            file_name="DataVista_Full_Report.pdf",
                            mime="application/pdf"
                        )
    except Exception as e:
        st.warning(f"Could not generate AI summary: {e}")


# -------------------------
# 2️⃣ CORRELATION REPORT
# -------------------------
elif report_type == "Correlation Report":
    st.subheader("📈 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        st.warning("Need at least two numeric columns for correlation.")
    else:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -------------------------
# 3️⃣ VISUAL ANALYSIS
# -------------------------
elif report_type == "Visual Analysis":
    st.subheader("📊 Visual Analysis")

    st.write("Choose columns to visualize:")
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    x_axis = st.selectbox("X-axis:", all_cols)
    y_axis = st.selectbox("Y-axis:", num_cols)

    if st.button("Generate Chart"):
        fig, ax = plt.subplots()
        sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
