import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.ai_insight import generate_data_summary, display_ai_summary
from utils.data_cleaner import create_eda_report
from utils.report_generator import generate_combined_report


# ----------------------------------------------------------
# Utility: Display AI summary if available
# ----------------------------------------------------------
def display_legacy_summary(summary, df):
    """Display legacy AI summary fallback."""
    if isinstance(summary, dict):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", summary.get("total_rows", 0))
            st.metric("Missing Values", summary.get("missing_values", 0))
        with col2:
            st.metric("Total Columns", summary.get("total_columns", 0))
            st.metric("Duplicate Rows", summary.get("duplicate_rows", 0))

        if summary.get("numeric_columns"):
            st.write("📊 **Numeric Columns:**", ", ".join(summary["numeric_columns"]))
        if summary.get("categorical_columns"):
            st.write("📝 **Categorical Columns:**", ", ".join(summary["categorical_columns"]))
    else:
        st.write(summary)


# ----------------------------------------------------------
# Streamlit Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="DataVista", layout="wide")

st.title("📊 DataVista — AI Insight Generator")
st.caption("Upload your dataset → choose a report → get instant insights")

# Sidebar
st.sidebar.header("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])

report_type = st.sidebar.selectbox(
    "Choose report type:",
    ["Data Summary", "Correlation Report", "Visual Analysis"],
)

df = None  # Initialize


# ----------------------------------------------------------
# File Upload & Data Preparation
# ----------------------------------------------------------
if uploaded_file:
    st.success("✅ File uploaded successfully!")

    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload only CSV or Excel files.")
            st.stop()

        # 🔹 Smart dtype correction (keeps numeric, date, string)
        df = df.convert_dtypes()

        # Fix mixed-type columns that break Arrow serialization
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, str)).any() and df[col].apply(lambda x: not isinstance(x, str)).any():
                df[col] = df[col].astype(str)

        # Convert nullable numeric dtypes to regular float
        for col in df.select_dtypes(include=["Int64", "Float64"]).columns:
            df[col] = df[col].astype(float)

        # Auto-convert potential datetime columns
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore", infer_datetime_format=True)
                except Exception:
                    pass

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.info("👆 Upload a file to begin.")
    st.stop()


# ----------------------------------------------------------
# 1️⃣ DATA SUMMARY
# ----------------------------------------------------------
if report_type == "Data Summary":
    st.subheader("📘 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isna().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.write("### 🧱 Data Types")
    st.dataframe(df.dtypes.rename("Type"))

    st.write("### ⚠️ Missing Values by Column")
    missing = df.isna().sum()
    st.dataframe(
        pd.DataFrame(
            {
                "Column": missing.index,
                "Missing Count": missing.values,
                "Missing %": (missing.values / len(df) * 100).round(2),
            }
        )
    )

    st.write("### 📊 Statistical Summary")
    try:
        st.dataframe(df.describe(include="all"))
    except Exception:
        st.warning("Could not display full summary due to mixed column types.")

    st.write("### 🤖 AI-Generated Summary")
    try:
        with st.spinner("Generating insights..."):
            summary = generate_data_summary(df)
            display_ai_summary(summary, df)

            if st.button("📄 Generate Full AI Report (PDF)"):
                with st.spinner("Preparing full AI Report..."):
                    file_path = generate_combined_report(summary, df)
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Report (PDF)",
                            data=f,
                            file_name="DataVista_Report.pdf",
                            mime="application/pdf",
                        )
    except Exception as e:
        st.warning(f"⚠️ AI summarization unavailable: {e}")


# ----------------------------------------------------------
# 2️⃣ CORRELATION REPORT
# ----------------------------------------------------------
elif report_type == "Correlation Report":
    st.subheader("📈 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.warning("At least two numeric columns are needed for correlation.")
    else:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


# ----------------------------------------------------------
# 3️⃣ VISUAL ANALYSIS
# ----------------------------------------------------------
elif report_type == "Visual Analysis":
    st.subheader("📊 Visual Analysis")

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    x_axis = st.selectbox("X-axis:", all_cols)
    y_axis = st.selectbox("Y-axis (numeric):", num_cols)

    if st.button("Generate Chart"):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
