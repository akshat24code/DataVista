import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
