<h1 align="center">📊 DataVista — AI Insight Generator</h1>
<p align="center">
  <i>Transform raw data into smart AI-driven insights with NVIDIA’s Llama 4 Maverick Model.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Powered%20by-NVIDIA%20LLMs-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Language-Python-3670A0?style=for-the-badge&logo=python&logoColor=yellow"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

## 🚀 Overview

** Link ** - https://akshat24code-datavista-app-gnuiag.streamlit.app/
**DataVista** is a Streamlit-based AI web app that automatically analyzes any dataset you upload and generates:
- A **comprehensive data summary**
- **Correlation & visual insights**
- An **AI-powered narrative report** using NVIDIA’s Llama 4 Maverick model

Whether you're a data scientist, analyst, or student — DataVista helps you quickly understand your dataset’s health, structure, and trends.

---

## 🧩 Features

✅ **Upload CSV/Excel datasets** with one click  
✅ **AI-generated summaries** powered by NVIDIA LLM API  
✅ **Correlation heatmaps** to identify key relationships  
✅ **Visual analytics** with Seaborn & Matplotlib  
✅ **PDF report generator** for sharing insights  
✅ **Clean, dark UI** for a professional look  

---

## 🧠 Tech Stack

| Category | Technology |
|-----------|-------------|
| Frontend / UI | [Streamlit](https://streamlit.io) |
| AI Model | [NVIDIA Llama 4 Maverick 17B](https://developer.nvidia.com/) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Backend Logic | Python |
| Environment | `.env` or `secrets.toml` for secure API key handling |

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/DataVista.git
cd DataVista
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate      # On Windows
source .venv/bin/activate     # On macOS/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### Add your NVIDIA API key

Option A — Local .env file

Create a .env file in your project root:

```bash
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
```

Option B — Streamlit Secrets (recommended)

Create .streamlit/secrets.toml and add:

```bash
NVIDIA_API_KEY = "nvapi-xxxxxxxxxxxxxxxxxxxxxxxx"
``` 

### 🧾 Run the App
```bash
streamlit run app.py
```

---

## 🧰 Folder Structure
```
DataVista/
│
├── app.py                         # Main Streamlit app
├── utils/
│   ├── ai_insight_1.py            # NVIDIA AI summarizer logic
│   ├── data_cleaner.py            # EDA and data cleaning module
│   ├── report_generator.py        # PDF generator
│
├── .streamlit/
│   └── secrets.toml               # Secure API key storage
│
├── requirements.txt
└── README.md
```

## 🧠 Future Enhancements
 - Add support for multi-model AI backends (OpenAI, HuggingFace)
 - Add time-series insight detection
 - Integrate with Power BI/Tableau connectors


---

<p align="center"> Made with ❤️ using <b>Streamlit + NVIDIA LLM API</b><br> <img src="https://img.shields.io/badge/AI%20Insight%20Powered%20By-NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/> </p> ```
