from fpdf import FPDF
import pandas as pd
import datetime
import os

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

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(26, 188, 156)
        self.cell(0, 10, 'DataVista - AI Insight Report', align='C', ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 9)
        self.set_text_color(169, 169, 169)
        self.cell(0, 10, f'Generated on {datetime.date.today().strftime("%B %d, %Y")}', 0, 0, 'C')

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 13)
        self.set_text_color(0, 191, 166)
        self.cell(0, 10, f"{title}", ln=True)
        self.ln(3)
        self.set_font('Arial', '', 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 8, content)
        self.ln(5)

def generate_combined_report(summary, df):
    """Generate a comprehensive PDF report combining AI insights and data analysis."""
    import tempfile
    import os
    
    # Clean the summary text to handle special characters
    summary = clean_text_for_pdf(summary)
    
    # Create PDF report
    pdf = PDFReport()
    pdf.add_page()
    
    # Dataset Overview
    if df is not None:
        rows, cols = df.shape
        missing_values = df.isna().sum().sum()
        duplicates = df.duplicated().sum()
        overview = f"""
        Dataset Statistics:
        - Total Records: {rows:,}
        - Total Features: {cols}
        - Missing Values: {missing_values:,}
        - Duplicate Rows: {duplicates:,}
        """
        pdf.add_section("Dataset Overview", overview)
    
    # AI Insights
    pdf.add_section("AI-Generated Insights", summary)
    
    # Data Types Analysis
    if df is not None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        data_types = f"""
        Numeric Features ({len(numeric_cols)}):
        {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
        
        Categorical Features ({len(categorical_cols)}):
        {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}
        """
        pdf.add_section("Data Types Analysis", data_types)
    
        # Basic Statistics
        stats = df.describe().round(2)
        stats_text = "Statistical Summary:\n\n"
        for col in stats.columns[:5]:  # Limit to first 5 columns
            stats_text += f"{col}:\n"
            for stat, value in stats[col].items():
                stats_text += f"- {stat}: {value}\n"
            stats_text += "\n"
        pdf.add_section("Statistical Analysis", stats_text)
    
    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, 'DataVista_Full_Report.pdf')
    pdf.output(file_path)
    
    return file_path
