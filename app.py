import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-20b"   # or "openai/gpt-oss-120b" for deeper analysis

st.set_page_config(page_title="CSV Data Analysis Agent", layout="wide")
st.title("CSV Data Analysis Agent")
st.write("Upload any CSV — the AI agent automatically analyzes it and generates a full report.")

st.sidebar.header("Settings")
# Uses the GROQ_API_KEY environment variable if set, otherwise the sidebar box
_env_key = os.environ.get("GROQ_API_KEY", "").strip()
_typed_key = st.sidebar.text_input(
    "Groq API Key", type="password", placeholder="Enter your free Groq API key"
).strip()
GROQ_API_KEY = _typed_key or _env_key
st.sidebar.markdown("[Get free API key here](https://console.groq.com/keys)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


# ---------------------------------------------------------------
# Step 1 - Summary
# ---------------------------------------------------------------
def collect_summary(df):
    summary = f"""
Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns
Column Names: {list(df.columns)}
Data Types:
{df.dtypes.to_string()}
First 5 Rows:
{df.head().to_string()}
Statistical Summary:
{df.describe().to_string()}
Missing Values Per Column:
{df.isnull().sum().to_string()}
Duplicate Rows: {df.duplicated().sum()}
    """
    return summary


# ---------------------------------------------------------------
# Step 2 - AI analysis
# ---------------------------------------------------------------
def run_ai_analysis(summary, client):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": """You are an expert Data Analyst.
Analyze the dataset summary and give a detailed report with these exact sections:

1. DATASET OVERVIEW
   - What this dataset is about
   - Type of data

2. KEY OBSERVATIONS (5 points)
   - Use actual column names and numbers from the data

3. DATA QUALITY ISSUES
   - Missing values count
   - Duplicates
   - Any anomalies found

4. TOP INSIGHTS (3 points)
   - Use actual values from the statistics provided

5. RECOMMENDATIONS
   - What actions should be taken

Be specific. Always use actual numbers and column names."""
            },
            {
                "role": "user",
                "content": f"Analyze this dataset:\n\n{summary}"
            }
        ],
        max_tokens=4000,   # reasoning models use some tokens for thinking
        temperature=0.7
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(
            "The model returned an empty answer. Try again, or switch MODEL_NAME "
            "to 'openai/gpt-oss-120b'."
        )
    return content


# ---------------------------------------------------------------
# Step 3 - Charts
# ---------------------------------------------------------------
def pick_value_column(numeric_cols):
    """Pick a numeric column that makes sense to average (skip Age / ID-like columns)."""
    skip = ("age", "id", "roll", "index")
    for col in numeric_cols:
        if not any(word in col.lower() for word in skip):
            return col
    return numeric_cols[0]


def generate_charts(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    charts_made = 0

    # Chart 1 - Distribution
    if len(numeric_cols) >= 1:
        st.write("#### Distribution of Numeric Columns")
        cols_to_plot = numeric_cols[:4]
        fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(5 * len(cols_to_plot), 4))
        if len(cols_to_plot) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols_to_plot):
            df[col].hist(ax=ax, bins=15, color='steelblue', edgecolor='white')
            ax.set_title(col, fontsize=11)
            ax.set_ylabel("Count")
        plt.suptitle("Distribution Plot", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        charts_made += 1

    # Chart 2 - Correlation Heatmap
    if len(numeric_cols) >= 2:
        st.write("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(9, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.5)
        ax.set_title("Correlation Between Numeric Columns", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        charts_made += 1

    # Chart 3 - Bar chart for first low-cardinality categorical column
    for col in categorical_cols:
        if df[col].nunique() <= 15:
            st.write(f"#### Value Counts — {col}")
            fig, ax = plt.subplots(figsize=(8, 4))
            df[col].value_counts().plot(kind='bar', ax=ax,
                                        color='coral', edgecolor='white')
            ax.set_title(f"Value Counts — {col}", fontsize=14)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            charts_made += 1
            break

    # Chart 4 - Boxplot
    if len(numeric_cols) >= 1:
        st.write("#### Boxplot — Outlier Detection")
        cols_to_box = numeric_cols[:5]
        fig, ax = plt.subplots(figsize=(10, 5))
        df[cols_to_box].boxplot(ax=ax, patch_artist=True)
        ax.set_title("Boxplot — Outlier Detection", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        charts_made += 1

    # Chart 5 - Average of a numeric column by category
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        cat_col = categorical_cols[-1]
        num_col = pick_value_column(numeric_cols)
        if df[cat_col].nunique() <= 10:
            st.write(f"#### Average {num_col} by {cat_col}")
            fig, ax = plt.subplots(figsize=(8, 4))
            df.groupby(cat_col)[num_col].mean().sort_values().plot(
                kind='barh', ax=ax, color='teal', edgecolor='white')
            ax.set_title(f"Average {num_col} by {cat_col}", fontsize=14)
            ax.set_xlabel(f"Average {num_col}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            charts_made += 1

    return charts_made


# ---------------------------------------------------------------
# Report download
# ---------------------------------------------------------------
def generate_report(df, analysis):
    report = f"""
CSV DATA ANALYSIS REPORT
Generated by AI Data Analysis Agent
{'='*60}

DATASET SUMMARY
---------------
Total Rows     : {df.shape[0]}
Total Columns  : {df.shape[1]}
Columns        : {', '.join(df.columns.tolist())}
Missing Values : {df.isnull().sum().sum()} total
Duplicate Rows : {df.duplicated().sum()}

{'='*60}
AI ANALYSIS REPORT
{'='*60}
{analysis}

{'='*60}
STATISTICAL SUMMARY
{'='*60}
{df.describe().to_string()}

{'='*60}
MISSING VALUES PER COLUMN
{'='*60}
{df.isnull().sum().to_string()}

{'='*60}
DATA TYPES
{'='*60}
{df.dtypes.to_string()}
    """
    return report


# ---------------------------------------------------------------
# Main app logic
# ---------------------------------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success(f"File uploaded successfully — {df.shape[0]} rows, {df.shape[1]} columns")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Numeric Columns", len(df.select_dtypes(include='number').columns))

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    if st.button("Run AI Analysis Agent", type="primary"):
        if not GROQ_API_KEY:
            st.error("Please enter your Groq API key in the sidebar first.")
        else:
            client = Groq(api_key=GROQ_API_KEY)

            with st.spinner("Step 1 of 3 — Collecting data summary..."):
                summary = collect_summary(df)
            st.success("Step 1 done — Data summary collected")

            analysis = None
            with st.spinner("Step 2 of 3 — AI agent analyzing your data..."):
                try:
                    analysis = run_ai_analysis(summary, client)
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")

            if analysis:
                st.success("Step 2 done — AI analysis complete")
                st.subheader("AI Analysis Report")
                st.markdown(analysis)

                st.subheader("Auto-Generated Charts")
                with st.spinner("Step 3 of 3 — Generating charts automatically..."):
                    charts_made = generate_charts(df)
                st.success(f"Step 3 done — {charts_made} charts generated")

                st.subheader("Download Full Report")
                report = generate_report(df, analysis)
                st.download_button(
                    label="Download Report as TXT",
                    data=report,
                    file_name="analysis_report.txt",
                    mime="text/plain"
                )

                st.balloons()
                st.success("Analysis complete! You can now upload any other CSV and run again.")

else:
    st.info("Upload the students.csv file provided to get started.")
    st.markdown("""
    **What this agent does:**
    - Reads any CSV file
    - Collects full data summary automatically
    - Sends it to Groq AI (GPT-OSS) for intelligent analysis
    - Generates 5 charts automatically
    - Downloads a full report
    """)