# AI-Powered CSV Data Analysis Tool

A Streamlit web app that takes any CSV file, profiles the data automatically,
and uses an LLM (GPT-OSS via the Groq API) to write a report with insights
and recommendations.

![App screenshot](screenshot.png)

## Features
- Upload any CSV and see a preview, key metrics, and descriptive statistics
- Data-quality checks: missing values, duplicate rows, data types
- LLM-generated report: overview, key observations, data-quality issues,
  insights, and recommendations
- Five auto-generated charts: distributions, correlation heatmap,
  category counts, outlier boxplot, average by category
- Downloadable TXT report

## Tech Stack
Python, Streamlit, Pandas, Matplotlib, Seaborn, Groq API (GPT-OSS)

## How It Works
1. Pandas computes the dataset summary and statistics.
2. The summary is sent to the Groq API with a structured prompt.
3. The LLM returns the written analysis.
4. Matplotlib and Seaborn draw the charts, and Streamlit displays everything.

## Setup
```bash
git clone https://github.com/vaish-navi1/csv-data-analysis-tool.git
cd csv-data-analysis-tool
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. Get a free API key at https://console.groq.com/keys
2. Paste the key in the app sidebar (or set the `GROQ_API_KEY` environment variable)
3. Upload a CSV (`students.csv` is included as a sample)
4. Click **Run AI Analysis Agent**

## Note
The API key is never stored in the code. The steps run in a fixed pipeline;
the LLM is used for generating the written analysis.
