import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Page configuration
st.set_page_config(page_title="AI Sentiment Pipeline", layout="wide")
nltk.download('vader_lexicon', quiet=True)

# --- BACKEND LOGIC (From your script) ---
def scrape_and_analyze(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        elements = soup.find_all("div", class_="quote")
        
        records = []
        sia = SentimentIntensityAnalyzer()
        
        for item in elements:
            text = item.find("span", class_="text").text.strip() if item.find("span", class_="text") else ""
            author = item.find("small", class_="author").text.strip() if item.find("small", class_="author") else "Anonymous"
            if text:
                scores = sia.polarity_scores(text)
                compound = scores['compound']
                sentiment = 'Positive' if compound >= 0.05 else ('Negative' if compound <= -0.05 else 'Neutral')
                records.append({"User": author, "Review": text, "Compound": compound, "Sentiment": sentiment})
        
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- FRONTEND DASHBOARD LAYOUT ---
st.title("📊 Real-Time Automated Review Sentiment Analyzer")
st.markdown("This intelligent pipeline extracts raw website HTML data and passes it through a VADER NLP model to instantly evaluate consumer emotional trends.")
st.write("---")

# Sidebar
st.sidebar.header("Pipeline Configurations")
target_url = st.sidebar.text_input("Target Web URL", value="https://quotes.toscrape.com/")
run_pipeline = st.sidebar.button("🚀 Run Automated NLP Pipeline")

if run_pipeline:
    with st.spinner("Connecting to target server, extracting DOM, and processing NLP matrices..."):
        df = scrape_and_analyze(target_url)
        
    if not df.empty:
        # KPI Metric Cards
        total_reviews = len(df)
        pos_pct = (df['Sentiment'] == 'Positive').sum() / total_reviews * 100
        neg_pct = (df['Sentiment'] == 'Negative').sum() / total_reviews * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews Extracted", f"{total_reviews} rows")
        col2.metric("Positive Sentiment", f"{pos_pct:.1f}%", delta=f"{pos_pct:.1f}%")
        col3.metric("Negative Sentiment", f"{neg_pct:.1f}%", delta=f"-{neg_pct:.1f}%", delta_color="inverse")
        
        st.write("### 📦 Processed Dataset Matrix")
        st.dataframe(df, use_container_width=True)
        
        # Visualizations
        st.write("### 📈 Statistical Distribution Breakdown")
        chart_data = df['Sentiment'].value_counts()
        st.bar_chart(chart_data)
        
        # Download Button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned CSV Dataset",
            data=csv_data,
            file_name="sentiment_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("No structural records found. Please verify target HTML tags.")