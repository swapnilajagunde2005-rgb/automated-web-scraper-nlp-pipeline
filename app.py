import re
from collections import Counter
import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Automatically download required NLTK datasets on launch
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

setup_nltk()

# --- Page Layout & Config ---
st.set_page_config(page_title="Web Scraper & NLP Pipeline", layout="wide")
st.title("🌐 Automated Web Scraper & NLP Pipeline")
st.write("Extract content from any public webpage and run real-time Natural Language Processing.")

# --- Interactive Input ---
raw_input_url = st.text_input("Enter Webpage URL:", value="https://en.wikipedia.org/wiki/Natural_language_processing")

if st.button("Scrape & Analyze", type="primary"):
    # Sanitize and clean URL input (removes extra spaces, parentheses, quotes)
    clean_url = raw_input_url.strip(" ()'\"")
    if clean_url and not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    if not clean_url:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner(f"Scraping {clean_url} and analyzing text..."):
            try:
                # 1. Web Scraping
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                }
                response = requests.get(clean_url, headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = soup.find_all("p")
                raw_text = " ".join([p.get_text() for p in paragraphs])

                if not raw_text.strip():
                    st.error("No readable paragraph text could be extracted from this webpage.")
                    st.stop()

                # 2. NLP Processing
                clean_text = re.sub(r'[^a-zA-Z\s]', '', raw_text).lower()
                tokens = word_tokenize(clean_text)
                
                stop_words = set(stopwords.words("english"))
                filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
                
                # Word Frequency & Sentiment Analysis
                word_counts = Counter(filtered_tokens).most_common(10)
                sia = SentimentIntensityAnalyzer()
                sentiment = sia.polarity_scores(raw_text)

                # 3. Streamlit Display Output
                st.success("Analysis complete!")
                st.divider()

                # Metric Cards
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Paragraphs Found", len(paragraphs))
                col2.metric("Total Words", len(tokens))
                col3.metric("Filtered Keywords", len(set(filtered_tokens)))

                compound = sentiment['compound']
                if compound >= 0.05:
                    sent_label = "Positive 😃"
                elif compound <= -0.05:
                    sent_label = "Negative 🙁"
                else:
                    sent_label = "Neutral 😐"
                col4.metric("Overall Sentiment", sent_label)

                # Data Sections
                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.subheader("📊 Top 10 Keywords")
                    st.dataframe(
                        {"Keyword": [w for w, _ in word_counts], "Frequency": [c for _, c in word_counts]},
                        use_container_width=True
                    )

                with res_col2:
                    st.subheader("🎭 Sentiment Distribution")
                    st.json({
                        "Positive Score": f"{sentiment['pos']:.2%}",
                        "Neutral Score": f"{sentiment['neu']:.2%}",
                        "Negative Score": f"{sentiment['neg']:.2%}",
                        "Compound Intensity": sentiment['compound']
                    })

                with st.expander("📄 View Scraped Raw Text"):
                    st.write(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch webpage: {e}")
            except Exception as e:
                st.error(f"Pipeline error: {e}")