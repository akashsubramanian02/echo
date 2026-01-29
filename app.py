import streamlit as st
import pandas as pd
import pickle
from collections import Counter

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Echo – Sentiment Analysis",
    layout="wide"
)

st.title("🧠 AI Echo – Sentiment Analysis")

# -------------------------------------------------
# LOAD MODEL & DATA
# -------------------------------------------------
@st.cache_resource
def load_model():
    with open("models/logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@st.cache_data
def load_data():
    return pd.read_csv("data/clean_reviews.csv")

model, vectorizer = load_model()
df = load_data()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Sentiment Prediction", "📊 Sentiment Insights"]
)

# -------------------------------------------------
# HOME
# -------------------------------------------------
if page == "🏠 Home":
    st.subheader("📘 Project Overview")

    st.write("""
    **AI Echo** analyzes user reviews using **NLP & Machine Learning**  
    to understand customer sentiment and behavior.

    **Models Used**
    - Logistic Regression (Production)
    - LSTM (Experimental – Notebook)

    **Sentiment Classes**
    - Positive
    - Neutral
    - Negative
    """)

# -------------------------------------------------
# SENTIMENT PREDICTION
# -------------------------------------------------
elif page == "🔮 Sentiment Prediction":
    st.subheader("🔮 Predict Review Sentiment")

    emoji_map = {
        "Positive": "😊 Positive",
        "Neutral": "😐 Neutral",
        "Negative": "😠 Negative"
    }

    review = st.text_area(
        "✍️ Enter a review",
        height=150,
        placeholder="Type your review here..."
    )

    if st.button("Predict Sentiment"):
        if review.strip() == "":
            st.warning("Please enter some text.")
        else:
            vec = vectorizer.transform([review])
            prediction = model.predict(vec)[0]
            st.success(f"**Predicted Sentiment:** {emoji_map[prediction]}")

# -------------------------------------------------
# SENTIMENT INSIGHTS (10 QUESTIONS)
# -------------------------------------------------
elif page == "📊 Sentiment Insights":
    st.subheader("📊 Sentiment Analysis Insights")

    # 1️⃣ Overall Sentiment
    st.markdown("### 1️⃣ What is the overall sentiment of user reviews?")
    sentiment_counts = df["sentiment"].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_counts)

    # 2️⃣ Sentiment vs Rating
    st.markdown("### 2️⃣ How does sentiment vary by rating?")
    st.dataframe(pd.crosstab(df["rating"], df["sentiment"]))

    # 3️⃣ Keywords per Sentiment
    st.markdown("### 3️⃣ Keywords associated with each sentiment")
    sentiment_choice = st.selectbox(
        "Select sentiment",
        ["Positive", "Neutral", "Negative"]
    )
    text = " ".join(df[df["sentiment"] == sentiment_choice]["clean_review"])
    keywords = Counter(text.split()).most_common(15)
    st.dataframe(pd.DataFrame(keywords, columns=["Keyword", "Frequency"]))

    # 4️⃣ Sentiment Over Time
    st.markdown("### 4️⃣ How has sentiment changed over time?")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    trend = df.groupby(df["date"].dt.to_period("M"))["sentiment"].value_counts().unstack()
    st.line_chart(trend)

    # 5️⃣ Verified Users
    st.markdown("### 5️⃣ Do verified users leave different sentiment?")
    st.dataframe(pd.crosstab(df["verified_purchase"], df["sentiment"]))

    # 6️⃣ Review Length
    st.markdown("### 6️⃣ Are longer reviews more positive or negative?")
    df["review_length"] = df["clean_review"].str.split().apply(len)
    st.bar_chart(df.groupby("sentiment")["review_length"].mean())

    # 7️⃣ Location-wise Sentiment
    st.markdown("### 7️⃣ Which locations show strongest sentiment?")
    st.dataframe(df.groupby("location")["sentiment"].value_counts().unstack())

    # 8️⃣ Platform-wise Sentiment
    st.markdown("### 8️⃣ Is sentiment different across platforms?")
    st.bar_chart(pd.crosstab(df["platform"], df["sentiment"]))

    # 9️⃣ Version-wise Sentiment
    st.markdown("### 9️⃣ Which ChatGPT versions impact sentiment?")
    st.dataframe(pd.crosstab(df["version"], df["sentiment"]))

    # 🔟 Negative Feedback Themes
    st.markdown("### 🔟 Most common negative feedback themes")
    neg_text = " ".join(df[df["sentiment"] == "Negative"]["clean_review"])
    neg_words = Counter(neg_text.split()).most_common(20)
    st.dataframe(pd.DataFrame(neg_words, columns=["Theme / Keyword", "Frequency"]))
