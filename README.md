🧠 AI Echo – Sentiment Analysis System

An end-to-end Sentiment Analysis & Insight Generation application built using Machine Learning, NLP, and Streamlit.
This project analyzes ChatGPT-style user reviews to predict sentiment and extract meaningful business insights.

📌 Project Overview

User reviews contain valuable feedback but are often unstructured and difficult to analyze.
AI Echo automatically:

Classifies reviews into Positive / Neutral / Negative

Identifies sentiment trends and user pain points

Provides interactive dashboards for insights

Supports business decision-making using data

🧩 Project Workflow
1️⃣ Data Collection

Dataset: ChatGPT-style user reviews

Features include:

Review text

Rating (1–5)

Platform (Web / Mobile)

ChatGPT version

Location

Verified purchase status

2️⃣ Data Cleaning

Removed missing and duplicate values

Converted ratings to sentiment:

4–5 → Positive

3 → Neutral

1–2 → Negative

Cleaned text (lowercase, removed symbols, extra spaces)

3️⃣ Feature Engineering

Applied TF-IDF Vectorization to convert text into numerical features

Encoded sentiment labels for supervised learning

4️⃣ Model Training

Logistic Regression (Primary model – deployed)

LSTM (Deep Learning) – experimented offline

Model evaluation using:

Accuracy

Classification Report

Saved trained models and vectorizers for reuse

5️⃣ Exploratory Data Analysis (EDA)

Answered 10 key business questions, including:

Overall sentiment distribution

Sentiment vs rating mismatch

Keywords associated with each sentiment

Sentiment trends over time

Verified vs non-verified user sentiment

Review length vs sentiment

Location-wise sentiment

Platform-wise sentiment (Web vs Mobile)

Version-wise sentiment

Common negative feedback themes

6️⃣ Streamlit Application

Interactive web app with:

🔮 Sentiment Prediction Page

📊 Insights Dashboard (EDA-based)

Clean UI with navigation sidebar

Fast inference using pre-trained ML models

7️⃣ Deployment

Deployed using Streamlit Cloud

Dependency management via:

requirements.txt

runtime.txt (Python 3.11.2)

🛠️ Tech Stack
Category	Tools
Language	Python
Data	Pandas, NumPy
ML	Scikit-learn
DL (Experiment)	TensorFlow (LSTM)
Visualization	Matplotlib
Web App	Streamlit
Deployment	Streamlit Cloud
