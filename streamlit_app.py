# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import process
import warnings
warnings.filterwarnings("ignore")

# === Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("/content/books.csv", on_bad_lines='skip')
    df.drop_duplicates(subset="title", inplace=True)
    df.dropna(subset=["average_rating", "ratings_count", "language_code"], inplace=True)
    df['  num_pages'] = pd.to_numeric(df['  num_pages'], errors='coerce')
    df['text_reviews_count'] = pd.to_numeric(df['text_reviews_count'], errors='coerce')
    df.dropna(subset=['  num_pages', 'text_reviews_count'], inplace=True)
    df['average_rating'] = df['average_rating'].astype(float)
    df['ratings_count'] = df['ratings_count'].astype(int)
    df['text_reviews_count'] = df['text_reviews_count'].astype(int)
    return df

def prepare_features(df):
    df2 = df.copy()
    df2.loc[(df2['average_rating'] <= 1), 'rating_between'] = "0-1"
    df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "1-2"
    df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "2-3"
    df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "3-4"
    df2.loc[(df2['average_rating'] > 4), 'rating_between'] = "4-5"
    rating_dummies = pd.get_dummies(df2['rating_between'])
    lang_dummies = pd.get_dummies(df2['language_code'])
    features = pd.concat([
        rating_dummies,
        lang_dummies,
        df2[['average_rating', 'ratings_count', '  num_pages', 'text_reviews_count']]
    ], axis=1)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return df2, scaled_features

def recommend_books(book_name, df2, features_scaled):
    model = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features_scaled)
    distances, indices = model.kneighbors(features_scaled)
    match = process.extractOne(book_name, df2['title'])
    if not match or match[1] < 60:
        return None, []
    matched_title = match[0]
    idx = df2[df2['title'] == matched_title].index[0]
    recs = [df2.iloc[i]['title'] for i in indices[idx]]
    return matched_title, recs

# === Streamlit UI ===
st.title("📚 Book Recommendation System")
st.write("Type a book title to get similar books.")

df = load_data()
df2, features_scaled = prepare_features(df)

book_input = st.text_input("Enter a book title:")

if book_input:
    matched_title, recommendations = recommend_books(book_input, df2, features_scaled)
    if recommendations:
        st.success(f"Top recommendations based on 📘 **{matched_title}**:")
        for book in recommendations:
            st.markdown(f"- {book}")
    else:
        st.error("❌ No close match found. Try another title.")
