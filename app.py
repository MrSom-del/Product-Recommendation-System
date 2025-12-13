import streamlit as st, pandas as pd, joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🛒 Content‑Based Product Recommender", layout="wide")
st.title("🛒 Content‑Based Product Recommendation")


df = pd.read_csv("artefacts/products_clean.csv")
vectorizer = joblib.load("artefacts/vectorizer.joblib")

try:
    tfidf_matrix = joblib.load("artefacts/tfidf_matrix.joblib")
except FileNotFoundError:
    tfidf_matrix = vectorizer.transform(df["text"])


product_names = df["product"].tolist()
query = st.selectbox("Pick a product you like", options=product_names)

if st.button("Show similar products"):
    idx = df.index[df["product"] == query][0]
    query_vec = tfidf_matrix[idx]


    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[-11:][::-1]   

    results = (
        df.iloc[top_idx][["product", "brand", "category", "sub_category"]]
        .assign(similarity=sims[top_idx].round(3))
        .iloc[1:]                           
        .reset_index(drop=True)
    )

    st.write("### Recommended items")
    st.dataframe(results)
