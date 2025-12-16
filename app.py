# ==============================
# Product Recommendation System
# Streamlit Deployment
# ==============================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Page Config (MUST BE FIRST)
# -----------------------------
st.set_page_config(
    page_title="Product Recommendation System",
    layout="wide"
)

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f8fafc, #eef2ff);
}

.title-text {
    font-size: 42px;
    font-weight: 700;
    color: #1e293b;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle-text {
    text-align: center;
    font-size: 18px;
    color: #475569;
    margin-bottom: 30px;
}

.input-card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.result-card {
    background-color: #ecfeff;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

.cluster-badge {
    display: inline-block;
    background-color: #2563eb;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    margin-bottom: 15px;
}

.product-item {
    background-color: #f1f5f9;
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 16px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title-text">Product Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Cluster-based Item-to-Item Recommendation using K-Means and KNN</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("rating_short.csv")
    df.drop(columns=["date"], inplace=True)
    return df

df = load_data()

# -----------------------------
# Feature Engineering
# -----------------------------
@st.cache_data
def build_product_features(df):
    product_feats = df.groupby('productid')['rating'].agg(
        avg_rating='mean',
        num_ratings='count',
        std_rating='std'
    ).reset_index()

    product_feats['std_rating'] = product_feats['std_rating'].fillna(0)

    pos_frac = (
        df.assign(pos=(df['rating'] >= 4).astype(int))
          .groupby('productid')['pos']
          .mean()
          .reset_index()
          .rename(columns={'pos': 'frac_positive'})
    )

    product_feats = product_feats.merge(pos_frac, on='productid', how='left')
    product_feats['log_num_ratings'] = np.log1p(product_feats['num_ratings'])

    return product_feats

product_feats = build_product_features(df)

features = [
    'avg_rating',
    'num_ratings',
    'std_rating',
    'frac_positive',
    'log_num_ratings'
]

X = product_feats[features]

# -----------------------------
# Scaling + K-Means Clustering
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
product_feats['cluster'] = kmeans.fit_predict(X_scaled)

product_feats.set_index('productid', inplace=True)

X_scaled_df = pd.DataFrame(
    X_scaled,
    index=product_feats.index,
    columns=features
)

# -----------------------------
# Build KNN Models per Cluster
# -----------------------------
nn_models = {}

for cl in sorted(product_feats['cluster'].unique()):
    ids = product_feats[product_feats['cluster'] == cl].index
    data = X_scaled_df.loc[ids]

    if len(data) >= 2:
        nn = NearestNeighbors(
            n_neighbors=min(6, len(data)),
            metric='euclidean'
        )
        nn.fit(data)
        nn_models[cl] = (nn, list(ids))
    else:
        nn_models[cl] = None

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_similar(product_id, top_n=5):

    if product_id not in product_feats.index:
        return None, []

    cl = int(product_feats.loc[product_id, 'cluster'])
    model_info = nn_models.get(cl)

    if model_info is None:
        return cl, []

    nn, id_list = model_info

    vec = X_scaled_df.loc[product_id].values.reshape(1, -1)
    distances, indices = nn.kneighbors(vec)

    recs = []
    for idx in indices[0]:
        pid = id_list[idx]
        if pid != product_id:
            recs.append(pid)
        if len(recs) == top_n:
            break

    return cl, recs

# -----------------------------
# UI – Input Section
# -----------------------------
st.markdown('<div class="input-card">', unsafe_allow_html=True)

st.markdown("### Enter Product ID")
product_id = st.text_input("Example: 1400501776")

get_rec = st.button("Get Recommendations")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# UI – Result Section
# -----------------------------
if get_rec:
    cluster, recommendations = recommend_similar(product_id)

    if cluster is None:
        st.error("Product ID not found in dataset.")
    else:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.markdown(
            f'<span class="cluster-badge">Cluster {cluster}</span>',
            unsafe_allow_html=True
        )

        st.markdown("### Top-5 Recommended Products")

        if len(recommendations) == 0:
            st.warning("Not enough similar products in this cluster.")
        else:
            for i, rec in enumerate(recommendations, start=1):
                st.markdown(
                    f'<div class="product-item">{i}. {rec}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)
