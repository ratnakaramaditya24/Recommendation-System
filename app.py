import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Product Recommendation System", layout="centered")

st.title("Product Recommendation System")
st.write("Cluster-based Item-to-Item Recommendation using K-Means + KNN")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("rating_short.csv")
    df.drop(columns=['date'], inplace=True)
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

    pos_frac = df.assign(pos=(df['rating'] >= 4).astype(int)) \
                 .groupby('productid')['pos'].mean().reset_index()

    pos_frac.rename(columns={'pos': 'frac_positive'}, inplace=True)

    product_feats = product_feats.merge(pos_frac, on='productid', how='left')

    product_feats['log_num_ratings'] = np.log1p(product_feats['num_ratings'])

    return product_feats

product_feats = build_product_features(df)

features = ['avg_rating', 'num_ratings', 'std_rating', 'frac_positive', 'log_num_ratings']

X = product_feats[features]

# -----------------------------
# Scaling + Clustering
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
        nn = NearestNeighbors(n_neighbors=min(6, len(data)), metric='euclidean')
        nn.fit(data)
        nn_models[cl] = (nn, list(ids))
    else:
        nn_models[cl] = None

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_similar(product_id, top_n=5):

    if product_id not in product_feats.index:
        return None, None

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
# Streamlit UI
# -----------------------------
st.subheader("🔎 Enter Product ID")

product_id = st.text_input("Product ID (example: 1400501776)")

if st.button("Get Recommendations"):

    cluster, recommendations = recommend_similar(product_id)

    if cluster is None:
        st.error("Product ID not found in dataset.")
    else:
        st.success(f"Product belongs to Cluster: {cluster}")

        if len(recommendations) == 0:
            st.warning("Not enough similar products in this cluster.")
        else:
            st.write("### Top-5 Recommended Products")
            for i, rec in enumerate(recommendations, start=1):
                st.write(f"{i}. {rec}")
