import pandas as pd
import numpy as np
import streamlit as st
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Optimize Presentation Page View Layout
st.set_page_config(page_title="Course Recommender Engine", layout="wide")
st.title("🎓 Institutional Course Recommender")
st.caption("High-Performance Single-Process In-Memory Computing Paradigm")

# 2. Hour 1 Optimization: Cached Core Proximity Engine Pipeline
@st.cache_resource
def load_and_compute_matrix():
    # Load your reliable dataset asset directly into RAM addresses [cite: 139, 244]
    try:
        catalog_df = pd.read_csv("courses.csv")
    except FileNotFoundError:
        st.error("🚨 Critical Error: 'courses.csv' data file asset was not detected in the working folder path.")
        st.stop()
        
    # Standardize data: Fill missing description lines to prevent vectorizer code exceptions
    catalog_df['meta'] = catalog_df['meta'].fillna('')
    
    # NLP Engine Feature Vector Space Transformation Engine [cite: 254]
    # This automatically runs lowercase standardization, tokenization loops, and stop-word filtering [cite: 282, 287, 292]
    vec = TfidfVectorizer(stop_words='english')
    matrix = vec.fit_transform(catalog_df['meta'])
    
    # Generate the comprehensive non-Euclidean angular similarity grid matrix [cite: 330, 343]
    sim_grid = cosine_similarity(matrix, matrix)
    return catalog_df, sim_grid

# Boot application context and cache matching matrices in local memory spaces [cite: 241, 253]
courses_df, proximity_matrix = load_and_compute_matrix()

# 3. Reactive UI Dropdown Selection Mapping [cite: 260, 264]
st.subheader("Select an Anchor Curriculum Module")
selected_title = st.selectbox("Choose a baseline course from the platform catalog index:", courses_df['title'].tolist())

if selected_title:
    # Capture event and pull index mapping directly from RAM data tier rows [cite: 251, 264]
    idx = courses_df[courses_df['title'] == selected_title].index[0]
    
    # Benchmark computational calculation processing latency [cite: 441]
    start_time = time.perf_counter()
    similarity_scores = list(enumerate(proximity_matrix[idx]))
    
    # Sort matching nodes in descending alignment order, removing the anchor entry itself
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:4]
    execution_latency = (time.perf_counter() - start_time) * 1000 # Convert to milliseconds

    st.markdown("---")
    st.subheader("Recommended Next-Sequence Educational Modules")
    
    # Render layout dashboard columns reactively based on calculations [cite: 265]
    cols = st.columns(len(sorted_scores))
    for col, (recommend_idx, score) in zip(cols, sorted_scores):
        with col:
            st.info(f"**{courses_df.iloc[recommend_idx]['title']}**")
            st.metric(label="Thematic Alignment", value=f"{score*100:.1f}%")
            st.caption(f"Database Node ID: {courses_df.iloc[recommend_idx]['id']}")
            
    # System Status Performance Verification Tracker [cite: 441]
    st.markdown("---")
    st.caption(f"⚡ In-Memory Vector Compute Latency: {execution_latency:.4f} ms | Catalog Density: {len(courses_df)} records | Status: Optimal")
