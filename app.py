import pandas as pd
import numpy as np
import streamlit as st
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration and Layout
st.set_page_config(page_title="Course Recommender Engine", layout="wide")
st.title("🎓 Institutional Course Recommender")
st.caption("High-Performance Single-Process In-Memory Computing Paradigm")

# 2. In-Memory Data Tier & NLP Pipeline Engine
@st.cache_resource
def load_and_compute_matrix():
    try:
        # Read your uploaded JSON data file asset directly into operational RAM
        catalog_df = pd.read_json("coursera-course-data-metadata.json")
    except Exception as e:
        st.error(f"🚨 Critical Error: Could not load the dataset. Ensure 'coursera-course-data-metadata.json' is in your directory. Error: {e}")
        st.stop()
        
    # -------------------------------------------------------------------------
    # AUTOMATIC KEY MAPPING
    # Open your JSON file and look at the keys. Change the right-hand strings 
    # (e.g., 'course_name') to match the exact keys used in your JSON file!
    # -------------------------------------------------------------------------
    column_mapping = {
        'course_id': 'id',               # Maps your unique course tracking key
        'course_name': 'title',           # Maps your main course title key
        'course_description': 'meta'     # Maps the descriptive text/syllabus key
    }
    
    # Safely apply the rename operation in local memory memory registers
    catalog_df = catalog_df.rename(columns=column_mapping)
    
    # Filter the dataframe down to keep only our three core pipeline targets
    catalog_df = catalog_df[['id', 'title', 'meta']]
    
    # Structural Text Normalization: Clean empty spaces to prevent vectorizer errors
    catalog_df['meta'] = catalog_df['meta'].fillna('')
    
    # NLP Feature Space Engine: Automatic tokenization and lowercase standardization
    vec = TfidfVectorizer(stop_words='english')
    matrix = vec.fit_transform(catalog_df['meta'])
    
    # Generate the comprehensive non-Euclidean angular similarity grid matrix
    sim_grid = cosine_similarity(matrix, matrix)
    return catalog_df, sim_grid

# Run pipeline execution loop and host vectors in active memory addresses
courses_df, proximity_matrix = load_and_compute_matrix()

# 3. Reactive UI Presentation Interface Layer
st.subheader("Select an Anchor Curriculum Module")
selected_title = st.selectbox("Choose a baseline course from the platform index:", courses_df['title'].tolist())

if selected_title:
    # Capture dropdown selection event and extract row index directly from memory cache
    idx = courses_df[courses_df['title'] == selected_title].index[0]
    
    # Benchmark computational processing calculations latency
    start_time = time.perf_counter()
    similarity_scores = list(enumerate(proximity_matrix[idx]))
    
    # Sort matching nodes in descending order, stripping out the self-matched core item
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:4]
    execution_latency = (time.perf_counter() - start_time) * 1000 # Convert to milliseconds

    st.markdown("---")
    st.subheader("Recommended Next-Sequence Educational Modules")
    
    # Render layout dashboard columns reactively based on spatial closeness mapping
    cols = st.columns(len(sorted_scores))
    for col, (recommend_idx, score) in zip(cols, sorted_scores):
        with col:
            st.info(f"**{courses_df.iloc[recommend_idx]['title']}**")
            st.metric(label="Thematic Alignment", value=f"{score*100:.1f}%")
            st.caption(f"Database Node ID: {courses_df.iloc[recommend_idx]['id']}")
            
    # System Status Performance Verification Tracker
    st.markdown("---")
    st.caption(f"⚡ In-Memory Vector Compute Latency: {execution_latency:.4f} ms | Catalog Density: {len(courses_df)} records | Status: Optimal")
