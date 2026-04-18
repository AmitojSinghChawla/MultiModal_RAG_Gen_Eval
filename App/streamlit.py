import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder


@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return embedder, reranker