# Configuration settings for Financial Regulation Compliance Assistant

import os
import streamlit as st


def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key)


# API Keys
GROQ_API_KEY = get_secret("GROQ_API_KEY")
HUGGINGFACE_API_KEY = get_secret("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

# Model Configuration
GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"

# Vector Database Configuration
PINECONE_INDEX_NAME = "financial-regulations"

# Document Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_RETRIEVED = 5

# Streamlit Configuration
APP_TITLE = "Sri Lankan Financial Regulation Assistant"
APP_DESCRIPTION = "Chat with Sri Lankan financial regulations and compliance documents"

# Document Sources
DOCUMENT_PATHS = {
    "cbsl_circulars": "./documents/cbsl/",
    "companies_act": "./documents/companies_act/",
    "accounting_standards": "./documents/accounting/"
}

# RAG Configuration
TEMPERATURE = 0.1  # Low temperature for accurate regulatory responses
MAX_TOKENS = 1000
