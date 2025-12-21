# Configuration settings for Financial Regulation Compliance Assistant

# Ollama Server Configuration
OLLAMA_HOST = "http://192.168.1.43:11434"
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# Vector Database Configuration
VECTOR_DB_PATH = "./vector_store"
COLLECTION_NAME = "financial_regulations"

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