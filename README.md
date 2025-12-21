# Sri Lankan Finance Compliance Assistant

A RAG (Retrieval Augmented Generation) chatbot that helps users understand Sri Lankan financial regulations, banking circulars, and compliance requirements.

## Features

- **AI-Powered Q&A**: Chat interface powered by Llama 3.1 8B
- **Document Search**: Vector-based search through regulatory documents
- **Sri Lankan Focus**: Specialized for CBSL, Companies Act, and local standards
- **Source Citations**: Shows relevant document sources for each answer
- **Self-Hosted**: Runs on your own infrastructure for data privacy

## Architecture

- **Frontend**: Streamlit web interface
- **LLM**: Ollama with Llama 3.1 8B (self-hosted)
- **Embeddings**: nomic-embed-text (local)
- **Vector Database**: ChromaDB
- **Document Processing**: LangChain + PyPDF2

## Setup

### 1. Prerequisites
- Python 3.10+
- Ollama server running with:
  - `llama3.1:8b` model
  - `nomic-embed-text` embedding model

### 2. Installation
```bash
# Clone and setup
git clone https://github.com/tharu-jwd/finance-compliance-assistant
cd finance-compliance-assistant

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Edit `config.py` to set your Ollama server URL:
```python
OLLAMA_HOST = "http://your-server-ip:11434"
```

### 4. Add Documents
Place PDF documents in the appropriate folders:
```
documents/
├── cbsl/              # CBSL circulars and banking regulations
├── companies_act/     # Companies Act documents  
└── accounting/        # Accounting standards
```

### 5. Process Documents
```bash
python document_processor.py
```

### 6. Run Application
```bash
streamlit run app.py
```

## Usage

1. **Start the application**: Access via web browser (usually http://localhost:8501)
2. **Ask questions**: Type questions about Sri Lankan financial regulations
3. **View sources**: Expand source citations to see relevant document excerpts
4. **Example queries**:
   - "What are the KYC requirements for digital wallets?"
   - "How should we account for cryptocurrency holdings?"
   - "What are the capital adequacy requirements for banks?"

## Document Sources

The system processes documents from:
- **CBSL Circulars**: Banking regulations and payment system rules
- **Companies Act**: Corporate governance and compliance requirements  
- **Accounting Standards**: Sri Lankan accounting and reporting standards

## Technical Details

- **Chunking**: Documents split into 1000-character chunks with 200-character overlap
- **Retrieval**: Top 5 most similar chunks retrieved for each query
- **Generation**: Low temperature (0.1) for accurate regulatory responses
- **Privacy**: All processing done locally, no external API calls

## Limitations

**Important**: This system is for informational purposes only. Always consult official regulatory sources and legal professionals for compliance decisions.

---

**Powered by**: Llama 3.1 8B • ChromaDB • LangChain • Streamlit
