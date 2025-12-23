# Financial Regulation Compliance Assistant

## Overview
RAG-based system that provides accurate answers to financial regulation questions, accounting standards, and compliance requirements for Sri Lankan and international financial regulations.

**Project Type**: Portfolio/CV project demonstrating AI/ML skills for internship applications

## Problem Statement
Financial professionals need quick, accurate access to complex regulatory information across multiple jurisdictions and standards. Manual document search is time-consuming and error-prone.

## CV/Portfolio Value
- Demonstrates RAG (Retrieval-Augmented Generation) implementation
- Shows real-world AI application in finance domain
- Exhibits document processing and NLP skills
- Highlights ability to work with regulatory/compliance data
- Showcases end-to-end ML project development

## Core Features

### Document Coverage
- IFRS standards (International Financial Reporting Standards)
- Sri Lankan Accounting Standards
- CBSL (Central Bank of Sri Lanka) regulations
- Companies Act requirements
- Tax regulations (Inland Revenue)
- SEC (Securities and Exchange Commission) rules
- Basel III requirements (for banking)

### Example Use Cases
- "How should we recognize revenue for a SaaS subscription business?"
- "What are the KYC requirements for opening a digital wallet?"
- "How do we account for cryptocurrency holdings under IFRS?"
- "What are the capital adequacy requirements for payment institutions?"
- "What documentation is needed for cross-border payments?"

## Tech Stack

### Core Components
- **RAG Framework**: LangChain / LlamaIndex
- **Vector Database**: ChromaDB / Pinecone
- **LLM**: OpenAI GPT-4 / Gemini Pro (for regulatory accuracy)
- **UI**: Streamlit
- **Document Processing**: PyPDF
- **Language**: Python

## Advanced Features

### Phase 2 Features
- **Regulation Change Tracking**: "What changed in the latest IFRS update?"
- **Compliance Checklist Generator**: Input new feature → Output compliance requirements
- **Multi-jurisdiction Support**: Compare Sri Lankan vs International standards
- **Citation Verification**: Direct links to source documents
- **Confidence Scoring**: "High confidence" vs "Consult legal team"

## Data Sources

### Publicly Available Documents
- IFRS Foundation website (free PDFs)
- Sri Lanka Accounting Standards (SLAASMB website)
- CBSL circulars (Central Bank website)
- Companies Act (government publication)
- Inland Revenue regulations
- SEC rules and guidelines

## Project Scope - FINALIZED

### Target
- **Purpose**: Portfolio project for AI/ML, Data Science, Software Engineering internships
- **Geographic Focus**: Sri Lanka only (CBSL, local accounting standards, Companies Act)
- **Format**: RAG chatbot with web interface
- **Approach**: Use existing libraries, focus on implementation over building from scratch

### MVP Features
- [ ] Chat interface for asking regulation questions
- [ ] Document ingestion pipeline (PDF processing)
- [ ] Vector search and retrieval
- [ ] LLM-powered response generation
- [ ] Source citation in responses
- [ ] Basic confidence scoring

### Technical Decisions
- **UI**: Streamlit (simple, good for demos)
- **Documents**: CBSL circulars, Sri Lankan Accounting Standards, Companies Act
- **Deployment**: Streamlit Cloud (free, live demo link for CV)

## Technical Architecture

### RAG Pipeline
```
1. Document Processing
   PDFs → Text chunks → Embeddings → Vector DB

2. Query Processing
   User question → Embedding → Vector search → Retrieved chunks

3. Response Generation
   Retrieved chunks + User question → LLM → Answer + Citations
```

### What LLMs Do in Our System

**LLMs have 2 roles:**

1. **Text Understanding (Embeddings)**
   - Convert documents and questions into numerical vectors
   - Used for: Finding relevant document chunks via similarity search
   - Model: OpenAI text-embedding-ada-002

2. **Response Generation (Main LLM)**
   - Takes retrieved document chunks + user question
   - Synthesizes information into coherent answer
   - Adds context, explains regulations in plain language
   - Model: OpenAI GPT-3.5-turbo

**Detailed Example:**

**User Question:** "What are KYC requirements for digital wallets?"

**Step 1 - Vector Search finds these document chunks:**
```
Chunk 1 (CBSL Circular 03/2021):
"Licensed Payment Service Providers shall implement robust Customer Due Diligence (CDD) procedures including: (a) verification of customer identity using reliable documents; (b) understanding the purpose and intended nature of the business relationship..."

Chunk 2 (CBSL Circular 03/2021):  
"For electronic wallets with transaction limits exceeding LKR 100,000 per month, enhanced due diligence shall include: verification of source of funds, ongoing monitoring of transactions, and maintenance of records for minimum 5 years..."
```

**Step 2 - LLM Response Generation:**

**Without LLM (just raw chunks):**
User gets: *[shows above raw regulatory text chunks]*

**With LLM (GPT-3.5 synthesis):**
"For digital wallets in Sri Lanka, KYC requirements depend on transaction limits:

**Basic Requirements (all wallets):**
- Verify customer identity with reliable documents
- Understand the purpose of the business relationship
- Implement Customer Due Diligence (CDD) procedures

**Enhanced Requirements (wallets >LKR 100,000/month):**
- Verify source of funds
- Ongoing transaction monitoring  
- Keep records for minimum 5 years

Source: CBSL Circular 03/2021"
```

**The LLM transforms legal jargon into structured, actionable information.**

### Tech Stack Options

**Option 1 - OpenAI (Simple, costs ~$5-10):**
- **LLM**: OpenAI GPT-3.5-turbo
- **Embeddings**: OpenAI text-embedding-ada-002

**Option 2 - Self-hosted (CHOSEN - Free, 16GB RAM sufficient):**

**Best Available Models for RAG (Dec 2025):**
1. **llama3.1:8b** (~5GB) - Best practical choice for RAG & document analysis
2. **qwen3:8b** (~5GB) - Alternative strong reasoning model
3. **mistral:7b** (~4GB) - Lightweight backup option

**Embeddings**: 
- **nomic-embed-text** (~500MB) - High-quality open embedding model for RAG

**Shared Components:**
- **Frontend**: Streamlit
- **RAG Framework**: LangChain
- **Vector DB**: ChromaDB (local)
- **PDF Processing**: PyPDF2 + LangChain document loaders
- **Deployment**: Streamlit Cloud

### Development Phases
1. **Phase 1**: Basic document ingestion and vector storage
2. **Phase 2**: Simple Q&A functionality
3. **Phase 3**: Chat interface with Streamlit
4. **Phase 4**: Add citations and confidence scoring
5. **Phase 5**: Deploy and polish for CV

## What You Need to Provide

### Required from You:
1. **Access to your Ubuntu server** (SSH credentials)
2. **Install Ollama on the server** (I'll guide you through this)

### I'll Handle:
- All code development
- Document collection (I'll find and download Sri Lankan regulations)
- Environment setup
- Deployment guidance

### Documents I'll Collect:
- **CBSL circulars**: From cbsl.gov.lk (banking regulations, payment systems)
- **Companies Act**: From parliament.lk or legal databases
- **Accounting Standards**: From ca-si.lk or slaasmb.com
- **Selected tax regulations**: Basic corporate tax rules

## Next Steps
- [ ] You: Get OpenAI API key
- [ ] Me: Set up development environment
- [ ] Me: Collect and process Sri Lankan regulation documents
- [ ] Me: Build document ingestion pipeline
- [ ] Me: Implement basic RAG functionality
- [ ] Me: Create Streamlit chat interface
- [ ] Me: Deploy and document for portfolio

---
*Document created: 2025-12-21*
*Last updated: 2025-12-21*