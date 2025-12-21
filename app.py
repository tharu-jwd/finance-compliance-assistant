# Minimal working RAG app with basic vector search

import streamlit as st
import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import *

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'documents' not in st.session_state:
    st.session_state.documents = []

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []

class MinimalRAG:
    def __init__(self):
        """Initialize minimal RAG system."""
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)
        
        # Add some sample regulatory documents
        self.sample_docs = [
            "KYC requirements in Sri Lanka include identity verification, address proof, and source of funds documentation for all financial transactions above LKR 100,000.",
            "Digital payment providers must obtain license from CBSL and comply with anti-money laundering regulations including transaction monitoring and suspicious activity reporting.",
            "Capital adequacy ratio for banks in Sri Lanka must be maintained at minimum 12.5% as per CBSL guidelines, with Tier 1 capital ratio of at least 8.5%.",
            "Cross-border payments require proper documentation including invoice, import/export permits, and foreign exchange declaration forms as per CBSL regulations.",
            "Cryptocurrency trading and exchanges are not legally recognized in Sri Lanka and CBSL has issued warnings against their use for payments."
        ]
        
        # Generate embeddings for sample documents
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Generate embeddings for sample documents."""
        st.session_state.documents = self.sample_docs
        st.session_state.embeddings = []
        
        for doc in self.sample_docs:
            try:
                response = self.ollama_client.embeddings(
                    model=EMBEDDING_MODEL,
                    prompt=doc
                )
                st.session_state.embeddings.append(response['embedding'])
            except Exception as e:
                st.error(f"Error generating embeddings: {e}")
                return False
        return True
    
    def search_documents(self, query: str, top_k: int = 3):
        """Search for relevant documents using cosine similarity."""
        try:
            # Generate query embedding
            query_response = self.ollama_client.embeddings(
                model=EMBEDDING_MODEL,
                prompt=query
            )
            query_embedding = query_response['embedding']
            
            # Calculate similarities
            if not st.session_state.embeddings:
                return []
            
            similarities = cosine_similarity(
                [query_embedding], 
                st.session_state.embeddings
            )[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i in top_indices:
                results.append({
                    'content': st.session_state.documents[i],
                    'score': float(similarities[i])
                })
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def generate_response(self, query: str, context_docs: list) -> str:
        """Generate response using retrieved context."""
        try:
            # Prepare context
            context = ""
            if context_docs:
                context = "\n\n".join([doc['content'] for doc in context_docs[:2]])
            
            # Create prompt
            prompt = f"""You are an expert on Sri Lankan financial regulations. Answer the question using the provided context.

CONTEXT:
{context}

QUESTION: {query}

ANSWER (be brief and accurate):"""
            
            # Generate response
            response = self.ollama_client.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 150
                }
            )
            
            return response['response']
            
        except Exception as e:
            return f"Error: {e}"

def main():
    st.set_page_config(
        page_title="Financial RAG Assistant",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("Financial Regulation Assistant")
    st.markdown("With Retrieval Augmented Generation (RAG) powered by Ollama.")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            try:
                st.session_state.rag_system = MinimalRAG()
                st.success("Initialized with sample documents")
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                st.session_state.rag_system = None
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        if st.session_state.rag_system:
            st.metric("Documents", len(st.session_state.documents))
            st.metric("Embeddings", len(st.session_state.embeddings))
            
            st.subheader("Sample Documents")
            for i, doc in enumerate(st.session_state.documents[:3]):
                st.write(f"**Doc {i+1}:** {doc[:100]}...")
    
    # Chat interface
    st.subheader("Chat")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Retrieved Documents"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}** (Score: {source['score']:.3f})")
                        st.write(source['content'])
                        st.write("---")
    
    # Chat input
    if query := st.chat_input("Ask about financial regulations..."):
        if not st.session_state.rag_system:
            st.error("RAG system not initialized")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Search and generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                # Search relevant documents
                relevant_docs = st.session_state.rag_system.search_documents(query)
                
                # Generate response
                response = st.session_state.rag_system.generate_response(query, relevant_docs)
                
                st.markdown(response)
                
                # Show sources
                if relevant_docs:
                    with st.expander("Retrieved Documents"):
                        for i, source in enumerate(relevant_docs):
                            st.write(f"**Source {i+1}** (Score: {source['score']:.3f})")
                            st.write(source['content'])
                            st.write("---")
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": relevant_docs
        })
    
    # Clear chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()