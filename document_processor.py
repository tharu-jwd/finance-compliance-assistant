# Document processing and vector database management

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
import ollama

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with ChromaDB and Ollama client."""
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)
        
        # Initialize ChromaDB with in-memory client first (safer)
        try:
            self.chroma_client = chromadb.Client()  # In-memory client
            
            # Create collection
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Sri Lankan Financial Regulations"}
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_documents(self, directory_path: str) -> List[Document]:
        """Load all PDF documents from a directory."""
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents from {directory_path}: {e}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks using Ollama."""
        embeddings = []
        for text in texts:
            try:
                response = self.ollama_client.embeddings(
                    model=EMBEDDING_MODEL,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 768)  # Fallback embedding
        return embeddings
    
    def add_documents_to_vectordb(self, documents: List[Document]):
        """Add document chunks to ChromaDB."""
        if not documents:
            logger.warning("No documents to add to vector database")
            return
        
        try:
            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.get_embeddings(texts)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} document chunks to vector database")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
    
    def search_similar_documents(self, query: str, n_results: int = MAX_CHUNKS_RETRIEVED) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector database."""
        try:
            # Generate query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def process_all_documents(self):
        """Process all documents from configured directories."""
        all_documents = []
        
        # Create document directories if they don't exist
        for path in DOCUMENT_PATHS.values():
            os.makedirs(path, exist_ok=True)
        
        # Load documents from all configured directories
        for doc_type, path in DOCUMENT_PATHS.items():
            if os.path.exists(path):
                documents = self.load_documents(path)
                if documents:
                    # Add document type to metadata
                    for doc in documents:
                        doc.metadata['document_type'] = doc_type
                    all_documents.extend(documents)
                    logger.info(f"Processed {len(documents)} {doc_type} documents")
            else:
                logger.warning(f"Directory not found: {path}")
        
        if not all_documents:
            logger.warning("No documents found to process")
            return
        
        # Split documents into chunks
        chunks = self.split_documents(all_documents)
        
        # Add to vector database
        if chunks:
            self.add_documents_to_vectordb(chunks)
            logger.info("Document processing completed successfully")
        else:
            logger.error("No document chunks created")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': COLLECTION_NAME,
                'vector_db_path': VECTOR_DB_PATH
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    
    # Process all documents
    processor.process_all_documents()
    
    # Show stats
    stats = processor.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Test search
    query = "What are the KYC requirements?"
    results = processor.search_similar_documents(query, n_results=3)
    print(f"Search results for '{query}':")
    for result in results:
        print(f"- Score: {result['similarity_score']:.3f}")
        print(f"- Content: {result['content'][:100]}...")
        print()