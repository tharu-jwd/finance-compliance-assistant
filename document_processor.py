# Document processing and vector database management

import os
import logging
import tempfile
from typing import List, Dict, Any
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from config import (
    VECTOR_DB_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_CHUNKS_RETRIEVED, EMBEDDING_MODEL, HUGGINGFACE_API_KEY, DOCUMENT_PATHS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with ChromaDB and HuggingFace embeddings."""
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY
        )

        # Initialize ChromaDB with persistent client
        try:
            os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
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

    def get_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for text chunks using HuggingFace."""
        try:
            if is_query:
                # nomic-embed-text requires task prefix for queries
                prefixed = [f"search_query: {t}" for t in texts]
                return self.embeddings.embed_documents(prefixed)
            else:
                # nomic-embed-text requires task prefix for indexing
                prefixed = [f"search_document: {t}" for t in texts]
                return self.embeddings.embed_documents(prefixed)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [[0.0] * 768 for _ in texts]

    def add_documents_to_vectordb(self, documents: List[Document]):
        """Add document chunks to ChromaDB."""
        if not documents:
            logger.warning("No documents to add to vector database")
            return

        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Use unique IDs based on existing count to avoid collisions
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

            logger.info("Generating embeddings...")
            embeddings = self.get_embeddings(texts, is_query=False)

            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(documents)} document chunks to vector database")

        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")

    def process_uploaded_file(self, uploaded_file) -> int:
        """Process a Streamlit UploadedFile object and add to ChromaDB.

        Returns the number of chunks added.
        """
        try:
            # Write uploaded file to a temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()

            # Attach the original filename to metadata
            for doc in raw_docs:
                doc.metadata["source"] = uploaded_file.name

            chunks = self.split_documents(raw_docs)
            if chunks:
                self.add_documents_to_vectordb(chunks)

            os.unlink(tmp_path)
            logger.info(f"Processed uploaded file '{uploaded_file.name}': {len(chunks)} chunks added")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return 0

    def search_similar_documents(self, query: str, n_results: int = MAX_CHUNKS_RETRIEVED) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector database."""
        try:
            query_embedding = self.get_embeddings([query], is_query=True)[0]

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,
                        "rank": i + 1
                    })

            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def process_all_documents(self):
        """Process all documents from configured directories."""
        all_documents = []

        for path in DOCUMENT_PATHS.values():
            os.makedirs(path, exist_ok=True)

        for doc_type, path in DOCUMENT_PATHS.items():
            if os.path.exists(path):
                documents = self.load_documents(path)
                if documents:
                    for doc in documents:
                        doc.metadata["document_type"] = doc_type
                    all_documents.extend(documents)
                    logger.info(f"Processed {len(documents)} {doc_type} documents")
            else:
                logger.warning(f"Directory not found: {path}")

        if not all_documents:
            logger.warning("No documents found to process")
            return

        chunks = self.split_documents(all_documents)
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
                "total_documents": count,
                "collection_name": COLLECTION_NAME,
                "vector_db_path": VECTOR_DB_PATH
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_all_documents()

    stats = processor.get_collection_stats()
    print(f"Collection stats: {stats}")

    query = "What are the KYC requirements?"
    results = processor.search_similar_documents(query, n_results=3)
    print(f"Search results for '{query}':")
    for result in results:
        print(f"- Score: {result['similarity_score']:.3f}")
        print(f"- Content: {result['content'][:100]}...")
        print()
