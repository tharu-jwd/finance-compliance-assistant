# Document processing and vector database management

import os
import logging
import tempfile
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_RETRIEVED,
    EMBEDDING_MODEL, HUGGINGFACE_API_KEY,
    PINECONE_API_KEY, PINECONE_INDEX_NAME, DOCUMENT_PATHS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 768  # nomic-embed-text-v1


class NomicEmbeddings(HuggingFaceEndpointEmbeddings):
    """HuggingFaceEndpointEmbeddings with nomic-embed-text task prefixes."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"search_query: {text}")


class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with Pinecone and HuggingFace embeddings."""
        self.embeddings = NomicEmbeddings(
            model=EMBEDDING_MODEL,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY
        )

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Create index if it doesn't exist
        existing = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'…")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.pc_index = pc.Index(PINECONE_INDEX_NAME)

        self.vector_store = PineconeVectorStore(
            index=self.pc_index,
            embedding=self.embeddings
        )

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
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []

    def process_uploaded_file(self, uploaded_file) -> int:
        """Process a Streamlit UploadedFile and add chunks to Pinecone.

        Returns the number of chunks added.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()

            for doc in raw_docs:
                doc.metadata["source"] = uploaded_file.name

            chunks = self.split_documents(raw_docs)
            if chunks:
                self.vector_store.add_documents(chunks)

            os.unlink(tmp_path)
            logger.info(f"Processed '{uploaded_file.name}': {len(chunks)} chunks added")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return 0

    def search_similar_documents(self, query: str, n_results: int = MAX_CHUNKS_RETRIEVED) -> List[Dict[str, Any]]:
        """Search for similar documents in Pinecone."""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=n_results)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                    "rank": i + 1
                }
                for i, (doc, score) in enumerate(results)
            ]
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
            else:
                logger.warning(f"Directory not found: {path}")

        if not all_documents:
            logger.warning("No documents found to process")
            return

        chunks = self.split_documents(all_documents)
        if chunks:
            self.vector_store.add_documents(chunks)
            logger.info("Document processing completed successfully")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            stats = self.pc_index.describe_index_stats()
            return {
                "total_documents": stats.total_vector_count,
                "collection_name": PINECONE_INDEX_NAME,
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
