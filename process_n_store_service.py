"""
Utility script to process a PDF and store embeddings
Usage: python scripts/process_and_store.py <path_to_pdf>
"""

import sys
import os

from config import app_settings
from services.docment_processor_service import DocumentProcessorService
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

# app_settings = Settings()
def process_n_store(pdf_path: str):

    # Initialize services
    logger.info("Initializing services...")

    processor = DocumentProcessorService(
        chunk_size=app_settings.CHUNK_SIZE,
        chunk_overlap=app_settings.CHUNK_OVERLAP
    )

    embedding_service = EmbeddingService(
        endpoint=app_settings.AZURE_EMBEDDING_ENDPOINT,
        api_key=app_settings.AZURE_EMBEDDING_API_KEY,
        deployment_name=app_settings.AZURE_EMBEDDING_DEPLOYMENT_NAME,
        api_version=app_settings.AZURE_EMBEDDING_API_VERSION
    )

    # Get embedding dimension
    vector_size = embedding_service.get_embedding_dimension()
    logger.info(f"Embedding dimension: {vector_size}")

    vector_store = VectorStoreService(
        host=app_settings.QDRANT_HOST,
        port=app_settings.QDRANT_PORT,
        collection_name=app_settings.QDRANT_COLLECTION_NAME,
        vector_size=vector_size
    )

    # Create collection
    vector_store.create_collection(recreate=True)

    # Process document
    logger.info(f"Processing PDF: {pdf_path}")
    chunks = processor.process_document(pdf_path)

    # Generate embeddings
    logger.info("Generating embeddings...")
    chunks_with_embeddings = embedding_service.embed_chunks(chunks)

    # Store in vector database
    logger.info("Storing in Qdrant...")
    vector_store.upsert_chunks(chunks_with_embeddings)

    # Verify storage
    info = vector_store.get_collection_info()
    logger.info(f"Collection info: {info}")

    logger.info("✅ Document processing complete!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_n_store.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    process_n_store(pdf_path)