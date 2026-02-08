import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

load_dotenv(override=True)
class Settings(BaseSettings):
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION")

    # Azure Embedding
    AZURE_EMBEDDING_ENDPOINT: str = os.getenv("AZURE_EMBEDDING_ENDPOINT")
    AZURE_EMBEDDING_API_KEY: str = os.getenv("AZURE_EMBEDDING_API_KEY")
    AZURE_EMBEDDING_DEPLOYMENT_NAME: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
    AZURE_EMBEDDING_API_VERSION: str = os.getenv("AZURE_EMBEDDING_API_VERSION")

    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST")
    QDRANT_PORT: int = int(str(os.getenv("QDRANT_PORT")))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME")

    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD"))

    # Agent Configuration
    MAX_ITERATIONS: int = os.getenv("MAX_ITERATIONS")
    ENABLE_SELF_REFLECTION: bool = os.getenv("ENABLE_SELF_REFLECTION")

    # API Configuration
    API_HOST: str = os.getenv("API_HOST")
    API_PORT: int = os.getenv("API_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = True

app_settings = Settings()