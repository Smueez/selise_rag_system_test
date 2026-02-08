from fastapi import APIRouter, HTTPException
from fastapi.responses import  StreamingResponse

# from pathlib import Path
from loguru import logger
from typing import Optional, List
from pydantic import BaseModel

from services.vector_store_service import VectorStoreService
from services.embedding_service import EmbeddingService
from services.retriver_service import Retriever
from services.agent import ReflectiveAgent
from tools.semantic_search import SemanticSearchTool
from tools.multi_query_serch import MultiQuerySearchTool
from tools.exact_match import ExactMatchTool
from tools.validator import AnswerValidatorTool
from config import Settings

router = APIRouter()
settings = Settings()


class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[dict]] = None


class QueryResponse(BaseModel):
    answer: str
    context: str
    iterations: int
    success: bool


# Global agent instance (initialized lazily)
_agent_instance = None


def get_agent() -> ReflectiveAgent:
    """Get or create agent instance"""
    global _agent_instance

    if _agent_instance is None:
        logger.info("Initializing agent...")

        # Initialize services
        embedding_service = EmbeddingService(
            endpoint=settings.AZURE_EMBEDDING_ENDPOINT,
            api_key=settings.AZURE_EMBEDDING_API_KEY,
            deployment_name=settings.AZURE_EMBEDDING_DEPLOYMENT_NAME,
            api_version=settings.AZURE_EMBEDDING_API_VERSION
        )

        vector_size = embedding_service.get_embedding_dimension()

        vector_store = VectorStoreService(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vector_size=vector_size
        )

        retriever = Retriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
            top_k=settings.TOP_K_RESULTS,
            similarity_threshold=settings.SIMILARITY_THRESHOLD
        )

        # Initialize tools
        semantic_search = SemanticSearchTool(retriever)
        multi_query = MultiQuerySearchTool(retriever)
        exact_match = ExactMatchTool(vector_store)
        validator = AnswerValidatorTool()

        # Create agent
        _agent_instance = ReflectiveAgent(
            semantic_search_tool=semantic_search,
            multi_query_tool=multi_query,
            exact_match_tool=exact_match,
            validator_tool=validator,
            max_iterations=settings.MAX_ITERATIONS
        )

        logger.info("Agent initialized successfully")

    return _agent_instance


# ... (keep existing upload, job status, health endpoints)

@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Query the agent with streaming response
    Returns Server-Sent Events (SSE)
    """
    try:
        agent = get_agent()

        async def event_generator():
            async for chunk in agent.process_query_stream(
                    query=request.query,
                    conversation_history=request.conversation_history
            ):
                yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Error in query stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the agent without streaming (for testing)
    """
    try:
        agent = get_agent()

        result = agent.process_query(
            query=request.query,
            conversation_history=request.conversation_history
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))