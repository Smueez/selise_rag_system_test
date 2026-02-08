from typing import List, Dict, Any, Optional
from loguru import logger

from services.embedding_service import EmbeddingService
# from app.services.vector_store import VectorStore
# from app.services.embedding_service import EmbeddingService
# from app.config import get_settings

from services.vector_store_service import VectorStoreService


class Retriever:
    """Handles document retrieval with various strategies"""

    def __init__(
            self,
            vector_store: VectorStoreService,
            embedding_service: EmbeddingService,
            top_k: int = 5,
            similarity_threshold: float = 0.7
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized Retriever with top_k={top_k}, threshold={similarity_threshold}")

    def retrieve(
            self,
            query: str,
            top_k: Optional[int] = None,
            threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        """
        try:
            # Use default values if not provided
            k = top_k or self.top_k
            thresh = threshold or self.similarity_threshold

            # Generate query embedding
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            query_embedding = self.embedding_service.generate_embedding(query)

            # Search vector store
            results = self.vector_store.search(
                query_vector=query_embedding,
                limit=k,
                score_threshold=thresh
            )

            logger.info(f"Retrieved {len(results)} documents with scores above {thresh}")

            return results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    def retrieve_with_context(
            self,
            query: str,
            top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents and format with context
        Returns structured context for the agent
        """
        results = self.retrieve(query, top_k=top_k)

        # Format context
        context_parts = []
        sources = []

        for idx, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {idx}] (Relevance: {result['score']:.3f})\n{result['text']}"
            )
            sources.append({
                'chunk_id': result['chunk_id'],
                'score': result['score'],
                'metadata': result['metadata']
            })

        context = "\n\n".join(context_parts)

        return {
            'context': context,
            'sources': sources,
            'num_results': len(results)
        }

    def multi_query_retrieve(self, queries: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve documents for multiple query variations
        Deduplicates and ranks by best score
        """
        all_results = {}

        for query in queries:
            results = self.retrieve(query, top_k=top_k)

            for result in results:
                chunk_id = result['chunk_id']

                # Keep the result with highest score
                if chunk_id not in all_results or result['score'] > all_results[chunk_id]['score']:
                    all_results[chunk_id] = result

        # Sort by score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        logger.info(f"Multi-query retrieval: {len(queries)} queries -> {len(sorted_results)} unique results")

        return sorted_results[:self.top_k]