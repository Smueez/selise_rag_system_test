from typing import Dict, Any, List
from loguru import logger

from .base_tools import BaseTool
from services.retriver_service import Retriever


class MultiQuerySearchTool(BaseTool):
    """Tool for multi-query search with query reformulation"""

    name = "multi_query_search"
    description = (
        "Search using multiple reformulated queries to get comprehensive results. "
        "Use this when a single search might miss relevant information or when "
        "the question can be interpreted in multiple ways."
    )

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of reformulated search queries"
                }
            },
            "required": ["queries"]
        }

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-query search"""
        try:
            queries = input_data.get("queries", [])

            if not queries:
                return {
                    "success": False,
                    "error": "Queries parameter is required"
                }

            logger.info(f"Executing multi-query search with {len(queries)} queries")

            results = self.retriever.multi_query_retrieve(queries)

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
                "success": True,
                "data": {
                    "context": context,
                    "sources": sources,
                    "num_results": len(results),
                    "queries_used": queries
                },
                "tool_name": self.name
            }

        except Exception as e:
            logger.error(f"Error in multi-query search: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.name
            }