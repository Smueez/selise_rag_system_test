from typing import Dict, Any, Optional
from loguru import logger

from .base_tools import BaseTool
from services.retriver_service import Retriever


class SemanticSearchTool(BaseTool):
    """Tool for semantic search in the document"""

    name = "semantic_search"
    description = (
        "Search for relevant information in the document using semantic similarity. "
        "Use this tool when you need to find information related to the user's question. "
        "Input should be a clear search query."
    )

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic search"""
        try:
            query = input_data.get("query")
            top_k = input_data.get("top_k", 5)

            if not query:
                return {
                    "success": False,
                    "error": "Query parameter is required"
                }

            logger.info(f"Executing semantic search: {query}")

            result = self.retriever.retrieve_with_context(query, top_k=top_k)

            return {
                "success": True,
                "data": result,
                "tool_name": self.name
            }

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.name
            }