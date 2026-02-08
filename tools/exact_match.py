from typing import Dict, Any, List
from loguru import logger
import re

from .base_tools import BaseTool
from services.vector_store_service import VectorStoreService


class ExactMatchTool(BaseTool):
    """Tool for exact keyword/phrase matching"""

    name = "exact_match_search"
    description = (
        "Search for exact keywords or phrases in the document. "
        "Use this when you need to find specific terms, names, dates, or precise quotes. "
        "Case-insensitive search."
    )

    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "The exact keyword or phrase to search for"
                }
            },
            "required": ["keyword"]
        }

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute exact match search"""
        try:
            keyword = input_data.get("keyword", "").lower()

            if not keyword:
                return {
                    "success": False,
                    "error": "Keyword parameter is required"
                }

            logger.info(f"Executing exact match search: {keyword}")

            # Get all documents from collection (simplified approach)
            # In production, you might want to implement a full-text search index
            collection_info = self.vector_store.get_collection_info()

            # For now, we'll scroll through points and filter
            # This is a simplified implementation
            matches = []

            # Note: This is a basic implementation
            # For production, consider using a proper full-text search engine
            # or maintaining a separate keyword index

            scroll_result = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            for point in scroll_result[0]:
                text = point.payload.get('text', '').lower()
                if keyword in text:
                    # Find context around the keyword
                    index = text.find(keyword)
                    start = max(0, index - 100)
                    end = min(len(text), index + len(keyword) + 100)
                    context = text[start:end]

                    matches.append({
                        'chunk_id': point.payload.get('chunk_id'),
                        'context': context,
                        'full_text': point.payload.get('text'),
                        'metadata': point.payload.get('metadata')
                    })

            logger.info(f"Found {len(matches)} exact matches")

            return {
                "success": True,
                "data": {
                    "matches": matches,
                    "num_matches": len(matches),
                    "keyword": keyword
                },
                "tool_name": self.name
            }

        except Exception as e:
            logger.error(f"Error in exact match search: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.name
            }