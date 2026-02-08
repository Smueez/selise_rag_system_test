from typing import Dict, Any
from loguru import logger

from .base_tools import BaseTool


class AnswerValidatorTool(BaseTool):

    name = "validate_answer"
    description = (
        "Validate if a proposed answer is properly grounded in the retrieved context. "
        "Use this to check for hallucinations and ensure factual accuracy."
    )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The proposed answer to validate"
                },
                "context": {
                    "type": "string",
                    "description": "The retrieved context to validate against"
                }
            },
            "required": ["answer", "context"]
        }

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate answer against context"""
        try:
            answer = input_data.get("answer", "")
            context = input_data.get("context", "")

            if not answer or not context:
                return {
                    "success": False,
                    "error": "Both answer and context are required"
                }

            logger.info("Validating answer against context")

            # Simple validation heuristics
            answer_lower = answer.lower()
            context_lower = context.lower()

            # Check 1: Are key terms from answer in context?
            answer_words = set(answer_lower.split())
            context_words = set(context_lower.split())

            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was',
                          'were', 'be', 'been', 'being'}
            answer_keywords = answer_words - stop_words

            overlap = answer_keywords & context_words
            coverage = len(overlap) / len(answer_keywords) if answer_keywords else 0

            # Check 2: Length check (very long answers might indicate hallucination)
            is_reasonable_length = len(answer) < len(context) * 1.5

            # Check 3: Does answer contain specific claims not in context?
            # This is simplified - in production, use more sophisticated NLI models

            is_valid = coverage > 0.5 and is_reasonable_length

            validation_result = {
                "is_valid": is_valid,
                "keyword_coverage": coverage,
                "reasonable_length": is_reasonable_length,
                "confidence": coverage if is_valid else 0.3
            }

            logger.info(f"Validation result: {validation_result}")

            return {
                "success": True,
                "data": validation_result,
                "tool_name": self.name
            }

        except Exception as e:
            logger.error(f"Error validating answer: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.name
            }