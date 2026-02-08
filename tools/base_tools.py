from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel


class ToolInput(BaseModel):
    """Base class for tool inputs"""
    pass


class ToolOutput(BaseModel):
    """Base class for tool outputs"""
    success: bool
    data: Any
    error: Optional[str] = None


class BaseTool(ABC):
    """Base class for all tools"""

    name: str
    description: str

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Return the tool schema for function calling"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters()
            }
        }

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter schema"""
        pass