from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import AzureOpenAI
import json
from loguru import logger

from config import Settings
from tools.semantic_search import SemanticSearchTool
from tools.multi_query_serch import MultiQuerySearchTool
from tools.exact_match import ExactMatchTool
from tools.validator import AnswerValidatorTool


class ReflectiveAgent:
    """
    Agent with self-reflection capabilities
    Orchestrates tools and validates answers
    """

    def __init__(
        self,
        semantic_search_tool: SemanticSearchTool,
        multi_query_tool: MultiQuerySearchTool,
        exact_match_tool: ExactMatchTool,
        validator_tool: AnswerValidatorTool,
        max_iterations: int = 3
    ):
        self.settings = Settings()

        # Initialize Azure OpenAI client for chat
        self.client = AzureOpenAI(
            azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT,
            api_key=self.settings.AZURE_OPENAI_API_KEY,
            api_version=self.settings.AZURE_OPENAI_API_VERSION
        )

        # Tools
        self.tools = {
            "semantic_search": semantic_search_tool,
            "multi_query_search": multi_query_tool,
            "exact_match_search": exact_match_tool,
            "validate_answer": validator_tool
        }

        self.max_iterations = max_iterations

        logger.info("Initialized ReflectiveAgent with self-reflection")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:

        schemas = []
        # Only include search tools in the initial phase
        for tool_name in ["semantic_search", "multi_query_search", "exact_match_search"]:
            tool = self.tools[tool_name]
            schemas.append(tool.get_schema())
        return schemas

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:

        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }

        tool = self.tools[tool_name]
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        result = tool.execute(arguments)
        return result

    def create_system_prompt(self) -> str:
        """Create the system prompt for the agent"""
        return """You are a helpful AI assistant with access to a document knowledge base.

Your goal is to answer user questions accurately based ONLY on the information in the retrieved documents.

IMPORTANT RULES:
1. ALWAYS start by using semantic_search to retrieve relevant information
2. Base your answers ONLY on the retrieved context
3. If information is not in the documents, clearly state "I don't have information about that in the provided documents"
4. Cite document references when possible (e.g., "According to Document 1...")
5. Be concise but comprehensive
6. After retrieving information, provide your answer WITHOUT calling more tools

WORKFLOW:
1. Use semantic_search to find relevant information
2. Review the retrieved context
3. Formulate a clear answer based on the context
4. DO NOT keep searching - answer based on what you found

Remember: It's better to say you don't know than to make up information not in the documents."""

    def reflect_on_results(
        self,
        query: str,
        retrieved_context: str,
        current_answer: str,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Self-reflection mechanism
        Evaluates if the current answer is satisfactory or needs improvement
        """
        reflection_prompt = f"""You are evaluating the quality of a RAG system's answer.

USER QUESTION: {query}

RETRIEVED CONTEXT (first 2000 chars):
{retrieved_context[:2000]}

CURRENT ANSWER:
{current_answer}

REFLECTION TASK:
Evaluate if the answer is:
1. Grounded in the retrieved context (no hallucinations)
2. Comprehensive enough to answer the question
3. Accurate and well-supported

Respond in JSON format:
{{
    "is_satisfactory": true/false,
    "issues": ["list of issues if any"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Be lenient - if the answer is reasonable and grounded, mark as satisfactory."""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating answer quality. Be fair and lenient."},
                    {"role": "user", "content": reflection_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            reflection_text = response.choices[0].message.content

            # Try to parse JSON
            try:
                # Extract JSON from markdown code blocks if present
                if "```json" in reflection_text:
                    reflection_text = reflection_text.split("```json")[1].split("```")[0].strip()
                elif "```" in reflection_text:
                    reflection_text = reflection_text.split("```")[1].split("```")[0].strip()

                reflection = json.loads(reflection_text)
            except json.JSONDecodeError:
                logger.warning("Could not parse reflection as JSON, marking as satisfactory")
                reflection = {
                    "is_satisfactory": True,
                    "issues": [],
                    "confidence": 0.75,
                    "reasoning": "Default assessment"
                }

            logger.info(f"Reflection (iteration {iteration}): Satisfactory={reflection.get('is_satisfactory')}, Confidence={reflection.get('confidence')}")

            return reflection

        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return {
                "is_satisfactory": True,
                "issues": [],
                "confidence": 0.7,
                "reasoning": "Error in reflection, accepting answer"
            }

    async def process_query_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:

        if conversation_history is None:
            conversation_history = []

        # Build messages
        messages = [
            {"role": "system", "content": self.create_system_prompt()}
        ]

        # Add conversation history
        messages.extend(conversation_history)

        # Add current query
        messages.append({"role": "user", "content": query})

        # Get tool schemas
        tools = self.get_tool_schemas()

        iteration = 0
        retrieved_context = ""
        final_answer = ""
        tool_call_count = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{self.max_iterations}")

            try:
                # After first iteration with tools, force answer generation
                if tool_call_count > 0:
                    # Don't provide tools - force the model to answer
                    response = self.client.chat.completions.create(
                        model=self.settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                else:
                    # First iteration - allow tool use
                    response = self.client.chat.completions.create(
                        model=self.settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=0.7,
                        max_tokens=2000
                    )

                assistant_message = response.choices[0].message

                # Check if tool calls are needed
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    tool_call_count += len(assistant_message.tool_calls)

                    # Add assistant message to history
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in assistant_message.tool_calls
                        ]
                    })

                    # Execute tools
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name

                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        # Execute tool
                        tool_result = self.execute_tool(tool_name, arguments)

                        # Store context if it's a search tool
                        if tool_name in ["semantic_search", "multi_query_search"] and tool_result.get("success"):
                            context_data = tool_result.get("data", {})
                            if isinstance(context_data, dict):
                                new_context = context_data.get("context", "")
                                if new_context:
                                    retrieved_context += "\n\n" + new_context

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result)
                        })

                        # Yield progress update
                        yield f"data: {json.dumps({'type': 'tool_execution', 'tool': tool_name, 'status': 'completed'})}\n\n"

                    # Continue to next iteration to get response after tool execution
                    continue

                else:
                    # No tool calls - we have a final answer
                    final_answer = assistant_message.content or ""

                    # Self-reflection (if enabled and we have context)
                    if self.settings.ENABLE_SELF_REFLECTION and retrieved_context and iteration < self.max_iterations:
                        yield f"data: {json.dumps({'type': 'reflection', 'status': 'started'})}\n\n"

                        reflection = self.reflect_on_results(
                            query=query,
                            retrieved_context=retrieved_context,
                            current_answer=final_answer,
                            iteration=iteration
                        )

                        yield f"data: {json.dumps({'type': 'reflection', 'result': reflection})}\n\n"

                        # If not satisfactory, try one more time
                        if not reflection.get("is_satisfactory") and iteration < self.max_iterations:
                            logger.info("Answer not satisfactory according to reflection, refining...")

                            # Add refinement instruction
                            refinement_msg = f"Please refine your answer. Issues: {', '.join(reflection.get('issues', ['general quality']))}. Make sure to base your answer strictly on the retrieved context."
                            messages.append({"role": "user", "content": refinement_msg})
                            continue

                    # Stream final answer
                    yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"

                    # Stream answer in chunks
                    words = final_answer.split()
                    for i in range(0, len(words), 3):
                        chunk = " ".join(words[i:i+3]) + " "
                        yield f"data: {json.dumps({'type': 'answer_chunk', 'content': chunk})}\n\n"

                    yield f"data: {json.dumps({'type': 'answer_end'})}\n\n"

                    # Send metadata
                    metadata = {
                        'type': 'metadata',
                        'iterations': iteration,
                        'tool_calls': tool_call_count,
                        'has_context': bool(retrieved_context)
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"

                    # Send completion signal
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break

            except Exception as e:
                logger.error(f"Error in agent iteration {iteration}: {e}")
                error_message = f"I encountered an error while processing your query: {str(e)}"
                yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break

        # If max iterations reached without answer
        if iteration >= self.max_iterations and not final_answer:
            fallback = "I apologize, but I had trouble formulating a clear answer. Please try rephrasing your question."
            yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"
            yield f"data: {json.dumps({'type': 'answer_chunk', 'content': fallback})}\n\n"
            yield f"data: {json.dumps({'type': 'answer_end'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process query without streaming (for testing)
        """
        if conversation_history is None:
            conversation_history = []

        messages = [
            {"role": "system", "content": self.create_system_prompt()}
        ]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})

        tools = self.get_tool_schemas()

        iteration = 0
        retrieved_context = ""
        tool_call_count = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Processing iteration {iteration}")

            try:
                # After tools are called, force answer generation
                if tool_call_count > 0:
                    response = self.client.chat.completions.create(
                        model=self.settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=0.7,
                        max_tokens=2000
                    )

                assistant_message = response.choices[0].message

                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    tool_call_count += len(assistant_message.tool_calls)

                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in assistant_message.tool_calls
                        ]
                    })

                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name

                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        tool_result = self.execute_tool(tool_name, arguments)

                        if tool_name in ["semantic_search", "multi_query_search"] and tool_result.get("success"):
                            context_data = tool_result.get("data", {})
                            if isinstance(context_data, dict):
                                new_context = context_data.get("context", "")
                                if new_context:
                                    retrieved_context += "\n\n" + new_context

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result)
                        })

                    continue

                else:
                    final_answer = assistant_message.content or ""

                    return {
                        "answer": final_answer,
                        "context": retrieved_context,
                        "iterations": iteration,
                        "tool_calls": tool_call_count,
                        "success": True
                    }

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                return {
                    "answer": f"Error: {str(e)}",
                    "context": retrieved_context,
                    "iterations": iteration,
                    "success": False
                }

        return {
            "answer": "I apologize, but I had trouble formulating an answer. Please try rephrasing your question.",
            "context": retrieved_context,
            "iterations": iteration,
            "success": False
        }