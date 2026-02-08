from typing import List, Dict, Any
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import tiktoken


class EmbeddingService:


    def __init__(
            self,
            endpoint: str,
            api_key: str,
            deployment_name: str,
            api_version: str,
            max_tokens: int = 8191  # Safe limit for text-embedding-ada-002
    ):
        try:
            endpoint = endpoint.rstrip('/')

            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                timeout=60.0,
                max_retries=2
            )
            self.deployment_name = deployment_name
            self.max_tokens = max_tokens

            # Initialize tokenizer for text-embedding-ada-002
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except:
                logger.warning("Could not load tiktoken encoding, using character-based estimation")
                self.encoding = None

            logger.info(f"Initialized EmbeddingService with deployment: {deployment_name}")

        except Exception as e:
            logger.error(f"Error initializing AzureOpenAI client: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Rough estimation: 1 token ≈ 4 characters
            return len(text) // 4

    def truncate_text(self, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within token limit"""
        if max_tokens is None:
            max_tokens = self.max_tokens

        if self.encoding:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text

            # Truncate tokens and decode back
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            logger.warning(f"Truncated text from {len(tokens)} to {max_tokens} tokens")
            return truncated_text
        else:
            # Character-based truncation (rough estimate)
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text

            logger.warning(f"Truncated text from {len(text)} to {max_chars} characters")
            return text[:max_chars]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text with retry logic"""
        try:
            # Truncate text if necessary
            truncated_text = self.truncate_text(text)

            # Verify token count
            token_count = self.count_tokens(truncated_text)
            if token_count > self.max_tokens:
                logger.error(f"Text still too long after truncation: {token_count} tokens")
                # Force truncate to safe limit
                truncated_text = self.truncate_text(text, max_tokens=self.max_tokens - 100)

            response = self.client.embeddings.create(
                model=self.deployment_name,
                input=truncated_text
            )
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(
            self,
            texts: List[str],
            batch_size: int = 10
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []

        # Truncate all texts first
        truncated_texts = [self.truncate_text(text) for text in texts]

        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]

            try:
                # Verify batch doesn't exceed limits
                for idx, text in enumerate(batch):
                    token_count = self.count_tokens(text)
                    if token_count > self.max_tokens:
                        logger.warning(f"Text {i + idx} has {token_count} tokens, re-truncating")
                        batch[idx] = self.truncate_text(text, max_tokens=self.max_tokens - 100)

                response = self.client.embeddings.create(
                    model=self.deployment_name,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                logger.info(
                    f"Generated embeddings for batch {i // batch_size + 1}/{(len(truncated_texts) - 1) // batch_size + 1}")

            except Exception as e:
                logger.error(f"Error in batch {i // batch_size + 1}: {e}")
                # Fallback to individual processing
                for text in batch:
                    try:
                        emb = self.generate_embedding(text)
                        embeddings.append(emb)
                    except Exception as e2:
                        logger.error(f"Failed to generate embedding for text: {e2}")
                        # Use zero vector as fallback (last resort)
                        logger.warning("Using zero vector as fallback")
                        embeddings.append([0.0] * 1536)

        return embeddings

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embeddings to document chunks"""
        texts = [chunk['text'] for chunk in chunks]

        # Log token statistics
        token_counts = [self.count_tokens(text) for text in texts]
        logger.info(
            f"Token statistics - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts) / len(token_counts):.1f}")

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings_batch(texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding

        logger.info("Successfully embedded all chunks")
        return chunks

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)