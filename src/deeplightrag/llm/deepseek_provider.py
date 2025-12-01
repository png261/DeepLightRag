"""
DeepSeek LLM Provider for DeepLightRAG

Wraps the existing DeepSeekR1 implementation with the BaseLLM interface.
Refactored to use AbstractLLMProvider for reduced code duplication.
"""

import logging
from typing import Any

from .abstract_provider import AbstractLLMProvider
from .deepseek_r1 import DeepSeekR1

logger = logging.getLogger(__name__)


class DeepSeekLLM(AbstractLLMProvider):
    """
    DeepSeek LLM Provider using MLX backend.

    Wraps the existing DeepSeekR1 implementation.
    Inherits common functionality from AbstractLLMProvider.
    """

    def initialize_client(self) -> Any:
        """
        Initialize and return DeepSeekR1 client.

        Returns:
            Initialized DeepSeekR1 instance

        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            return DeepSeekR1(
                model_name=self.config.model,
                quantization="4bit",  # Default for Apple Silicon
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        except ImportError as e:
            raise ImportError(
                f"DeepSeek/MLX dependencies not installed: {e}. "
                "Install with: pip install 'deeplightrag[mlx]'"
            ) from e

    def _call_model(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        """
        Call the DeepSeek model with given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override

        Returns:
            Model response text

        Raises:
            RuntimeError: If API call fails
        """
        try:
            # Extract context and query from messages
            context = ""
            query = ""
            for msg in messages:
                if msg["role"] == "user":
                    # Parse context and question from combined message
                    content = msg["content"]
                    if "Context:" in content and "Question:" in content:
                        parts = content.split("Context:")
                        if len(parts) > 1:
                            context_part = parts[1].split("Question:")[0].strip()
                            context = context_part
                            query_part = parts[1].split("Question:")[1].strip()
                            query = query_part
                    else:
                        query = content

            # DeepSeekR1 takes context and query directly
            response = self.client.generate(context, query)
            return self._validate_response(response)
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            raise RuntimeError(f"DeepSeek error: {e}") from e

    def get_model_info(self) -> dict:
        """Get DeepSeek model information"""
        return self.model_info
