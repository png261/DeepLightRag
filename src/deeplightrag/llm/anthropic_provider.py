"""
Anthropic LLM Provider for DeepLightRAG

Supports Claude-3 family and other Anthropic models.
Refactored to use AbstractLLMProvider for reduced code duplication.
"""

import logging
from typing import TYPE_CHECKING, Any

from .abstract_provider import AbstractLLMProvider

logger = logging.getLogger(__name__)

try:
    from langchain_anthropic import ChatAnthropic

    HAS_LANGCHAIN_ANTHROPIC = True
except ImportError:
    HAS_LANGCHAIN_ANTHROPIC = False
    ChatAnthropic = None  # type: ignore


class AnthropicLLM(AbstractLLMProvider):
    """
    Anthropic LLM Provider using LangChain.

    Inherits all common functionality from AbstractLLMProvider.
    Only implements Anthropic-specific client initialization.
    """

    def initialize_client(self) -> Any:
        """
        Initialize and return Anthropic ChatAnthropic client.

        Returns:
            Initialized ChatAnthropic instance

        Raises:
            ImportError: If langchain-anthropic is not installed
            ValueError: If API key is not found
        """
        if not HAS_LANGCHAIN_ANTHROPIC:
            raise ImportError(
                "langchain-anthropic not installed. "
                "Install with: pip install 'deeplightrag[llm]' or 'langchain-anthropic'"
            )

        api_key = self._get_or_raise_api_key(
            self.config.api_key, "ANTHROPIC_API_KEY", "Anthropic"
        )

        return ChatAnthropic(
            api_key=api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            max_retries=self.config.retry_attempts,
        )

    def _call_model(
        self, messages: list[dict[str, str]], temperature: float | None = None
    ) -> str:
        """
        Call the Anthropic model with given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override

        Returns:
            Model response text

        Raises:
            RuntimeError: If API call fails
        """
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                else:
                    lc_messages.append(HumanMessage(content=msg["content"]))

            # Call model
            response = self.client.invoke(lc_messages)
            return response.content
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise RuntimeError(f"Anthropic API error: {e}") from e

    def get_model_info(self) -> dict:
        """Get Anthropic model information"""
        return self.model_info
