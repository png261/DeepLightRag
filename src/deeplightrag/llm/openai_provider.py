"""
OpenAI LLM Provider for DeepLightRAG

Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
Refactored to use AbstractLLMProvider for reduced code duplication.
"""

import logging
from typing import Any

from .abstract_provider import AbstractLLMProvider

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI

    HAS_LANGCHAIN_OPENAI = True
except ImportError:
    HAS_LANGCHAIN_OPENAI = False
    ChatOpenAI = None  # type: ignore


class OpenAILLM(AbstractLLMProvider):
    """
    OpenAI LLM Provider using LangChain.

    Inherits all common functionality (prompt building, response parsing, etc.)
    from AbstractLLMProvider. Only implements OpenAI-specific client initialization.
    """

    def initialize_client(self) -> Any:
        """
        Initialize and return OpenAI ChatOpenAI client.

        Returns:
            Initialized ChatOpenAI instance

        Raises:
            ImportError: If langchain-openai is not installed
            ValueError: If API key is not found
        """
        if not HAS_LANGCHAIN_OPENAI:
            raise ImportError(
                "langchain-openai not installed. "
                "Install with: pip install 'deeplightrag[llm]' or 'langchain-openai'"
            )

        api_key = self._get_or_raise_api_key(
            self.config.api_key, "OPENAI_API_KEY", "OpenAI"
        )

        return ChatOpenAI(
            api_key=api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=self.config.timeout,
            max_retries=self.config.retry_attempts,
        )

    def _call_model(
        self, messages: list[dict[str, str]], temperature: float | None = None
    ) -> str:
        """
        Call the OpenAI model with given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override

        Returns:
            Model response text

        Raises:
            RuntimeError: If API call fails
        """
        try:
            # Convert messages to LangChain format
            from langchain_core.messages import HumanMessage, SystemMessage

            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                else:
                    lc_messages.append(HumanMessage(content=msg["content"]))

            # Call model with optional temperature override
            if temperature is not None:
                response = self.client.with_config(
                    configurable={"temperature": temperature}
                ).invoke(lc_messages)
            else:
                response = self.client.invoke(lc_messages)

            return response.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def get_model_info(self) -> dict:
        """Get OpenAI model information"""
        return self.model_info
