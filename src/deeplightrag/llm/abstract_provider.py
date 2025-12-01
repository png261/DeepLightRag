"""
Abstract LLM Provider Base Class

Provides common implementation logic for all LLM providers, eliminating code duplication.
Concrete providers only need to implement client initialization and model-specific details.
"""

import logging
import os
import re
from abc import abstractmethod
from typing import Any, Dict, Optional

from .base import BaseLLM, LLMConfig

logger = logging.getLogger(__name__)


class AbstractLLMProvider(BaseLLM):
    """
    Abstract base class for LLM providers.

    This class consolidates common functionality across all LLM implementations:
    - Prompt building and formatting
    - Response validation and parsing
    - Error handling and logging
    - API configuration management

    Subclasses only need to implement:
    1. initialize_client() - Create and return the LLM client
    2. _call_model() - Call the actual LLM client
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize abstract provider.

        Args:
            config: LLM configuration

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        self.client = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the provider. Can be overridden by subclasses."""
        try:
            self.client = self.initialize_client()
            self.model_info["initialized"] = True
            logger.info(
                f"Initialized {self.config.provider} LLM",
                extra={"model": self.config.model},
            )
        except ImportError as e:
            logger.error(f"Missing dependencies for {self.config.provider}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize {self.config.provider}: {e}")
            raise

    def initialize_client(self) -> Any:
        """
        Initialize and return the LLM client.

        Default implementation for providers that don't need special initialization.
        Subclasses can override this method for provider-specific initialization.

        Returns:
            The initialized LLM client

        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If configuration is invalid
        """
        # Default implementation - return self as the client
        return self

    def _call_model(
        self, messages: list[dict[str, str]], temperature: Optional[float] = None
    ) -> str:
        """
        Call the LLM model with given messages.

        Default implementation that extracts content and uses generate method.
        Subclasses should override this method for provider-specific logic.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Optional temperature override

        Returns:
            Model response text

        Raises:
            RuntimeError: If API call fails
        """
        # Extract user message and any system context
        user_message = ""
        system_message = ""

        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
            elif msg.get("role") == "system":
                system_message = msg.get("content", "")

        # Use generate method if available
        if hasattr(self, "generate") and callable(getattr(self, "generate")):
            return self.generate(context="", query=user_message, system_prompt=system_message)

        # Fallback response
        return f"Response to: {user_message}"

    def generate(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate answer from context and query.

        Args:
            context: Retrieved context
            query: User query
            system_prompt: Optional system prompt (uses default if not provided)

        Returns:
            Generated answer

        Raises:
            ValueError: If context or query is invalid
            RuntimeError: If API call fails
        """
        self._validate_inputs(context, query)

        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        messages = self._build_messages(system_prompt, context, query)

        try:
            response = self._call_model(messages, self.config.temperature)
            return self._validate_response(response)
        except Exception as e:
            logger.error(f"Generation failed for {self.config.provider}: {e}")
            raise

    def generate_with_reasoning(
        self, context: str, query: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer with reasoning/chain-of-thought.

        Args:
            context: Retrieved context
            query: User query
            system_prompt: Optional system prompt

        Returns:
            Dictionary with 'reasoning' and 'answer' keys, plus metadata

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If API call fails
        """
        self._validate_inputs(context, query)

        if not system_prompt:
            system_prompt = self._get_system_prompt_with_reasoning()

        messages = self._build_messages(system_prompt, context, query)

        try:
            response = self._call_model(messages, self.config.temperature)
            reasoning, answer = self._parse_reasoning_and_answer(response)

            return {
                "reasoning": reasoning,
                "answer": answer,
                "model": self.config.model,
                "tokens_used": self._estimate_tokens(response),
                "provider": self.config.provider,
            }
        except Exception as e:
            logger.error(f"Reasoning generation failed for {self.config.provider}: {e}")
            raise

    # Utility Methods

    @staticmethod
    def _validate_inputs(context: str, query: str) -> None:
        """
        Validate input parameters.

        Args:
            context: Retrieved context
            query: User query

        Raises:
            ValueError: If inputs are invalid
        """
        if not context or not context.strip():
            raise ValueError("Context cannot be empty")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(context) > 50000:
            logger.warning("Context is very large (>50K chars), may hit token limits")
        if len(query) > 5000:
            raise ValueError("Query exceeds maximum length of 5000 characters")

    @staticmethod
    def _validate_response(response: str) -> str:
        """
        Validate and clean model response.

        Args:
            response: Raw model response

        Returns:
            Cleaned response

        Raises:
            ValueError: If response is invalid
        """
        if not response or not response.strip():
            raise ValueError("Empty response from model")
        return response.strip()

    def _build_messages(self, system_prompt: str, context: str, query: str) -> list[dict[str, str]]:
        """
        Build message list in OpenAI format.

        Args:
            system_prompt: System instruction
            context: Retrieved context
            query: User query

        Returns:
            List of message dicts
        """
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]

    @staticmethod
    def _get_default_system_prompt() -> str:
        """
        Get default system prompt for answer generation.

        Returns:
            System prompt text
        """
        return (
            "You are a helpful assistant that answers questions based on provided context. "
            "Be concise and accurate. If the context doesn't contain relevant information, "
            "say so explicitly."
        )

    @staticmethod
    def _get_system_prompt_with_reasoning() -> str:
        """
        Get system prompt that encourages reasoning.

        Returns:
            System prompt with reasoning instructions
        """
        return (
            "You are a helpful assistant. First, analyze the context and question carefully. "
            "Show your reasoning step by step in a <reasoning> section. "
            "Then provide your final answer in an <answer> section. "
            "Format: <reasoning>your thinking here</reasoning><answer>your answer here</answer>"
        )

    @staticmethod
    def _parse_reasoning_and_answer(response: str) -> tuple[str, str]:
        """
        Parse reasoning and answer from response.

        Looks for <reasoning>...</reasoning> and <answer>...</answer> sections.
        Falls back to using entire response as answer if sections not found.

        Args:
            response: Model response text

        Returns:
            Tuple of (reasoning, answer)
        """
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else response.strip()

        if not answer:
            answer = response.strip()

        return reasoning, answer

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count for text (rough approximation).

        Uses 1 token ≈ 4 characters rule of thumb.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text.split()) + len(re.findall(r"\S+", text))

    @classmethod
    def _get_or_raise_api_key(cls, key: Optional[str], env_var: str, provider_name: str) -> str:
        """
        Get API key from argument or environment variable.

        Args:
            key: Directly provided API key
            env_var: Environment variable name
            provider_name: Provider name for error messages

        Returns:
            API key

        Raises:
            ValueError: If key not found
        """
        if key:
            return key

        key = os.getenv(env_var)
        if not key:
            raise ValueError(
                f"{provider_name} API key not found. "
                f"Provide via config or set {env_var} environment variable."
            )

        return key
