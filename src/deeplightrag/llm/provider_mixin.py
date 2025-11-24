"""
LLM Provider Mixin Classes
Shared functionality for all LLM providers to reduce code duplication
"""

import os
from typing import Any, Dict, Optional


class PromptHandlerMixin:
    """Mixin for common prompt and response handling across all providers"""

    @staticmethod
    def _get_api_key(env_var: str) -> Optional[str]:
        """
        Get API key from environment variable

        Args:
            env_var: Environment variable name

        Returns:
            API key or None
        """
        return os.getenv(env_var)

    @staticmethod
    def _parse_reasoning_and_answer(response_text: str) -> Dict[str, str]:
        """
        Parse reasoning and answer from response text
        Handles format: [REASONING]...text...[ANSWER]...text...

        Args:
            response_text: Raw response containing reasoning and answer

        Returns:
            Dictionary with 'reasoning' and 'answer' keys
        """
        reasoning = ""
        answer = response_text

        if "[REASONING]" in response_text and "[ANSWER]" in response_text:
            parts = response_text.split("[ANSWER]")
            if "[REASONING]" in parts[0]:
                reasoning = parts[0].split("[REASONING]")[1].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""

        return {"reasoning": reasoning, "answer": answer}

    @staticmethod
    def _build_generation_result(
        response_text: str, model: str, has_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Build standard generation result dictionary

        Args:
            response_text: Generated text
            model: Model name/identifier
            has_reasoning: Whether to parse reasoning

        Returns:
            Standard result dictionary
        """
        result = {"model": model, "tokens_used": len(response_text.split())}

        if has_reasoning:
            parsed = PromptHandlerMixin._parse_reasoning_and_answer(response_text)
            result.update(parsed)
        else:
            result["answer"] = response_text

        return result

    def _validate_response(self, response: Any) -> str:
        """
        Extract text from response object
        Handles different response formats from different providers

        Args:
            response: Response from LLM API

        Returns:
            Text content as string
        """
        # Handle LangChain ChatMessage format
        if hasattr(response, "content"):
            return response.content

        # Handle dict format
        if isinstance(response, dict):
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Handle string format
        return str(response).strip()


class ApiConfigMixin:
    """Mixin for managing provider-specific API configurations"""

    @staticmethod
    def _setup_api_environment(**kwargs) -> Dict[str, str]:
        """
        Setup environment variables for API access

        Args:
            **kwargs: API configuration key-value pairs

        Returns:
            Dictionary of set environment variables
        """
        env_vars = {}
        for key, value in kwargs.items():
            if value:
                os.environ[key] = value
                env_vars[key] = value
        return env_vars

    @staticmethod
    def _get_or_raise_api_key(
        config_key: Optional[str], env_var_name: str, provider_name: str
    ) -> str:
        """
        Get API key from config or environment, raise if not found

        Args:
            config_key: API key from config
            env_var_name: Environment variable name to check
            provider_name: Provider name for error message

        Returns:
            API key string

        Raises:
            ValueError: If API key not found in either location
        """
        api_key = config_key or os.getenv(env_var_name)
        if not api_key:
            raise ValueError(
                f"{provider_name} API key not provided and not found in {env_var_name} environment variable"
            )
        return api_key
