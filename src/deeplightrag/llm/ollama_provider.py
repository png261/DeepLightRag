"""
Ollama LLM Provider for DeepLightRAG
Supports local Ollama models like Llama 2, Mistral, etc.
"""

from typing import Any, Dict, Optional

from .base import BaseLLM, LLMConfig
from .provider_mixin import PromptHandlerMixin

try:
    from langchain_ollama import OllamaLLM as LangChainOllama

    HAS_LANGCHAIN_OLLAMA = True
except ImportError:
    HAS_LANGCHAIN_OLLAMA = False


class OllamaLLM(BaseLLM, PromptHandlerMixin):
    """Ollama LLM Provider using LangChain"""

    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama LLM

        Args:
            config: LLM configuration with provider='ollama'
                    base_url should be set (default: http://localhost:11434)
        """
        super().__init__(config)

        if not HAS_LANGCHAIN_OLLAMA:
            raise ImportError(
                "langchain-ollama not installed. " "Install with: pip install langchain-ollama"
            )

        # Check if Ollama is running
        base_url = config.base_url or "http://localhost:11434"
        self._verify_ollama_running(base_url)

        # Initialize LangChain OllamaLLM
        self.client = OllamaLLM(
            base_url=base_url,
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        self.model_info["initialized"] = True
        print(f"Ollama LLM initialized: {config.model} (base_url: {base_url})")

    @staticmethod
    def _verify_ollama_running(base_url: str) -> None:
        """Verify Ollama service is running"""
        try:
            import requests

            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API returned status {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {base_url}. "
                "Make sure Ollama is running: ollama serve"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to verify Ollama: {e}") from e

    def generate(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate answer from context and query

        Args:
            context: Retrieved context
            query: User query
            system_prompt: Optional system prompt

        Returns:
            Generated answer
        """
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        prompt = self._build_rag_prompt(system_prompt, context, query)

        try:
            response = self.client.invoke(prompt)
            return response.strip() if isinstance(response, str) else str(response)
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def generate_with_reasoning(
        self, context: str, query: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate answer with chain-of-thought reasoning"""
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        prompt = self._build_reasoning_prompt(system_prompt, context, query)

        try:
            response = self.client.invoke(prompt)
            response_text = response.strip() if isinstance(response, str) else str(response)
            parsed = self._parse_reasoning_and_answer(response_text)

            return {**parsed, "model": self.config.model, "tokens_used": len(response_text.split())}
        except Exception as e:
            raise RuntimeError(f"Ollama generation with reasoning failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            **self.model_info,
            "base_url": self.config.base_url or "http://localhost:11434",
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
        }
