"""
LiteLLM Provider for DeepLightRAG
Universal LLM provider supporting 100+ models from different providers
Supports OpenAI, Anthropic, Cohere, Replicate, HuggingFace, Ollama, Azure, AWS, Google, etc.
"""

from typing import Any, Dict, Optional

from .base import BaseLLM, LLMConfig
from .provider_mixin import ApiConfigMixin, PromptHandlerMixin

try:
    import litellm

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


class LiteLLMProvider(BaseLLM, PromptHandlerMixin, ApiConfigMixin):
    """Universal LLM Provider using LiteLLM"""

    def __init__(self, config: LLMConfig):
        """
        Initialize LiteLLM Provider

        Args:
            config: LLM configuration with any model identifier supported by LiteLLM
                    Examples:
                    - openai/gpt-4
                    - claude-3-opus-20240229
                    - cohere/command
                    - replicate/llama-7b
                    - huggingface/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
                    - ollama/llama2
                    - azure/deployment-name
                    - etc.
        """
        super().__init__(config)

        if not HAS_LITELLM:
            raise ImportError("litellm not installed. " "Install with: pip install litellm")

        # Store model identifier
        self.model_id = config.model
        self.base_url = config.base_url

        # Configure LiteLLM with parameters
        self.litellm_config = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "timeout": config.timeout,
            "num_retries": config.retry_attempts,
        }

        # Remove None values
        self.litellm_config = {k: v for k, v in self.litellm_config.items() if v is not None}

        # Set API keys if provided
        if config.api_key:
            import os

            # Get provider from model ID or config
            provider = config.provider.lower()
            env_var_mapping = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "cohere": "COHERE_API_KEY",
                "replicate": "REPLICATE_API_KEY",
                "huggingface": "HUGGINGFACEHUB_API_TOKEN",
                "azure": "AZURE_API_KEY",
                "aws": "AWS_ACCESS_KEY_ID",
                "google": "GOOGLE_API_KEY",
            }
            env_var = env_var_mapping.get(provider, f"{provider.upper()}_API_KEY")
            os.environ[env_var] = config.api_key

        # Set custom base URL if provided
        if config.base_url:
            litellm.api_base = config.base_url

        # Set custom parameters
        if config.custom_params:
            self.litellm_config.update(config.custom_params)

        self.model_info["initialized"] = True
        print(f"LiteLLM Provider initialized: {config.model}")

    def generate(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        """Generate answer from context and query"""
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        prompt = self._build_rag_prompt(system_prompt, context, query)

        try:
            response = litellm.completion(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **self.litellm_config,
            )
            return self._validate_response(response)
        except Exception as e:
            raise RuntimeError(f"LiteLLM generation failed: {e}") from e

    def generate_with_reasoning(
        self, context: str, query: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate answer with chain-of-thought reasoning"""
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        prompt = self._build_reasoning_prompt(system_prompt, context, query)

        try:
            response = litellm.completion(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                **self.litellm_config,
            )

            response_text = self._validate_response(response)
            parsed = self._parse_reasoning_and_answer(response_text)

            return {**parsed, "model": self.config.model, "tokens_used": len(response_text.split())}
        except Exception as e:
            raise RuntimeError(f"LiteLLM generation with reasoning failed: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            **self.model_info,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "supports_providers": [
                "openai",
                "anthropic",
                "cohere",
                "replicate",
                "huggingface",
                "azure",
                "aws",
                "google",
                "ollama",
                "together",
                "nlp-cloud",
                "petals",
                "aleph-alpha",
                "baseten",
                "baseten-production",
                "and 80+ more providers",
            ],
        }

    @staticmethod
    def get_supported_models() -> Dict[str, list]:
        """Get list of supported models by provider"""
        return {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            "cohere": ["command-r-plus", "command-r", "command"],
            "huggingface": ["meta-llama/Llama-2-7b", "mistralai/Mistral-7B", "tiiuae/Falcon-7b"],
            "ollama": ["llama2", "mistral", "neural-chat", "orca-mini"],
            "google": ["gemini-pro", "palm-2"],
            "azure": ["deployment-name (configure via AZURE_*)"],
            "replicate": ["meta/llama-2-7b", "stability-ai/stable-diffusion"],
        }
