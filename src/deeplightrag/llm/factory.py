"""
LLM Factory for Dynamic Provider Loading
Supports creating LLM instances from configuration
"""

from typing import Any, Dict, Type

from .base import BaseLLM, LLMConfig


class LLMFactory:
    """Factory for creating LLM instances from configuration"""

    # Provider mapping
    _providers: Dict[str, Type[BaseLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLM]) -> None:
        """
        Register a new LLM provider

        Args:
            name: Provider name (e.g., 'openai', 'anthropic')
            provider_class: LLM provider class
        """
        cls._providers[name.lower()] = provider_class

    @classmethod
    def from_config(cls, config: LLMConfig) -> BaseLLM:
        """
        Create LLM instance from configuration

        Args:
            config: LLMConfig instance

        Returns:
            Initialized LLM provider instance

        Raises:
            ValueError: If provider is not supported
            ImportError: If provider dependencies are not installed
        """
        provider_name = config.provider.lower()

        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown LLM provider: {config.provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class(config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> BaseLLM:
        """
        Create LLM instance from dictionary configuration

        Args:
            config_dict: Configuration dictionary with provider, model, etc.

        Returns:
            Initialized LLM provider instance
        """
        config = LLMConfig(**config_dict)
        return cls.from_config(config)

    @classmethod
    def list_providers(cls) -> list:
        """List all registered providers"""
        return list(cls._providers.keys())

    @classmethod
    def is_provider_available(cls, provider: str) -> bool:
        """Check if a provider is registered and available"""
        return provider.lower() in cls._providers


# Register built-in providers
def _register_builtin_providers():
    """Register built-in LLM providers"""
    import logging
    logger = logging.getLogger(__name__)

    # Core providers
    from .anthropic_provider import AnthropicLLM
    from .deepseek_provider import DeepSeekLLM
    from .openai_provider import OpenAILLM

    LLMFactory.register_provider("deepseek", DeepSeekLLM)
    LLMFactory.register_provider("openai", OpenAILLM)
    LLMFactory.register_provider("anthropic", AnthropicLLM)

    # Optional providers - handle import errors gracefully
    _register_optional_provider("huggingface", ".huggingface_provider", "HuggingFaceLLM", logger)
    _register_optional_provider("ollama", ".ollama_provider", "OllamaLLM", logger)
    _register_optional_provider("litellm", ".litellm_provider", "LiteLLMProvider", logger)


def _register_optional_provider(provider_name: str, module_path: str, class_name: str, logger) -> None:
    """Register an optional provider, skipping on import errors"""
    try:
        module = __import__(module_path, fromlist=[class_name], level=1)
        provider_class = getattr(module, class_name)
        LLMFactory.register_provider(provider_name, provider_class)
    except ImportError as e:
        logger.debug(f"Optional provider '{provider_name}' skipped: {e}")
    except Exception as e:
        logger.debug(f"Failed to register optional provider '{provider_name}': {e}")


# Register providers on import
_register_builtin_providers()
