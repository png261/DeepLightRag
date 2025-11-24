"""
Base LLM Interface for DeepLightRAG
Provides abstraction for different LLM providers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    """LLM Configuration"""

    provider: str  # openai, anthropic, huggingface, ollama, deepseek, etc.
    model: str  # Model name/ID
    api_key: Optional[str] = None  # API key if needed
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Provider-specific config
    base_url: Optional[str] = None  # For local models, Ollama, etc.
    timeout: int = 30
    retry_attempts: int = 3

    # Custom parameters
    custom_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


class BaseLLM(ABC):
    """Base class for LLM implementations"""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM

        Args:
            config: LLM configuration
        """
        self.config = config
        self.model_info = {"provider": config.provider, "model": config.model, "initialized": False}

    @abstractmethod
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
        ...

    @abstractmethod
    def generate_with_reasoning(
        self, context: str, query: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer with reasoning/chain-of-thought

        Args:
            context: Retrieved context
            query: User query
            system_prompt: Optional system prompt

        Returns:
            Dict with answer and reasoning
        """
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info

    @staticmethod
    def _get_default_system_prompt() -> str:
        """Get default system prompt for RAG"""
        return """You are a helpful assistant that answers questions based on the provided context.

Your responses should be:
1. **Accurate** - Only use information from the provided context
2. **Concise** - Be direct and to the point
3. **Well-structured** - Use clear formatting when appropriate
4. **Honest** - If the context doesn't contain enough information, say so
5. **Referenced** - When possible, mention the source (page, section, entity)

When analyzing:
- **Tables/Figures**: Describe what the visual shows and interpret the data
- **Multiple entities**: Explain relationships between them
- **Contradictions**: Note if different parts of context conflict
- **Uncertainties**: Indicate confidence levels in your answer"""

    @staticmethod
    def _build_rag_prompt(system_prompt: str, context: str, query: str) -> str:
        """Build RAG prompt"""
        return f"""[SYSTEM]
{system_prompt}

[CONTEXT]
{context}

[QUESTION]
{query}

[ANSWER]
"""

    @staticmethod
    def _build_reasoning_prompt(system_prompt: str, context: str, query: str) -> str:
        """Build prompt for reasoning/chain-of-thought"""
        return f"""[SYSTEM]
{system_prompt}

Think through this step by step:
1. What information is relevant to answering the question?
2. What connections can you make between different parts of the context?
3. What is the most accurate answer based on the evidence?

[CONTEXT]
{context}

[QUESTION]
{query}

[REASONING]
Let me think about this step by step:

[ANSWER]
"""
