"""
HuggingFace LLM Provider for DeepLightRAG
Supports open-source models like Llama, Mistral, etc.
"""

from typing import Any, Dict, Optional

from .base import BaseLLM, LLMConfig
from .provider_mixin import ApiConfigMixin, PromptHandlerMixin

try:
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline

    HAS_LANGCHAIN_HF = True
except ImportError:
    HAS_LANGCHAIN_HF = False


class HuggingFaceLLM(BaseLLM, PromptHandlerMixin, ApiConfigMixin):
    """HuggingFace LLM Provider using LangChain"""

    def __init__(self, config: LLMConfig):
        """
        Initialize HuggingFace LLM

        Args:
            config: LLM configuration with provider='huggingface'
        """
        super().__init__(config)

        if not HAS_LANGCHAIN_HF:
            raise ImportError(
                "langchain-huggingface not installed. "
                "Install with: pip install langchain-huggingface"
            )

        # Use local model or Hugging Face Inference API
        if config.base_url:
            # Custom endpoint (HuggingFace Inference API)
            api_key = self._get_or_raise_api_key(
                config.api_key, "HUGGINGFACEHUB_API_TOKEN", "HuggingFace"
            )

            self.client = HuggingFaceEndpoint(
                endpoint_url=config.base_url,
                huggingfacehub_api_token=api_key,
                model_kwargs={
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "top_p": config.top_p,
                },
            )
        else:
            # Local model
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

                model = AutoModelForCausalLM.from_pretrained(
                    config.model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                )
                tokenizer = AutoTokenizer.from_pretrained(config.model)

                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    do_sample=True,
                    top_p=config.top_p,
                    temperature=config.temperature,
                    max_new_tokens=config.max_tokens,
                )

                self.client = HuggingFacePipeline(
                    pipeline=pipe,
                    model_kwargs={
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "top_p": config.top_p,
                    },
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load local HuggingFace model: {e}") from e

        self.model_info["initialized"] = True
        print(f"HuggingFace LLM initialized: {config.model}")

    def generate(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        """Generate answer from context and query"""
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        prompt = self._build_rag_prompt(system_prompt, context, query)

        try:
            response = self.client.invoke(prompt)
            # Handle HuggingFace-specific response format
            if isinstance(response, dict):
                return response.get("generated_text", str(response)).strip()
            return str(response).strip()
        except Exception as e:
            raise RuntimeError(f"HuggingFace generation failed: {e}") from e

    def generate_with_reasoning(
        self, context: str, query: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate answer with chain-of-thought reasoning"""
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()

        prompt = self._build_reasoning_prompt(system_prompt, context, query)

        try:
            response = self.client.invoke(prompt)

            # Handle HuggingFace-specific response format
            response_text = (
                response.get("generated_text", str(response))
                if isinstance(response, dict)
                else str(response)
            )
            parsed = self._parse_reasoning_and_answer(response_text)

            return {**parsed, "model": self.config.model, "tokens_used": len(response_text.split())}
        except Exception as e:
            raise RuntimeError(f"HuggingFace generation with reasoning failed: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            **self.model_info,
            "endpoint_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
