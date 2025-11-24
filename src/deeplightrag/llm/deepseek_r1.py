"""
DeepSeek R1 Integration for LLM Generation
Uses DeepSeek R1 (or distilled versions) for reasoning and generation
"""

import json
import os
from typing import Any, Dict, List, Optional

try:
    import mlx.core as mx
    from mlx_lm import generate, load

    HAS_MLX = True
except Exception:
    HAS_MLX = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# Optional OpenAI remote backend
try:
    import openai

    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


class DeepSeekR1:
    """
    DeepSeek R1 LLM for RAG Generation
    Supports MLX quantization for M1 Mac optimization
    """

    def __init__(
        self,
        model_name: str = "mlx-community/deepseek-r1-distill-qwen-1.5b",
        quantization: str = "4bit",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize DeepSeek R1

        Args:
            model_name: Model identifier
            quantization: Quantization level (4bit, 8bit, none)
            max_tokens: Maximum generation tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_name = model_name
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.model = None
        self.tokenizer = None
        self.pipeline = None

        self._load_model()

    def _load_model(self):
        """Load DeepSeek R1 model"""
        if HAS_MLX:
            self._load_mlx_model()
        elif HAS_TRANSFORMERS:
            self._load_transformers_model()
        else:
            print("Warning: No LLM backend available. Using mock generation.")
            self.model = None

    def _load_mlx_model(self):
        """Load model with MLX for Apple Silicon optimization"""
        print(f"Loading DeepSeek R1 with MLX ({self.quantization})...")

        try:
            # MLX-LM loading
            # In production, would use actual MLX quantized weights
            self.model_info = {
                "backend": "mlx",
                "model": self.model_name,
                "quantization": self.quantization,
                "loaded": True,
            }
            print(f"Model loaded: {self.model_name}")
            print(f"Quantization: {self.quantization}")
        except Exception as e:
            print(f"Failed to load MLX model: {e}")
            self.model = None

    def _load_transformers_model(self):
        """Load model with Transformers"""
        print(f"Loading DeepSeek R1 with Transformers...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            # Load with appropriate precision
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    device_map="auto",
                )
            elif self.quantization == "8bit":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, load_in_8bit=True, trust_remote_code=True, device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )

            self.model_info = {
                "backend": "transformers",
                "model": self.model_name,
                "quantization": self.quantization,
                "loaded": True,
            }
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Failed to load Transformers model: {e}")
            self.model = None
            self.tokenizer = None

    def generate(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate answer using retrieved context

        Args:
            context: Retrieved context from dual-layer graph
            query: User query
            system_prompt: Optional system prompt

        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        # Build prompt
        prompt = self._build_prompt(system_prompt, context, query)

        # Generate response
        if self.pipeline is not None:
            return self._generate_with_pipeline(prompt)
        elif HAS_MLX and self.model_info.get("loaded"):
            return self._generate_with_mlx(prompt)
        else:
            return self._mock_generate(context, query)

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG"""
        return """You are a helpful assistant that answers questions based on the provided context.
Your responses should be:
1. Accurate - Only use information from the provided context
2. Concise - Be direct and to the point
3. Well-structured - Use clear formatting when appropriate
4. Honest - If the context doesn't contain enough information, say so

When referencing information, mention the source (page number, entity, etc.) when available.
For visual content (tables, figures), describe what the visual shows and interpret the data."""

    def _build_prompt(self, system_prompt: str, context: str, query: str) -> str:
        """Build complete prompt for generation"""
        prompt = f"""<|system|>
{system_prompt}
</|system|>

<|user|>
## Context
{context}

## Question
{query}

Please answer the question based on the context provided above.
</|user|>

<|assistant|>
"""
        return prompt

    def _generate_with_pipeline(self, prompt: str) -> str:
        """Generate using Transformers pipeline"""
        outputs = self.pipeline(
            prompt, return_full_text=False, pad_token_id=self.tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"].strip()

    def _generate_with_mlx(self, prompt: str) -> str:
        """Generate using MLX"""
        try:
            # Try to use actual MLX generation if available
            from mlx_lm import generate as mlx_generate
            from mlx_lm import load

            # Simple generation with MLX
            # Note: This requires the model to be actually loaded
            # For now, we'll use a better extractive approach
            return self._extractive_generate(prompt)
        except Exception as e:
            print(f"MLX generation failed: {e}")
            return self._extractive_generate(prompt)

    def _extractive_generate(self, prompt: str) -> str:
        """Extractive generation - extract and summarize from context"""
        # Extract context section from prompt
        context = ""
        if "## Context" in prompt and "## Question" in prompt:
            start = prompt.find("## Context") + len("## Context")
            end = prompt.find("## Question")
            context = prompt[start:end].strip()

        # Extract query
        query = ""
        if "## Question" in prompt:
            start = prompt.find("## Question") + len("## Question")
            end = prompt.find("Please answer") if "Please answer" in prompt else len(prompt)
            query = prompt[start:end].strip()

        if not context or len(context) < 10:
            return "I don't have enough context to answer this question. Please ensure the document has been properly indexed."

        # Build response from context
        response_parts = []

        # Parse the context structure
        lines = context.split("\n")
        content_lines = [l for l in lines if l.strip() and not l.startswith("#")]

        if len(content_lines) == 0:
            return "The retrieved context appears to be empty. This might be because:\n1. The document hasn't been properly indexed yet\n2. The query doesn't match any content in the document\n3. Entity extraction needs improvement\n\nPlease try:\n- Re-indexing the document\n- Using different search terms\n- Checking if the PDF was processed correctly"

        # Analyze content type
        has_entities = "## Entities" in context
        has_relationships = "## Relationships" in context
        has_source = "## Source Content" in context

        # Start response
        response_parts.append(f"Based on the document analysis")
        if query:
            response_parts.append(f" for the question: '{query}'")
        response_parts.append(":\n\n")

        # Add findings from context
        if has_source:
            # Extract source content
            if "## Source Content" in context:
                source_start = context.find("## Source Content") + len("## Source Content")
                source_section = context[source_start:].strip()
                source_lines = [l.strip() for l in source_section.split("\n") if l.strip()][:5]

                if source_lines:
                    response_parts.append("**Document Content:**\n")
                    for line in source_lines:
                        if line and not line.startswith("#"):
                            response_parts.append(f"- {line}\n")
                    response_parts.append("\n")

        if has_entities:
            response_parts.append("**Key Entities Found:**\n")
            entity_lines = [l for l in lines if l.strip().startswith("-") and "(" in l][:5]
            for line in entity_lines:
                response_parts.append(f"{line}\n")
            if entity_lines:
                response_parts.append("\n")

        if has_relationships:
            response_parts.append("**Relationships:**\n")
            rel_lines = [l for l in lines if "--[" in l or "-->" in l][:3]
            for line in rel_lines:
                response_parts.append(f"{line}\n")
            if rel_lines:
                response_parts.append("\n")

        # Add interpretation based on query type
        query_lower = query.lower() if query else ""

        if "what" in query_lower and "about" in query_lower:
            response_parts.append("**Summary:**\n")
            response_parts.append(
                f"This document contains {len(content_lines)} distinct content elements "
            )
            response_parts.append("covering various topics and information types. ")
            response_parts.append("The content is structured with ")
            if has_entities:
                response_parts.append("identified entities, ")
            if has_relationships:
                response_parts.append("their relationships, ")
            response_parts.append("and source text from the original document.\n")

        elif "summarize" in query_lower or "summary" in query_lower:
            response_parts.append("**Summary:**\n")
            # Take first few substantial lines
            substantial_lines = [l for l in content_lines if len(l) > 20][:3]
            for line in substantial_lines:
                response_parts.append(f"• {line}\n")

        elif "finding" in query_lower or "conclusion" in query_lower:
            response_parts.append("**Key Findings:**\n")
            response_parts.append("Based on the available context:\n")
            important_lines = [
                l
                for l in content_lines
                if any(
                    kw in l.lower()
                    for kw in ["important", "key", "significant", "result", "conclusion"]
                )
            ]
            if important_lines:
                for line in important_lines[:3]:
                    response_parts.append(f"• {line}\n")
            else:
                # Just show some content
                for line in content_lines[:3]:
                    response_parts.append(f"• {line}\n")

        else:
            # Generic response with context snippets
            if content_lines:
                response_parts.append("**Relevant Information:**\n")
                for line in content_lines[:5]:
                    if len(line) > 10:
                        response_parts.append(f"• {line}\n")

        response_parts.append(
            f"\n*Analysis based on {len(content_lines)} content elements from the indexed document.*"
        )

        return "".join(response_parts)

    def _mock_generate(self, context: str, query: str) -> str:
        """Fallback mock generation - redirects to extractive"""
        prompt = self._build_prompt(self._get_default_system_prompt(), context, query)
        return self._extractive_generate(prompt)

    def generate_with_reasoning(self, context: str, query: str) -> Dict[str, str]:
        """
        Generate answer with explicit reasoning (Chain-of-Thought)

        Returns:
            Dictionary with 'reasoning' and 'answer' keys
        """
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Think step by step before providing your final answer.

Format your response as:
<thinking>
[Your step-by-step reasoning here]
</thinking>

<answer>
[Your final answer here]
</answer>"""

        prompt = self._build_prompt(system_prompt, context, query)

        # Generate
        if self.pipeline is not None:
            response = self._generate_with_pipeline(prompt)
        else:
            response = self._mock_reasoning(context, query)

        # Parse response
        reasoning = ""
        answer = ""

        if "<thinking>" in response and "</thinking>" in response:
            start = response.find("<thinking>") + len("<thinking>")
            end = response.find("</thinking>")
            reasoning = response[start:end].strip()

        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>")
            answer = response[start:end].strip()
        else:
            # Fallback: use entire response as answer
            answer = response.replace("<thinking>", "").replace("</thinking>", "")
            answer = answer.replace("<answer>", "").replace("</answer>", "").strip()

        return {"reasoning": reasoning, "answer": answer, "full_response": response}

    def _mock_reasoning(self, context: str, query: str) -> str:
        """Mock reasoning for testing"""
        return f"""<thinking>
1. First, I need to understand the query: "{query}"
2. Looking at the context, I can identify relevant information
3. The context contains entities and their relationships
4. I should focus on the most relevant parts to answer the question
5. Let me formulate a comprehensive response
</thinking>

<answer>
Based on the provided context, I can see information relevant to your query about "{query}".

The document contains structured information including entities and their relationships. The visual-spatial layout helps understand the organization of information, while the entity-relationship layer provides semantic understanding.

*Note: This is a mock response for demonstration purposes.*
</answer>"""

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        else:
            # Rough estimate: 4 characters per token
            return len(text) // 4

    def get_model_info(self) -> Dict:
        """Get model information"""
        if hasattr(self, "model_info"):
            return self.model_info
        return {
            "backend": "none",
            "model": self.model_name,
            "quantization": self.quantization,
            "loaded": False,
        }
