# DeepLightRAG

DeepLightRAG is a high-performance document indexing and retrieval system designed to work with any Large Language Model (LLM). It features a dual-layer graph architecture (Visual-Spatial and Entity-Relationship) to provide context-aware and visually-grounded retrieval.

## Features

- **Dual-Layer Graph**: Combines visual layout awareness with semantic entity relationships.
- **Visual-Grounded Retrieval**: Retrieves not just text, but visual regions and their spatial context.
- **Robust OCR**: Integrated with DeepSeek-OCR and EasyOCR fallback for reliable text extraction.
- **Advanced NER**: Uses GLiNER for zero-shot entity recognition.
- **Flexible LLM Support**: Compatible with OpenAI, Google Gemini, Anthropic, and local LLMs via MLX/Ollama.

## Installation

### Standard Installation
```bash
pip install deeplightrag
```

### With GPU Support (NVIDIA CUDA)
For optimized performance using quantization (4-bit/8-bit):
```bash
pip install "deeplightrag[gpu]"
```

### For macOS (Apple Silicon)
For optimization on M1/M2/M3 chips:
```bash
pip install "deeplightrag[macos]"
```

## Usage

### Command Line Interface

Index a document:
```bash
# Basic usage
deeplightrag index document.pdf

# With custom configuration
deeplightrag index document.pdf --config config.yaml
```

Retrieve information:
```bash
deeplightrag retrieve "What is the main topic?" --config config.yaml
```

### Configuration File (config.yaml)

You can customize the model and system behavior using a YAML file:

```yaml
ocr:
  model_name: "deepseek-ai/deepseek-ocr"
  # Override MLX automatic selection (useful for some models)
  use_mlx: false 
  resolution: "base"

retrieval:
  top_k: 5
  rerank: true
```

### Python API

```python
from deeplightrag.core import DeepLightRAG

# Initialize with hardware auto-detection
rag = DeepLightRAG(config={"ocr": {"use_mlx": True}})

# Index
rag.index_document("research_paper.pdf", document_id="doc_001")

# Retrieve
result = rag.retrieve("Summarize the methodology")
print(result)
```

## License

MIT License
