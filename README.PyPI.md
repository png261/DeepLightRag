# DeepLightRAG

**Efficient Document-based RAG with Vision-Text Compression and GPU Acceleration**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **9-10x Vision-Text Compression** using DeepSeek-OCR
- **Dual-Layer Graph Architecture** (Visual-Spatial + Entity-Relationship)
- **Adaptive Token Budgeting** (2K-12K tokens vs 30K fixed)
- **Auto GPU Detection** - CUDA, MPS, or CPU
- **60-80% Cost Savings** compared to traditional RAG
- **LLM Fallback** for robust entity/relation extraction
- **Cross-Platform** - Windows, Linux, macOS

## 📦 Installation

### Basic Installation (All Platforms)

```bash
pip install deeplightrag
```

### GPU Acceleration (NVIDIA CUDA)

```bash
pip install deeplightrag[gpu]
```

### macOS Apple Silicon

```bash
pip install deeplightrag[macos]
```

### Advanced Features

```bash
# All optional features
pip install deeplightrag[all]

# Just web interface
pip install deeplightrag[web]

# Advanced relation extraction
pip install deeplightrag[advanced-re]
```

## 🎯 Quick Start

### Python API

```python
from deeplightrag import DeepLightRAG

# Initialize (auto-detects GPU!)
rag = DeepLightRAG()

# Index a document
results = rag.index_document("document.pdf")

# Query
answer = rag.query("What is this document about?")
print(answer['answer'])
```

### Command Line

```bash
# Index a document
deeplightrag index document.pdf

# Query
deeplightrag query "What are the key findings?"

# Show system info
deeplightrag info
```

## 💻 System Requirements

- **Python**: 3.9 or higher
- **OS**: Windows, Linux, macOS
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: Optional but recommended (NVIDIA CUDA or Apple Silicon)

## 🔧 Configuration

The system automatically optimizes for your hardware:

- **GPU (CUDA)**: Uses larger models, FP16 precision, batch processing
- **macOS (M1/M2)**: Uses MLX framework for Apple Silicon optimization
- **CPU**: Uses quantized models for efficiency

### Custom Configuration

```python
config = {
    "ocr": {
        "device": "cuda",  # or "cpu", "mps"
        "batch_size": 4,
        "resolution": "large"
    },
    "ner": {
        "gliner": {
            "batch_size": 16,
            "confidence_threshold": 0.3
        }
    }
}

rag = DeepLightRAG(config=config)
```

## 📊 Performance

| Document Size | CPU Time | GPU Time | Speedup |
|--------------|----------|----------|---------|
| 5 pages | 120-180s | 30-45s | **4x** |
| 15 pages | 400-600s | 90-120s | **4-5x** |
| 30 pages | 800-1200s | 180-240s | **4-5x** |

## 🎓 Examples

### Basic Document Processing

```python
from deeplightrag import DeepLightRAG

rag = DeepLightRAG(storage_dir="./my_data")

# Index
results = rag.index_document(
    "research_paper.pdf",
    document_id="paper_001"
)

print(f"Entities: {results['graph_stats']['entity_nodes']}")
print(f"Relationships: {results['graph_stats']['relationships']}")
print(f"Compression: {results['compression_ratio_str']}")

# Query
answer = rag.query("What is the research question?")
print(answer['answer'])
```

### Batch Processing

```python
from pathlib import Path

rag = DeepLightRAG()
pdf_files = Path("documents/").glob("*.pdf")

for pdf in pdf_files:
    print(f"Processing: {pdf.name}")
    results = rag.index_document(str(pdf))
    
    # Clean GPU memory between documents
    rag.cleanup_gpu_memory()
```

### Advanced Queries

```python
# Simple factual query
rag.query("What is the main topic?")

# Complex analytical query
rag.query("What methodology was used and why?")

# Multi-document query
rag.query("Compare findings across all indexed documents")
```

## 🔍 Key Components

### 1. Vision-Text Compression
- DeepSeek-OCR for multimodal understanding
- PCA compression for visual embeddings
- 80-90% token reduction

### 2. Dual-Layer Graph
- **Visual-Spatial Layer**: WHERE things are
- **Entity-Relationship Layer**: WHAT things mean
- Cross-layer connections for context

### 3. Adaptive Retrieval
- Query classification (5 levels)
- Dynamic token budgeting (2K-12K)
- Visual-aware context selection

### 4. LLM Integration
- Gemini (primary)
- OpenAI (optional)
- Anthropic (optional)
- LLM fallback for edge cases

## 🌟 Use Cases

- **Research Papers**: Extract entities, relationships, findings
- **Financial Documents**: Analyze reports, statements, filings
- **Legal Documents**: Process contracts, agreements, cases
- **Technical Manuals**: Index specifications, procedures
- **Academic Literature**: Build knowledge graphs from papers

## 📚 Documentation

- **Full Documentation**: [GitHub Repository](https://github.com/png261/DeepLightRag)
- **API Reference**: See source code docstrings
- **Examples**: Check `examples/` directory in repository

## 🛠️ Troubleshooting

### GPU Not Detected

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
```

### Out of Memory

```python
# Reduce batch sizes
config = {
    "ocr": {"batch_size": 2},
    "ner": {"gliner": {"batch_size": 8}}
}
rag = DeepLightRAG(config=config)
```

### Slow Performance

```bash
# On GPU systems, install CUDA support
pip install deeplightrag[gpu]

# On macOS, install MLX
pip install deeplightrag[macos]
```

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/png261/DeepLightRag/blob/main/CONTRIBUTING.md)

## 📄 License

MIT License - see [LICENSE](https://github.com/png261/DeepLightRag/blob/main/LICENSE) file

## 🙏 Credits

- DeepSeek OCR for vision-language understanding
- GLiNER for named entity recognition
- Gemini for LLM generation
- Transformers & PyTorch for ML infrastructure

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/png261/DeepLightRag/issues)
- **Email**: nhphuong.code@gmail.com
- **Documentation**: [Full Docs](https://github.com/png261/DeepLightRag)

---

**Made with ❤️ by the DeepLightRAG Team**

Transform your documents into knowledge graphs with the power of vision-text compression and GPU acceleration!