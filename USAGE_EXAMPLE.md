# DeepLightRAG Usage Examples

## Overview

DeepLightRAG focuses **ONLY** on:
1. **Indexing**: Convert documents into knowledge graphs
2. **Retrieval**: Get relevant context for queries

**You provide your own LLM** for final answer generation.

## Quick Start

### Installation

```bash
# Basic installation (CPU)
pip install deeplightrag

# With GPU support
pip install deeplightrag[gpu]

# With MLX (macOS)
pip install deeplightrag[macos]
```

### Basic Usage

```python
from deeplightrag import DeepLightRAG

# 1. Initialize system (auto-detects GPU)
rag = DeepLightRAG(storage_dir="./my_docs")

# 2. Index a document
results = rag.index_document("research_paper.pdf")
print(f"Indexed {results['total_pages']} pages")
print(f"Found {results['graph_stats']['entity_nodes']} entities")

# 3. Retrieve context for a query
retrieval = rag.retrieve("What is the main research question?")

# 4. Use the context with YOUR LLM
context = retrieval['context']
question = retrieval['question']

# Now send to your LLM:
# - OpenAI: client.chat.completions.create(messages=[...])
# - Anthropic: client.messages.create(messages=[...])
# - Gemini: model.generate_content(...)
# - Local: ollama, llama.cpp, etc.
```

## Use with Different LLMs

### 1. OpenAI

```python
from deeplightrag import DeepLightRAG
from openai import OpenAI

# Initialize
rag = DeepLightRAG()
client = OpenAI()

# Index document
rag.index_document("document.pdf")

# Retrieve context
retrieval = rag.retrieve("What are the key findings?")

# Generate with OpenAI
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Answer based on the provided context."},
        {"role": "user", "content": f"Context:\n{retrieval['context']}\n\nQuestion: {retrieval['question']}"}
    ]
)

answer = response.choices[0].message.content
print(answer)
```

### 2. Anthropic Claude

```python
from deeplightrag import DeepLightRAG
from anthropic import Anthropic

# Initialize
rag = DeepLightRAG()
client = Anthropic()

# Index and retrieve
rag.index_document("document.pdf")
retrieval = rag.retrieve("Summarize the methodology")

# Generate with Claude
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": f"Context:\n{retrieval['context']}\n\nQuestion: {retrieval['question']}"
        }
    ]
)

answer = message.content[0].text
print(answer)
```

### 3. Google Gemini

```python
from deeplightrag import DeepLightRAG
import google.generativeai as genai

# Initialize
rag = DeepLightRAG()
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-pro')

# Index and retrieve
rag.index_document("document.pdf")
retrieval = rag.retrieve("What are the conclusions?")

# Generate with Gemini
prompt = f"Context:\n{retrieval['context']}\n\nQuestion: {retrieval['question']}"
response = model.generate_content(prompt)

answer = response.text
print(answer)
```

### 4. Local LLMs (Ollama)

```python
from deeplightrag import DeepLightRAG
import requests

# Initialize
rag = DeepLightRAG()

# Index and retrieve
rag.index_document("document.pdf")
retrieval = rag.retrieve("Explain the approach")

# Generate with Ollama
response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'llama2',
    'prompt': f"Context:\n{retrieval['context']}\n\nQuestion: {retrieval['question']}",
    'stream': False
})

answer = response.json()['response']
print(answer)
```

### 5. Hugging Face Models

```python
from deeplightrag import DeepLightRAG
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize
rag = DeepLightRAG()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Index and retrieve
rag.index_document("document.pdf")
retrieval = rag.retrieve("What is discussed?")

# Generate with HuggingFace
prompt = f"Context:\n{retrieval['context']}\n\nQuestion: {retrieval['question']}"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
answer = tokenizer.decode(outputs[0])

print(answer)
```

## Advanced Usage

### Batch Retrieval

```python
# Retrieve context for multiple questions
questions = [
    "What is the main topic?",
    "What methodology was used?",
    "What are the results?"
]

results = rag.batch_retrieve(questions)

# Process each with your LLM
for result in results:
    context = result['context']
    question = result['question']
    # Send to your LLM...
```

### Custom Configuration

```python
# Configure for GPU
config = {
    "ocr": {
        "device": "cuda",
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

rag = DeepLightRAG(config=config, storage_dir="./data")
```

### Query Classification & Adaptive Retrieval

```python
# System automatically classifies queries and adjusts token budget

# Simple query (Level 1) → ~2K tokens
result = rag.retrieve("What is the title?")
print(f"Token budget: {result['token_budget']}")

# Complex query (Level 5) → ~12K tokens
result = rag.retrieve("Compare the methodology with related work and explain implications")
print(f"Token budget: {result['token_budget']}")
```

### Visual Context

```python
# Retrieve visual embeddings (for multimodal LLMs)
retrieval = rag.retrieve("Describe the diagram on page 3")

context = retrieval['context']  # Text context
visual = retrieval['visual_embeddings']  # Visual embeddings

# Use with multimodal LLM (e.g., GPT-4V, Gemini Vision)
```

## Statistics & Monitoring

```python
# Get system stats
stats = rag.get_statistics()
print(f"Documents indexed: {stats['system_stats']['documents_indexed']}")
print(f"Queries processed: {stats['system_stats']['queries_processed']}")
print(f"Entities in graph: {stats['graph_stats']['entity_relationship']['entity_count']}")

# Clean up GPU memory (if using GPU)
rag.cleanup_gpu_memory()
```

## Command Line Interface

```bash
# Index a document
deeplightrag index document.pdf --output ./my_data

# Retrieve context for a query
deeplightrag retrieve "What is this about?" --storage ./my_data

# Show system info
deeplightrag info
```

## Why Separate Indexing & Retrieval?

✅ **Flexibility**: Use any LLM (commercial, local, open-source)  
✅ **Cost Control**: You manage LLM costs and usage  
✅ **Performance**: Optimize retrieval separately from generation  
✅ **Privacy**: Keep generation in your own infrastructure  
✅ **Experimentation**: Test different LLMs without re-indexing  

## What You Get

### Retrieval Result
```python
{
    'question': 'Your question',
    'context': '...retrieved context text...',  # Use this with your LLM!
    'visual_embeddings': [...],  # Optional visual context
    'query_level': 3,
    'level_name': 'Detailed',
    'strategy': 'entity_centric',
    'tokens_used': 5234,
    'token_budget': 6000,
    'nodes_retrieved': 42,
    'entities_found': 15,
    'regions_accessed': 8,
    'retrieval_time': '1.23s'
}
```

### How to Use
```python
# Extract what you need
context = result['context']
question = result['question']

# Send to your LLM
answer = your_llm.generate(
    prompt=f"Context: {context}\n\nQuestion: {question}"
)
```

## Best Practices

1. **Index once, query many**: Indexing is slow, retrieval is fast
2. **Use GPU**: 4-5x faster indexing with GPU
3. **Batch queries**: Process multiple questions efficiently
4. **Monitor token usage**: Adjust query levels if needed
5. **Clean GPU memory**: Call `cleanup_gpu_memory()` between documents

## Performance

| Document Size | CPU Time | GPU Time |
|--------------|----------|----------|
| 5 pages | 2-3 min | 30-45 sec |
| 15 pages | 7-10 min | 1.5-2 min |
| 30 pages | 13-20 min | 3-4 min |

Retrieval: <2 seconds regardless of document size

## Support

- **Documentation**: Full docs in repository
- **Issues**: GitHub Issues
- **Examples**: Check `examples/` directory
- **Email**: nhphuong.code@gmail.com

---

**DeepLightRAG**: Powerful indexing & retrieval. Your choice of LLM. 🚀