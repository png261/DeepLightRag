# Installation Guide

## System Requirements

- **Python**: 3.9, 3.10, 3.11, 3.12, or 3.13
- **OS**: Windows 10+, Linux (Ubuntu 20.04+), macOS 11+
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 5GB for models (downloaded on first run)
- **GPU**: Optional but recommended

## Quick Install

```bash
pip install deeplightrag
```

## Platform-Specific Installation

### Windows (with GPU)

```bash
# Install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install DeepLightRAG
pip install deeplightrag[gpu]
```

### Linux (with GPU)

```bash
# CUDA should be installed on your system
# Install DeepLightRAG with GPU support
pip install deeplightrag[gpu]
```

### macOS (Apple Silicon)

```bash
# Basic installation
pip install deeplightrag

# With MLX optimization (recommended for M1/M2/M3)
pip install deeplightrag[macos]
```

### CPU Only (Any Platform)

```bash
pip install deeplightrag
```

## Installation Options

### Basic (Core Features)
```bash
pip install deeplightrag
```

Includes:
- PDF processing
- Entity/relation extraction
- Graph building
- LLM integration (Gemini)
- Basic retrieval

### With GPU Support
```bash
pip install deeplightrag[gpu]
```

Additional features:
- CUDA optimization
- FP16 precision
- Larger batch sizes
- 3-5x faster processing

### macOS Optimized
```bash
pip install deeplightrag[macos]
```

Additional features:
- MLX framework
- Apple Silicon optimization
- Metal GPU acceleration
- 4-bit quantization

### Advanced Relation Extraction
```bash
pip install deeplightrag[advanced-re]
```

Additional features:
- OpenNRE models
- Enhanced relationship detection
- Better accuracy

### Web Interface
```bash
pip install deeplightrag[web]
```

Additional features:
- Streamlit interface
- Interactive visualizations
- Result dashboard

### All Features
```bash
pip install deeplightrag[all]
```

## Development Installation

```bash
# Clone repository
git clone https://github.com/png261/DeepLightRag.git
cd DeepLightRag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Verification

### Check Installation

```bash
deeplightrag info
```

Expected output:
```
DeepLightRAG System Information
==================================================
Version: 1.0.0
Platform: Linux 5.15.0
Python: 3.10.12

GPU (CUDA): ✅ Available
  Device: NVIDIA Tesla T4
  Memory: 15.0GB

Dependencies:
  ✅ Transformers
  ✅ GLiNER
  ✅ Gemini
  ✅ Sentence Transformers
  ✅ Pillow
  ✅ PyMuPDF
```

### Test Basic Functionality

```python
from deeplightrag import DeepLightRAG

# Initialize
rag = DeepLightRAG()
print(f"Running on: {rag.device}")

# Should print: cuda, mps, or cpu
```

## Common Issues

### Issue 1: PyTorch Not Found

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
pip install torch torchvision
```

### Issue 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch sizes
config = {
    "ocr": {"batch_size": 2},
    "ner": {"gliner": {"batch_size": 8}}
}
rag = DeepLightRAG(config=config)
```

### Issue 3: MLX Not Available (macOS)

**Error:**
```
RuntimeError: MLX is only available on macOS
```

**Solution:**
- On macOS: `pip install deeplightrag[macos]`
- On other platforms: This is expected, MLX is macOS-only

### Issue 4: Import Error

**Error:**
```
ImportError: cannot import name 'DeepLightRAG'
```

**Solution:**
```bash
# Reinstall
pip uninstall deeplightrag
pip install --no-cache-dir deeplightrag
```

### Issue 5: Model Download Fails

**Error:**
```
ConnectionError: Unable to download model
```

**Solution:**
```bash
# Set HuggingFace cache directory
export HF_HOME="/path/to/cache"

# Or use offline mode if models are cached
export TRANSFORMERS_OFFLINE=1
```

## Platform-Specific Notes

### Windows

- Install Visual C++ Redistributable if you encounter DLL errors
- CUDA Toolkit must be installed for GPU support
- Use PowerShell or Command Prompt, not WSL

### Linux

- Requires `libGL` for OpenCV: `sudo apt-get install libgl1`
- CUDA drivers must match PyTorch version
- For headless servers, set `MPLBACKEND=Agg`

### macOS

- MLX only works on Apple Silicon (M1/M2/M3)
- Intel Macs use CPU mode
- Xcode Command Line Tools required: `xcode-select --install`

## Docker Installation

```bash
# Build Docker image
docker build -t deeplightrag .

# Run container
docker run -it --gpus all deeplightrag

# With volume mount
docker run -it --gpus all -v $(pwd)/data:/data deeplightrag
```

## Virtual Environment (Recommended)

### Using venv

```bash
python -m venv deeplightrag-env
source deeplightrag-env/bin/activate  # Linux/macOS
# or
deeplightrag-env\Scripts\activate  # Windows

pip install deeplightrag
```

### Using conda

```bash
conda create -n deeplightrag python=3.10
conda activate deeplightrag
pip install deeplightrag
```

## Upgrading

```bash
# Upgrade to latest version
pip install --upgrade deeplightrag

# Upgrade with specific extras
pip install --upgrade deeplightrag[gpu]

# Force reinstall
pip install --force-reinstall --no-cache-dir deeplightrag
```

## Uninstallation

```bash
pip uninstall deeplightrag

# Clean up cache
rm -rf ~/.cache/huggingface
rm -rf ~/.deeplightrag
```

## Environment Variables

```bash
# Gemini API Key (for LLM fallback)
export GEMINI_API_KEY="your-api-key"

# HuggingFace cache
export HF_HOME="/path/to/cache"

# Disable MLX on macOS (use PyTorch instead)
export USE_MLX=0

# Offline mode
export TRANSFORMERS_OFFLINE=1
```

## Next Steps

After installation:

1. **Test**: Run `deeplightrag info` to verify setup
2. **Quick Start**: Try the examples in README
3. **Documentation**: Read full docs on GitHub
4. **Community**: Join discussions and report issues

## Support

- **Issues**: [GitHub Issues](https://github.com/png261/DeepLightRag/issues)
- **Documentation**: [Full Docs](https://github.com/png261/DeepLightRag)
- **Email**: nhphuong.code@gmail.com

---

Happy processing! 🚀