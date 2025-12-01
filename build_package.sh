#!/bin/bash
# Build script for DeepLightRAG package

set -e

echo "🔧 Building DeepLightRAG Package"
echo "================================"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info src/*.egg-info

# Install build tools
echo "📦 Installing build tools..."
pip install --upgrade build twine wheel

# Build the package
echo "🏗️  Building package..."
python -m build

# Check the package
echo "✅ Checking package..."
twine check dist/*

echo ""
echo "✨ Build complete!"
echo "📦 Packages created in dist/"
ls -lh dist/

echo ""
echo "To upload to PyPI:"
echo "  twine upload dist/*"
echo ""
echo "To test locally:"
echo "  pip install dist/deeplightrag-*.whl"
