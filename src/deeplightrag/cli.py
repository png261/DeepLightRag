"""
DeepLightRAG Command Line Interface
Document Indexing and Retrieval (NO generation - use with your own LLM)
"""

import sys
import argparse
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DeepLightRAG: Document Indexing & Retrieval (use with any LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="DeepLightRAG 1.0.0"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    index_parser.add_argument("--output", "-o", type=str, default="./deeplightrag_data",
                              help="Output directory for indexed data")
    index_parser.add_argument("--doc-id", type=str,
                              help="Document ID (default: filename)")
    index_parser.add_argument("--config", "-c", type=str,
                              help="Path to configuration file (YAML)")

    # Retrieve command
    retrieve_parser = subparsers.add_parser(
        "retrieve", help="Retrieve context for a query")
    retrieve_parser.add_argument(
        "question", type=str, help="Question to retrieve context for")
    retrieve_parser.add_argument("--storage", "-s", type=str, default="./deeplightrag_data",
                                 help="Storage directory with indexed data")
    retrieve_parser.add_argument("--config", "-c", type=str,
                                 help="Path to configuration file (YAML)")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.command == "index":
        index_document(args)
    elif args.command == "retrieve":
        retrieve_context(args)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()
        sys.exit(1)


def index_document(args):
    """Index a document"""
    try:
        from .core import DeepLightRAG
        from .utils.config_manager import config_manager

        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            print(f"Error: File not found: {pdf_path}")
            sys.exit(1)

        doc_id = args.doc_id or pdf_path.stem

        print(f"Indexing document: {pdf_path.name}")
        print(f"Output directory: {args.output}")
        print(f"Document ID: {doc_id}\n")

        # Load config if provided
        config = None
        if args.config:
            print(f"Loading config from: {args.config}")
            try:
                config = config_manager.load_config(args.config)
            except Exception as e:
                print(f"Error loading config: {e}")
                sys.exit(1)

        rag = DeepLightRAG(config=config, storage_dir=args.output)
        results = rag.index_document(
            str(pdf_path), document_id=doc_id, save_to_disk=True)

        print(f"\n✅ Indexing complete!")
        print(f"Time: {results.get('indexing_time_str')}")
        print(f"Pages: {results.get('total_pages')}")
        print(f"Entities: {results['graph_stats'].get('entity_nodes')}")
        print(f"Relationships: {results['graph_stats'].get('relationships')}")
        print(f"Tokens saved: {results.get('tokens_saved'):,}")

    except Exception as e:
        print(f"Error indexing document: {e}")
        sys.exit(1)


def retrieve_context(args):
    """Retrieve context for a query (NO generation)"""
    try:
        from .core import DeepLightRAG
        from .utils.config_manager import config_manager

        storage_path = Path(args.storage)
        if not storage_path.exists():
            print(f"Error: Storage directory not found: {storage_path}")
            print("Index a document first with: deeplightrag index <pdf_path>")
            sys.exit(1)

        print(f"Question: {args.question}\n")

        # Load config if provided
        config = None
        if args.config:
            print(f"Loading config from: {args.config}")
            try:
                config = config_manager.load_config(args.config)
            except Exception as e:
                print(f"Error loading config: {e}")
                sys.exit(1)

        rag = DeepLightRAG(config=config, storage_dir=args.storage)
        result = rag.retrieve(args.question)

        print(f"\n📄 Retrieved Context:")
        print("-" * 50)
        print(result['context'])
        print("-" * 50)
        print(f"\nRetrieval Stats:")
        print(f"  • Level: {result['level_name']}")
        print(f"  • Tokens used: {result['tokens_used']} / {result['token_budget']}")
        print(f"  • Entities found: {result['entities_found']}")
        print(f"  • Visual regions: {result['regions_accessed']}")
        print(f"  • Time: {result['retrieval_time']}")
        print(f"\n💡 Use this context with YOUR LLM for generation!")

    except Exception as e:
        print(f"Error retrieving context: {e}")
        sys.exit(1)


def show_info():
    """Show system information"""
    import platform

    print("DeepLightRAG System Information")
    print("=" * 50)
    print(f"Version: 1.0.0")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")

    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"\nGPU (CUDA): {'✅ Available' if gpu_available else '❌ Not available'}")

        if gpu_available:
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    except ImportError:
        print(f"\nGPU (CUDA): ⚠️ PyTorch not installed")

    # Check MLX availability (macOS)
    is_macos = platform.system() == "Darwin"
    if is_macos:
        try:
            import mlx.core as mx
            print(f"MLX (macOS): ✅ Available")
        except ImportError:
            print(f"MLX (macOS): ❌ Not installed (optional)")
            print(f"  Install with: pip install mlx mlx-lm")

    # Check dependencies
    print("\nCore Dependencies:")
    deps = [
        ("transformers", "Transformers"),
        ("gliner", "GLiNER"),
        ("sentence_transformers", "Sentence Transformers"),
        ("PIL", "Pillow"),
        ("fitz", "PyMuPDF"),
    ]

    for module, name in deps:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")


if __name__ == "__main__":
    main()
