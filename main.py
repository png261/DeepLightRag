#!/usr/bin/env python3
"""
DeepLightRAG CLI Interface
Main entry point for the system
"""

import argparse
import sys
from pathlib import Path
import yaml
import json

from src.deeplightrag import DeepLightRAG


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with automatic Kaggle detection"""
    import os

    # Auto-detect Kaggle environment and use appropriate config
    is_kaggle = (
        os.path.exists("/kaggle")
        or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "/kaggle/" in os.getcwd()
    )

    if is_kaggle and Path("config_kaggle.yaml").exists():
        config_path = "config_kaggle.yaml"
        print(f"🔍 Detected Kaggle environment, using {config_path}")

    if Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Auto-configure GPU settings
        try:
            import torch

            if torch.cuda.is_available():
                print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
                # Update config for GPU usage
                if "ocr" in config:
                    config["ocr"]["device"] = "cuda"
                if "ner" in config:
                    config["ner"]["device"] = "cuda"
                if "relation_extraction" in config:
                    config["relation_extraction"]["device"] = "cuda"
        except ImportError:
            pass

        return config
    return {}


def cmd_index(args, rag_system):
    """Index a PDF document"""
    print(f"Indexing document: {args.pdf}")
    results = rag_system.index_document(
        args.pdf, document_id=args.doc_id, save_to_disk=not args.no_save
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


def cmd_query(args, rag_system):
    """Retrieve context for a query"""
    if args.load_doc:
        rag_system.load_document(args.load_doc)

    result = rag_system.retrieve(args.question, override_level=args.level)

    print(f"\n{'='*60}")
    print("RETRIEVED CONTEXT")
    print(f"{'='*60}")
    print(f"\n{result.get('context', 'No context')}\n")
    print(f"{'='*60}")
    print(f"Query Level: {result.get('level_name', 'N/A')} (Level {result.get('query_level', 'N/A')})")
    print(f"Tokens Used: {result.get('tokens_used', 'N/A')}/{result.get('token_budget', 'N/A')}")
    print(f"Entities Found: {result.get('entities_found', 'N/A')}")
    print(f"Nodes Retrieved: {result.get('nodes_retrieved', 'N/A')}")
    print(f"Retrieval Time: {result.get('retrieval_time', 'N/A')}")
    print(f"\n💡 Use this context with your own LLM for generation!")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return result


def cmd_interactive(args, rag_system):
    """Interactive retrieval mode"""
    if args.load_doc:
        rag_system.load_document(args.load_doc)

    print("\n" + "=" * 60)
    print("DeepLightRAG Interactive Retrieval Mode")
    print("=" * 60)
    print("Type your questions (or 'quit' to exit)")
    print("Commands: !stats, !level N")
    print("💡 Context retrieved - use with your own LLM for generation")
    print()

    override_level = None

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Handle commands
            if question.startswith("!"):
                if question == "!stats":
                    stats = rag_system.get_statistics()
                    print(json.dumps(stats, indent=2, default=str))
                elif question.startswith("!level"):
                    parts = question.split()
                    if len(parts) == 2:
                        override_level = int(parts[1])
                        print(f"Query level override set to {override_level}")
                    else:
                        override_level = None
                        print("Query level override cleared")
                else:
                    print("Unknown command")
                continue

            # Regular retrieval
            result = rag_system.retrieve(question, override_level=override_level)

            print(f"\n{'='*60}")
            print("CONTEXT")
            print(f"{'='*60}")
            print(f"{result.get('context', 'No context')[:500]}...")
            print(f"{'='*60}")
            print(
                f"Level: {result.get('level_name', 'N/A')} ({result.get('query_level', 'N/A')}) | "
                f"Tokens: {result.get('tokens_used', 'N/A')}/{result.get('token_budget', 'N/A')} | "
                f"Entities: {result.get('entities_found', 'N/A')}"
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="DeepLightRAG: Efficient Document-based RAG System"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument(
        "--storage", default="./deeplightrag_data", help="Storage directory for graphs and indices"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a PDF document")
    index_parser.add_argument("pdf", help="Path to PDF file")
    index_parser.add_argument("--doc-id", help="Document identifier")
    index_parser.add_argument("--no-save", action="store_true", help="Don't save to disk")
    index_parser.add_argument("--output", help="Save results to JSON file")

    # Query command
    query_parser = subparsers.add_parser("query", help="Retrieve context for a query")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--load-doc", help="Load specific document")
    query_parser.add_argument("--level", type=int, help="Override query level (1-5)")
    query_parser.add_argument("--output", help="Save results to JSON file")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive retrieval mode")
    interactive_parser.add_argument("--load-doc", help="Load specific document")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Initialize system
    rag_system = DeepLightRAG(config=config, storage_dir=args.storage)

    # Execute command
    if args.command == "index":
        cmd_index(args, rag_system)
    elif args.command == "query":
        cmd_query(args, rag_system)
    elif args.command == "interactive":
        cmd_interactive(args, rag_system)


if __name__ == "__main__":
    main()
