#!/usr/bin/env python3
"""
DeepLightRAG Command Line Interface
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DeepLightRAG: Efficient Document-based RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a PDF document
  deeplightrag index document.pdf --id my_doc

  # Query an indexed document
  deeplightrag query "What is the main idea?" --doc my_doc

  # Interactive mode
  deeplightrag interactive --doc my_doc

  # Show system info
  deeplightrag info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a PDF document")
    index_parser.add_argument("pdf_path", help="Path to PDF file")
    index_parser.add_argument("--id", dest="doc_id", help="Document ID (default: filename)")
    index_parser.add_argument("--storage", default="./deeplightrag_data", help="Storage directory")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query an indexed document")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--doc", dest="doc_id", required=True, help="Document ID to query")
    query_parser.add_argument("--storage", default="./deeplightrag_data", help="Storage directory")
    query_parser.add_argument(
        "--level", type=int, choices=[1, 2, 3, 4], help="Override query level"
    )
    query_parser.add_argument("--reasoning", action="store_true", help="Enable reasoning mode")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive query mode")
    interactive_parser.add_argument("--doc", dest="doc_id", required=True, help="Document ID")
    interactive_parser.add_argument(
        "--storage", default="./deeplightrag_data", help="Storage directory"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.add_argument("--storage", default="./deeplightrag_data", help="Storage directory")

    # List command
    list_parser = subparsers.add_parser("list", help="List indexed documents")
    list_parser.add_argument("--storage", default="./deeplightrag_data", help="Storage directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_index(args):
    """Index a PDF document"""
    from . import DeepLightRAG

    print("Initializing DeepLightRAG...")
    rag = DeepLightRAG(storage_dir=args.storage)

    doc_id = args.doc_id or Path(args.pdf_path).stem
    print(f"\nIndexing {args.pdf_path} as '{doc_id}'...")

    result = rag.index_document(args.pdf_path, doc_id)

    print("\n" + "=" * 60)
    print("INDEXING SUMMARY")
    print("=" * 60)
    print(f"Document ID: {result['document_id']}")
    print(f"Pages: {result['num_pages']}")
    print(f"Compression: {result['compression_ratio_str']}")
    print(f"Tokens Saved: {result['tokens_saved']:,}")
    print(f"Time: {result['indexing_time_str']}")
    print(
        f"Graph Nodes: {result['graph_stats']['visual_nodes']} visual, {result['graph_stats']['entity_nodes']} entities"
    )


def cmd_query(args):
    """Query an indexed document"""
    from . import DeepLightRAG

    print("Initializing DeepLightRAG...")
    rag = DeepLightRAG(storage_dir=args.storage)

    print(f"Loading document: {args.doc_id}")
    rag.load_document(args.doc_id)

    result = rag.query(args.question, enable_reasoning=args.reasoning, override_level=args.level)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "-" * 60)
    print(f"Query Level: {result['query_level']}")
    print(f"Tokens Used: {result['tokens_used']}")
    print(f"Token Savings: {result['token_savings']}")


def cmd_interactive(args):
    """Interactive query mode"""
    from . import DeepLightRAG

    print("Initializing DeepLightRAG...")
    rag = DeepLightRAG(storage_dir=args.storage)

    print(f"Loading document: {args.doc_id}")
    rag.load_document(args.doc_id)

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Type your questions below. Type 'quit' or 'exit' to stop.")
    print("Type 'stats' to see query statistics.")
    print("-" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if question.lower() == "stats":
                stats = rag.get_statistics()
                print(json.dumps(stats["system_stats"], indent=2))
                continue

            result = rag.query(question)
            print(f"\nAnswer: {result['answer']}")
            print(
                f"\n[Level {result['query_level']}, {result['tokens_used']} tokens, {result['token_savings']} saved]"
            )

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def cmd_info(args):
    """Show system information"""
    from . import DeepLightRAG, __version__

    print("=" * 60)
    print("VISUAL-GRAPH RAG SYSTEM INFO")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Storage Directory: {args.storage}")

    storage = Path(args.storage)
    if storage.exists():
        docs = [
            d.name for d in storage.iterdir() if d.is_dir() and (d / "cross_layer.json").exists()
        ]
        print(f"Indexed Documents: {len(docs)}")
        for doc in docs:
            print(f"  - {doc}")
    else:
        print("Storage directory not found.")

    print("\nFeatures:")
    print("  - 9-10x Vision-Text Compression")
    print("  - Dual-Layer Graph Architecture")
    print("  - Adaptive Token Budgeting (2K-12K)")
    print("  - 60-80% Cost Savings vs LightRAG")


def cmd_list(args):
    """List indexed documents"""
    storage = Path(args.storage)

    if not storage.exists():
        print("No indexed documents found.")
        return

    docs = []
    for doc_dir in storage.iterdir():
        if doc_dir.is_dir() and (doc_dir / "cross_layer.json").exists():
            # Get stats
            try:
                with open(doc_dir / "cross_layer.json") as f:
                    data = json.load(f)
                entity_count = len(data.get("entity_to_regions", {}))
                region_count = len(data.get("region_to_entities", {}))
                docs.append({"id": doc_dir.name, "entities": entity_count, "regions": region_count})
            except:
                docs.append({"id": doc_dir.name, "entities": "?", "regions": "?"})

    if not docs:
        print("No indexed documents found.")
        return

    print("=" * 60)
    print("INDEXED DOCUMENTS")
    print("=" * 60)
    for doc in docs:
        print(f"  {doc['id']}")
        print(f"    Entities: {doc['entities']}, Regions: {doc['regions']}")


if __name__ == "__main__":
    main()
