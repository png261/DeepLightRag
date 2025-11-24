#!/usr/bin/env python3
"""
Visual-Graph RAG: CLI Interface
Main entry point for the system
"""

import argparse
import sys
from pathlib import Path
import yaml
import json

from src.visual_graph_rag import VisualGraphRAG


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def cmd_index(args, rag_system):
    """Index a PDF document"""
    print(f"Indexing document: {args.pdf}")
    results = rag_system.index_document(
        args.pdf,
        document_id=args.doc_id,
        save_to_disk=not args.no_save
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


def cmd_query(args, rag_system):
    """Query the indexed document"""
    if args.load_doc:
        rag_system.load_document(args.load_doc)

    result = rag_system.query(
        args.question,
        enable_reasoning=args.reasoning,
        override_level=args.level
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return result


def cmd_interactive(args, rag_system):
    """Interactive query mode"""
    if args.load_doc:
        rag_system.load_document(args.load_doc)

    print("\n" + "=" * 60)
    print("Interactive Query Mode")
    print("=" * 60)
    print("Type your questions (or 'quit' to exit)")
    print("Commands: !stats, !compare, !reasoning on/off, !level N")
    print()

    enable_reasoning = False
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
                elif question.startswith("!compare"):
                    query = question.replace("!compare", "").strip()
                    if query:
                        comparison = rag_system.compare_with_lightrag(query)
                        print(json.dumps(comparison, indent=2))
                elif question == "!reasoning on":
                    enable_reasoning = True
                    print("Chain-of-thought reasoning enabled")
                elif question == "!reasoning off":
                    enable_reasoning = False
                    print("Chain-of-thought reasoning disabled")
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

            # Regular query
            result = rag_system.query(
                question,
                enable_reasoning=enable_reasoning,
                override_level=override_level
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def cmd_demo(args, rag_system):
    """Run demo with sample queries"""
    if args.pdf:
        print("Indexing sample document...")
        rag_system.index_document(args.pdf)
    elif args.load_doc:
        rag_system.load_document(args.load_doc)
    else:
        print("Please provide --pdf or --load-doc for demo")
        return

    # Sample queries for each level
    demo_queries = [
        {
            "query": "What is the Q3 revenue?",
            "expected_level": 1,
            "description": "Simple factual query (Level 1)"
        },
        {
            "query": "How does revenue correlate with iPhone sales trends?",
            "expected_level": 2,
            "description": "Complex reasoning query (Level 2)"
        },
        {
            "query": "Compare the performance metrics across different products",
            "expected_level": 3,
            "description": "Multi-document synthesis (Level 3)"
        },
        {
            "query": "Explain the chart on page 1 showing revenue trends",
            "expected_level": 4,
            "description": "Visual-semantic fusion (Level 4)"
        }
    ]

    print("\n" + "=" * 60)
    print("VISUAL-GRAPH RAG DEMO")
    print("=" * 60)

    results = []
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"Demo Query {i}: {demo['description']}")
        print(f"{'='*60}")

        result = rag_system.query(demo["query"])
        results.append(result)

        print(f"\nExpected Level: {demo['expected_level']}")
        print(f"Actual Level: {result['query_level']}")

    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)

    total_tokens = sum(r["tokens_used"] for r in results)
    lightrag_tokens = 30000 * len(results)

    print(f"Total queries: {len(results)}")
    print(f"Total tokens used: {total_tokens:,}")
    print(f"LightRAG equivalent: {lightrag_tokens:,}")
    print(f"Token savings: {((lightrag_tokens - total_tokens) / lightrag_tokens * 100):.1f}%")

    avg_time = sum(float(r["query_time"].replace("s", "")) for r in results) / len(results)
    print(f"Average query time: {avg_time:.2f}s")


def cmd_stats(args, rag_system):
    """Show system statistics"""
    stats = rag_system.get_statistics()
    print(json.dumps(stats, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Visual-Graph RAG: Efficient Document-based RAG System"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--storage", default="./visual_graph_rag_data",
        help="Storage directory for graphs and indices"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a PDF document")
    index_parser.add_argument("pdf", help="Path to PDF file")
    index_parser.add_argument("--doc-id", help="Document identifier")
    index_parser.add_argument("--no-save", action="store_true", help="Don't save to disk")
    index_parser.add_argument("--output", help="Save results to JSON file")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the indexed document")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--load-doc", help="Load specific document")
    query_parser.add_argument("--reasoning", action="store_true", help="Enable reasoning")
    query_parser.add_argument("--level", type=int, help="Override query level (1-4)")
    query_parser.add_argument("--output", help="Save results to JSON file")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive query mode")
    interactive_parser.add_argument("--load-doc", help="Load specific document")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with sample queries")
    demo_parser.add_argument("--pdf", help="PDF file to index for demo")
    demo_parser.add_argument("--load-doc", help="Load previously indexed document")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Initialize system
    rag_system = VisualGraphRAG(config=config, storage_dir=args.storage)

    # Execute command
    if args.command == "index":
        cmd_index(args, rag_system)
    elif args.command == "query":
        cmd_query(args, rag_system)
    elif args.command == "interactive":
        cmd_interactive(args, rag_system)
    elif args.command == "demo":
        cmd_demo(args, rag_system)
    elif args.command == "stats":
        cmd_stats(args, rag_system)


if __name__ == "__main__":
    main()
