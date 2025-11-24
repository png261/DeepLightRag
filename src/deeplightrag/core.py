"""
DeepLightRAG: Main Pipeline Orchestrator
Combines all components for end-to-end document QA
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .graph.dual_layer import DualLayerGraph
from .llm.base import LLMConfig
from .llm.factory import LLMFactory
from .ocr.deepseek_ocr import DeepSeekOCR
from .ocr.processor import PDFProcessor
from .retrieval.adaptive_retriever import AdaptiveRetriever
from .retrieval.query_classifier import QueryClassifier

# Setup logging
logger = logging.getLogger(__name__)


class DeepLightRAG:
    """
    DeepLightRAG System

    Efficient Document-based RAG with:
    - DeepSeek-OCR for vision-text compression
    - Dual-Layer Graph (Visual-Spatial + Entity-Relationship)
    - Adaptive Token Budgeting
    - DeepSeek R1 for LLM generation
    """

    def __init__(self, config: Optional[Dict] = None, storage_dir: str = "./deeplightrag_data"):
        """
        Initialize DeepLightRAG

        Args:
            config: Configuration dictionary
            storage_dir: Directory for storing graphs and indices
        """
        self.config = config or self._default_config()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        print("=" * 60)
        print("  DeepLightRAG System")
        print("  Efficient Document-based RAG with Vision-Text Compression")
        print("=" * 60)

        self._init_ocr()
        self._init_graph()
        self._init_retriever()
        self._init_llm()

        # Statistics
        self.stats = {
            "documents_indexed": 0,
            "queries_processed": 0,
            "total_pages": 0,
            "total_tokens_saved": 0,
        }

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "ocr": {
                "model_name": "mlx-community/DeepSeek-OCR-4bit",
                "quantization": "4bit",
                "resolution": "base",
            },
            "llm": {
                "provider": "deepseek",  # deepseek, openai, anthropic, huggingface, ollama, litellm
                "model": "mlx-community/deepseek-r1-distill-qwen-1.5b",
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9,
                "timeout": 30,
                "retry_attempts": 3,
                "api_key": None,  # Set via environment variables or config
                "base_url": None,  # For local models like Ollama
            },
            "retrieval": {"enable_adaptive": True, "default_level": 2},
        }

    def _init_ocr(self):
        """Initialize OCR components"""
        print("\n[1/4] Initializing DeepSeek-OCR...")
        ocr_config = self.config.get("ocr", {})
        self.ocr_model = DeepSeekOCR(
            model_name=ocr_config.get("model_name", "mlx-community/DeepSeek-OCR-4bit"),
            quantization=ocr_config.get("quantization", "4bit"),
            resolution=ocr_config.get("resolution", "base"),
        )
        self.pdf_processor = PDFProcessor(self.ocr_model)
        print("  DeepSeek-OCR initialized")

    def _init_graph(self):
        """Initialize graph components"""
        print("\n[2/4] Initializing Dual-Layer Graph...")
        self.dual_layer_graph = DualLayerGraph()
        print("  Dual-Layer Graph initialized")

    def _init_retriever(self):
        """Initialize retrieval components"""
        print("\n[3/4] Initializing Adaptive Retriever...")
        self.query_classifier = QueryClassifier()
        self.retriever = AdaptiveRetriever(self.dual_layer_graph, self.query_classifier)
        print("  Adaptive Retriever initialized")

    def _init_llm(self):
        """Initialize LLM components"""
        print("\n[4/4] Initializing LLM...")
        llm_config_dict = self.config.get("llm", {})

        try:
            # Create LLMConfig from dictionary
            llm_config = LLMConfig(
                provider=llm_config_dict.get("provider", "deepseek"),
                model=llm_config_dict.get("model", "mlx-community/deepseek-r1-distill-qwen-1.5b"),
                api_key=llm_config_dict.get("api_key"),
                temperature=llm_config_dict.get("temperature", 0.7),
                max_tokens=llm_config_dict.get("max_tokens", 2048),
                top_p=llm_config_dict.get("top_p", 0.9),
                top_k=llm_config_dict.get("top_k", 40),
                frequency_penalty=llm_config_dict.get("frequency_penalty", 0.0),
                presence_penalty=llm_config_dict.get("presence_penalty", 0.0),
                base_url=llm_config_dict.get("base_url"),
                timeout=llm_config_dict.get("timeout", 30),
                retry_attempts=llm_config_dict.get("retry_attempts", 3),
            )

            # Create LLM instance using factory
            self.llm = LLMFactory.from_config(llm_config)
            print("  LLM initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        print("\nSystem Ready!")

    def index_document(
        self, pdf_path: str, document_id: Optional[str] = None, save_to_disk: bool = True
    ) -> Dict[str, Any]:
        """
        Index a PDF document

        Args:
            pdf_path: Path to PDF file
            document_id: Optional document identifier
            save_to_disk: Save graph to disk

        Returns:
            Indexing statistics

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is invalid or empty
        """
        print("\n" + "=" * 60)
        print(f"INDEXING DOCUMENT: {pdf_path}")
        print("=" * 60)

        start_time = time.time()

        # Validate input
        if not pdf_path:
            raise ValueError("pdf_path cannot be empty")

        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError(f"File is not a PDF: {pdf_path}")

        # Get file size
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")

        logger.info(f"Starting indexing of {pdf_path} (size: {file_size / 1024:.1f}KB)")

        if document_id is None:
            document_id = Path(pdf_path).stem

        try:
            # Phase 1: OCR Processing
            print("\n[PHASE 1] PDF to Visual Tokens...")
            logger.info("Starting OCR processing")
            ocr_results = self.pdf_processor.process_pdf(pdf_path)

            if not ocr_results:
                raise ValueError(f"No text extracted from PDF: {pdf_path}")

            logger.info(f"OCR complete: {len(ocr_results)} pages processed")

            # Phase 2: Graph Construction
            print("\n[PHASE 2] Building Dual-Layer Graph...")
            logger.info("Starting graph construction")
            self.dual_layer_graph.build_from_ocr_results(ocr_results)
            logger.info("Graph construction complete")

            # Phase 3: Save to disk
            if save_to_disk:
                print("\n[PHASE 3] Saving to disk...")
                logger.info(f"Saving graph to {self.storage_dir}")
                doc_dir = self.storage_dir / document_id
                doc_dir.mkdir(parents=True, exist_ok=True)

                try:
                    self.dual_layer_graph.save(str(doc_dir))
                    logger.info(f"Graph saved to {doc_dir}")
                except Exception as e:
                    logger.error(f"Failed to save graph: {e}")
                    raise

                # Save OCR results
                try:
                    self.pdf_processor.save_results(ocr_results, str(doc_dir / "ocr_results.json"))
                    logger.info("OCR results saved")
                except Exception as e:
                    logger.error(f"Failed to save OCR results: {e}")
                    raise

            # Calculate statistics
            total_pages = len(ocr_results)
            total_tokens = sum(r.total_tokens for r in ocr_results)
            estimated_original = total_pages * 2500
            compression_ratio = estimated_original / total_tokens if total_tokens > 0 else 0

            indexing_time = time.time() - start_time

            # Update global stats
            self.stats["documents_indexed"] += 1
            self.stats["total_pages"] += total_pages
            self.stats["total_tokens_saved"] += estimated_original - total_tokens

            results = {
                "document_id": document_id,
                "pdf_path": pdf_path,
                "num_pages": total_pages,
                "total_tokens": total_tokens,
                "estimated_original_tokens": estimated_original,
                "compression_ratio": compression_ratio,
                "compression_ratio_str": f"{compression_ratio:.1f}x",
                "tokens_saved": estimated_original - total_tokens,
                "indexing_time": indexing_time,
                "indexing_time_str": f"{indexing_time:.2f}s",
                "time_per_page": indexing_time / total_pages if total_pages else 0,
                "time_per_page_str": f"{(indexing_time / total_pages) if total_pages else 0:.2f}s",
                "graph_stats": {
                    "visual_nodes": len(self.dual_layer_graph.visual_spatial.nodes),
                    "entity_nodes": len(self.dual_layer_graph.entity_relationship.entities),
                    "relationships": len(self.dual_layer_graph.entity_relationship.relationships),
                },
                "status": "success",
            }

            logger.info(f"Indexing complete: {results}")

            print("\n" + "=" * 60)
            print("INDEXING COMPLETE")
            print("=" * 60)
            print(f"Time: {results['indexing_time_str']}")
            print(f"Compression: {results['compression_ratio_str']}")
            print(f"Tokens Saved: {results['tokens_saved']:,}")
            print(f"Pages: {total_pages}")
            print(f"Entities: {results['graph_stats']['entity_nodes']}")
            print(f"Visual Regions: {results['graph_stats']['visual_nodes']}")

            return results

        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)
            print(f"\nERROR: Indexing failed - {e}")
            raise

    def query(
        self, question: str, enable_reasoning: bool = False, override_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query the indexed document

        Args:
            question: User question
            enable_reasoning: Enable chain-of-thought reasoning
            override_level: Override automatic query level classification

        Returns:
            Query results with answer and metadata
        """
        print("\n" + "-" * 60)
        print(f"QUERY: {question}")
        print("-" * 60)

        start_time = time.time()

        # Classify query
        classification = self.query_classifier.analyze_query(question)
        print(f"\nQuery Level: {classification['level']} ({classification['level_name']})")
        print(f"Token Budget: {classification['max_tokens']}")
        print(f"Strategy: {classification['strategy']}")

        # Retrieve context
        print("\n[Retrieving Context]...")
        retrieval_result = self.retriever.retrieve(question, override_level)
        context = retrieval_result["context"]

        print(f"Retrieved {retrieval_result['nodes_retrieved']} nodes")
        print(f"Token count: ~{retrieval_result['token_count']}")

        # Generate answer
        print("\n[Generating Answer]...")
        if enable_reasoning:
            llm_result = self.llm.generate_with_reasoning(context, question)
            answer = llm_result["answer"]
            reasoning = llm_result.get("reasoning", "")
        else:
            answer = self.llm.generate(context, question)
            reasoning = ""

        query_time = time.time() - start_time

        # Update stats
        self.stats["queries_processed"] += 1

        result = {
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "query_level": classification["level"],
            "strategy": classification["strategy"],
            "tokens_used": retrieval_result["token_count"],
            "tokens_vs_lightrag": f"{retrieval_result['token_count']} vs 30,000 (fixed)",
            "token_savings": f"{((30000 - retrieval_result['token_count']) / 30000 * 100):.1f}%",
            "nodes_retrieved": retrieval_result["nodes_retrieved"],
            "query_time": f"{query_time:.2f}s",
            "entities_found": len(retrieval_result.get("entities", [])),
            "regions_accessed": len(retrieval_result.get("regions", [])),
        }

        print("\n" + "-" * 60)
        print("ANSWER:")
        print("-" * 60)
        print(answer)
        print(f"\n[Query completed in {result['query_time']}]")
        print(f"[Token savings: {result['token_savings']}]")

        return result

    def batch_query(
        self, questions: List[str], enable_reasoning: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries

        Args:
            questions: List of questions
            enable_reasoning: Enable reasoning for all queries

        Returns:
            List of query results
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}/{len(questions)}")
            result = self.query(question, enable_reasoning)
            results.append(result)

        return results

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            "system_stats": self.stats,
            "graph_stats": {
                "visual_spatial": self.dual_layer_graph.visual_spatial.get_statistics(),
                "entity_relationship": self.dual_layer_graph.entity_relationship.get_statistics(),
            },
            "retrieval_stats": self.retriever.get_retrieval_stats(),
            "llm_info": self.llm.get_model_info(),
        }

    def save_state(self, path: Optional[str] = None):
        """Save system state"""
        if path is None:
            path = str(self.storage_dir / "system_state.json")

        state = {"config": self.config, "stats": self.stats, "storage_dir": str(self.storage_dir)}

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        print(f"System state saved to {path}")

    def load_document(self, document_id: str):
        """Load a previously indexed document"""
        doc_dir = self.storage_dir / document_id

        if not doc_dir.exists():
            raise FileNotFoundError(f"Document {document_id} not found")

        print(f"Loading document: {document_id}")
        self.dual_layer_graph.load(str(doc_dir))

        # Reinitialize retriever with loaded graph
        self.retriever = AdaptiveRetriever(self.dual_layer_graph, self.query_classifier)

        print(f"Document {document_id} loaded successfully")

    def compare_with_lightrag(self, query: str) -> Dict:
        """
        Compare performance with LightRAG baseline

        Returns metrics comparison
        """
        # Our approach
        our_result = self.query(query)

        comparison = {
            "query": query,
            "visual_graph_rag": {
                "tokens_used": our_result["tokens_used"],
                "query_time": our_result["query_time"],
                "strategy": our_result["strategy"],
                "nodes_retrieved": our_result["nodes_retrieved"],
            },
            "lightrag_baseline": {
                "tokens_used": 30000,  # Fixed
                "estimated_time": "3-5s",
                "strategy": "fixed_chunking",
            },
            "improvements": {
                "token_reduction": f"{((30000 - our_result['tokens_used']) / 30000 * 100):.1f}%",
                "estimated_cost_savings": f"{((30000 - our_result['tokens_used']) / 30000 * 100):.1f}%",
            },
        }

        return comparison

    def build_graph_from_ocr(self, ocr_results: List) -> "GraphWrapper":
        """
        Build dual-layer graph from OCR results

        Args:
            ocr_results: List of PageOCRResult from DeepSeek-OCR

        Returns:
            GraphWrapper object with retriever and summary methods
        """
        print("\n[GRAPH] Building Dual-Layer Graph from OCR results...")
        self.dual_layer_graph.build_from_ocr_results(ocr_results)
        return GraphWrapper(self)

    def load_graph(self, graph_path: str):
        """
        Load a graph from a file path

        Args:
            graph_path: Path to the graph directory or JSON file
        """
        # Handle different path formats
        if graph_path.endswith(".json"):
            # If path is to a specific JSON file, get the directory
            graph_dir = os.path.dirname(graph_path)
        else:
            graph_dir = graph_path

        if not os.path.exists(graph_dir):
            raise FileNotFoundError(f"Graph directory not found: {graph_dir}")

        print(f"Loading graph from: {graph_dir}")
        self.dual_layer_graph.load(graph_dir)

        # Reinitialize retriever with loaded graph
        self.retriever = AdaptiveRetriever(self.dual_layer_graph, self.query_classifier)

        print(f"Graph loaded successfully from {graph_dir}")


class GraphWrapper:
    """
    Wrapper for DualLayerGraph providing simplified API for testing
    """

    def __init__(self, rag_system: DeepLightRAG):
        self.rag_system = rag_system
        self.graph = rag_system.dual_layer_graph
        self.retriever = rag_system.retriever

    def get_retriever(self) -> "RetrieverWrapper":
        """Get a retriever object for this graph"""
        return RetrieverWrapper(self.rag_system)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the graph"""
        vs_stats = self.graph.visual_spatial.get_statistics()
        er_stats = self.graph.entity_relationship.get_statistics()

        return {
            "visual_spatial": {
                "nodes": vs_stats["num_nodes"],
                "edges": vs_stats["num_edges"],
                "pages": vs_stats["num_pages"],
                "avg_degree": vs_stats["avg_degree"],
            },
            "entity_relationship": {
                "entities": er_stats["num_entities"],
                "relationships": er_stats["num_relationships"],
                "avg_degree": er_stats["avg_entity_degree"],
            },
            "cross_layer": {
                "entity_to_region_mappings": len(self.graph.entity_to_regions),
                "region_to_entity_mappings": len(self.graph.region_to_entities),
                "figure_caption_links": len(self.graph.figure_caption_links),
            },
            "total_compressed_tokens": sum(
                node.region.token_count for node in self.graph.visual_spatial.nodes.values()
            ),
        }


class RetrieverWrapper:
    """
    Wrapper for AdaptiveRetriever providing simplified retrieve method
    """

    def __init__(self, rag_system: DeepLightRAG):
        self.rag_system = rag_system
        self.retriever = rag_system.retriever

    def retrieve(self, query: str) -> str:
        """
        Retrieve context for a query

        Args:
            query: User query string

        Returns:
            Context string for LLM generation
        """
        result = self.retriever.retrieve(query)
        return result["context"]
