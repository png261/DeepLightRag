"""
DeepLightRAG: Document Indexing and Retrieval System
Focus: High-performance indexing and retrieval (NO generation)
Use with any LLM of your choice for generation
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .graph.dual_layer import DualLayerGraph
from .ocr.deepseek_ocr import DeepSeekOCR
from .ocr.processor import PDFProcessor
from .retrieval.adaptive_retriever import AdaptiveRetriever
from .retrieval.query_classifier import QueryClassifier

# Setup logging
logger = logging.getLogger(__name__)


class DeepLightRAG:
    def __init__(self, config: Optional[Dict] = None, storage_dir: str = "./deeplightrag_data"):
        self.config = config or self._default_config()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect GPU and optimize configuration
        self._setup_gpu_optimization()

        print("=" * 60)
        print("  DeepLightRAG: Indexing & Retrieval System")
        print("  High-performance indexing with vision-text compression")
        print("  Use with ANY LLM for generation")
        if hasattr(self, "device") and self.device == "cuda":
            print(f"  🎮 GPU Acceleration: {self.gpu_name}")
        print("=" * 60)

        self._init_ocr()
        self._init_graph()
        self._init_retriever()

        self.stats = {
            "documents_indexed": 0,
            "queries_processed": 0,
            "total_pages": 0,
            "total_tokens_saved": 0,
        }

    def cleanup_gpu_memory(self):
        """Clean up GPU memory if available"""
        if hasattr(self, "device") and self.device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
                print("🧹 GPU memory cleaned")
            except Exception as e:
                print(f"⚠️ GPU cleanup warning: {e}")

    def _setup_gpu_optimization(self):
        """Auto-detect and configure GPU settings"""
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.device == "cuda":
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

                # Auto-configure for GPU
                if "ocr" in self.config:
                    self.config["ocr"]["device"] = "cuda"
                    self.config["ocr"]["torch_dtype"] = "float16"
                if "ner" in self.config:
                    self.config["ner"]["device"] = "cuda"
                if "relation_extraction" in self.config:
                    self.config["relation_extraction"]["device"] = "cuda"

        except ImportError:
            self.device = "cpu"

    def _default_config(self) -> Dict:
        """Default configuration with auto GPU detection"""
        import torch
        import platform

        is_macos = platform.system() == "Darwin"
        
        # Auto-detect optimal model based on hardware
        if torch.cuda.is_available():
            # GPU configuration - better models
            ocr_config = {
                "model_name": "deepseek-ai/deepseek-ocr",
                "quantization": "none",
                "resolution": "large",
                "device": "cuda",
                "torch_dtype": "float16",
                "batch_size": 4,
                "enable_visual_embeddings": True,
                "embedding_compression": "pca",
                "target_embedding_dim": 512,
            }
            ner_config = {
                "primary_model": "gliner",
                "device": "cuda",
                "gliner": {
                    "model_name": "urchade/gliner_large-v2.1",
                    "confidence_threshold": 0.3,
                    "device": "cuda",
                    "torch_dtype": "float16",
                    "batch_size": 16,
                },
                "bert": {
                    "model_name": "microsoft/deberta-v3-large",
                    "confidence_threshold": 0.4,
                    "device": "cuda",
                    "torch_dtype": "float16",
                },
            }
            re_config = {
                "primary_model": "deberta",
                "device": "cuda",
                "deberta": {
                    "model_name": "microsoft/deberta-v3-large",
                    "confidence_threshold": 0.4,
                    "device": "cuda",
                    "torch_dtype": "float16",
                    "max_length": 768,
                    "batch_size": 8,
                },
            }
        else:
            # CPU configuration - use MLX on macOS, transformers elsewhere
            if is_macos:
                # macOS: Use MLX 4-bit quantized model
                ocr_config = {
                    "model_name": "mlx-community/DeepSeek-OCR-4bit",
                    "quantization": "4bit",
                    "resolution": "base",
                    "device": "cpu",
                    "enable_visual_embeddings": True,
                    "embedding_compression": "pca",
                    "target_embedding_dim": 256,
                }
            else:
                # Other platforms: Use transformers CPU mode
                ocr_config = {
                    "model_name": "deepseek-ai/deepseek-ocr",
                    "quantization": "8bit",  # CPU-friendly quantization
                    "resolution": "base",
                    "device": "cpu",
                    "enable_visual_embeddings": True,
                    "embedding_compression": "pca",
                    "target_embedding_dim": 256,
                }
            ner_config = {
                "primary_model": "gliner",
                "device": "cpu",
                "gliner": {
                    "model_name": "urchade/gliner_base",
                    "confidence_threshold": 0.4,
                    "device": "cpu",
                },
            }
            re_config = {
                "primary_model": "deberta",
                "device": "cpu",
                "deberta": {
                    "model_name": "microsoft/deberta-v3-base",
                    "confidence_threshold": 0.5,
                    "device": "cpu",
                },
            }

        return {
            "ocr": ocr_config,
            "ner": ner_config,
            "relation_extraction": re_config,
            "retrieval": {
                "enable_adaptive": True,
                "default_level": 2,
                "visual_weight": 0.3,
                "enable_visual_fallback": True,
            },
        }

    def _init_ocr(self):
        """Initialize OCR components with GPU support"""
        device_info = f" on {self.device}" if hasattr(self, "device") else ""
        print(f"\n[1/3] Initializing DeepSeek-OCR{device_info}...")
        ocr_config = self.config.get("ocr", {})

        # Pass all GPU-related parameters
        init_kwargs = {
            "model_name": ocr_config.get("model_name", "mlx-community/DeepSeek-OCR-4bit"),
            "quantization": ocr_config.get("quantization", "4bit"),
            "resolution": ocr_config.get("resolution", "base"),
            "enable_visual_embeddings": ocr_config.get("enable_visual_embeddings", True),
            "embedding_compression": ocr_config.get("embedding_compression", "pca"),
            "target_embedding_dim": ocr_config.get("target_embedding_dim", 256),
        }

        # Add GPU parameters if available
        if "device" in ocr_config:
            init_kwargs["device"] = ocr_config["device"]
        if "torch_dtype" in ocr_config:
            import torch

            init_kwargs["torch_dtype"] = (
                torch.float16 if ocr_config["torch_dtype"] == "float16" else torch.float32
            )
        if "batch_size" in ocr_config:
            init_kwargs["batch_size"] = ocr_config["batch_size"]

        self.ocr_model = DeepSeekOCR(**init_kwargs)
        self.pdf_processor = PDFProcessor(self.ocr_model)

        visual_status = (
            "enabled" if ocr_config.get("enable_visual_embeddings", True) else "disabled"
        )
        print(f"  ✅ DeepSeek-OCR initialized (Visual embeddings: {visual_status})")

    def _init_graph(self):
        """Initialize graph components with GPU awareness"""
        device_info = f" on {self.device}" if hasattr(self, "device") else ""
        print(f"\n[2/3] Initializing Dual-Layer Graph{device_info}...")

        # Pass device and configuration to graph
        graph_kwargs = {
            "device": getattr(self, "device", "cpu"),
            "ner_config": self.config.get("ner", {}),
            "re_config": self.config.get("relation_extraction", {}),
        }

        if hasattr(self, "device") and self.device == "cuda":
            graph_kwargs["enable_gpu_acceleration"] = True

        self.dual_layer_graph = DualLayerGraph(**graph_kwargs)
        print("  ✅ Dual-Layer Graph initialized")

    def _init_retriever(self):
        """Initialize retrieval components with visual awareness"""
        print("\n[3/3] Initializing Visual-Aware Adaptive Retriever...")
        retrieval_config = self.config.get("retrieval", {})

        self.query_classifier = QueryClassifier()

        # Use visual-aware retriever if visual embeddings are enabled
        if self.config.get("ocr", {}).get("enable_visual_embeddings", True):
            from .retrieval.visual_aware_retriever import VisualAwareRetriever

            self.retriever = VisualAwareRetriever(
                self.dual_layer_graph,
                self.query_classifier,
                visual_weight=retrieval_config.get("visual_weight", 0.3),
                enable_visual_fallback=retrieval_config.get("enable_visual_fallback", True),
            )
            print("  ✅ Visual-Aware Retriever initialized")
        else:
            self.retriever = AdaptiveRetriever(self.dual_layer_graph, self.query_classifier)
            print("  ✅ Traditional Adaptive Retriever initialized")
        
        print("\n🚀 System Ready! (Indexing & Retrieval Only)")



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

    def retrieve(
        self, question: str, override_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query (NO generation)
        
        Use the returned context with ANY LLM of your choice for generation.

        Args:
            question: User question
            override_level: Override automatic query level classification

        Returns:
            Retrieval results with context and metadata
        """
        print("\n" + "-" * 60)
        print(f"RETRIEVAL: {question}")
        print("-" * 60)

        start_time = time.time()

        # Classify query
        classification = self.query_classifier.analyze_query(question)
        print(f"\nQuery Level: {classification['level']} ({classification['level_name']})")
        print(f"Token Budget: {classification['max_tokens']}")
        print(f"Strategy: {classification['strategy']}")

        # Retrieve context with visual awareness
        print("\n[Retrieving Context]...")
        retrieval_result = self.retriever.retrieve(question, override_level)

        # Check if visual-aware retrieval was used
        is_visual_retrieval = hasattr(retrieval_result, "visual_mode_used")

        if is_visual_retrieval:
            context = retrieval_result.context
            visual_embeddings = retrieval_result.visual_context
            visual_mode = retrieval_result.visual_mode_used

            print(
                f"Retrieved {retrieval_result.nodes_retrieved} nodes (Visual mode: {visual_mode})"
            )
            print(f"Token count: ~{retrieval_result.token_count}")
            if visual_embeddings:
                print(f"Visual embeddings: {len(visual_embeddings)}")
        else:
            # Traditional retrieval result
            context = retrieval_result["context"]
            visual_embeddings = []
            visual_mode = False

            print(f"Retrieved {retrieval_result['nodes_retrieved']} nodes")
            print(f"Token count: ~{retrieval_result['token_count']}")

        retrieval_time = time.time() - start_time

        # Update stats
        self.stats["queries_processed"] += 1

        # Handle both dictionary (from traditional retriever) and object (from visual retriever)
        if hasattr(retrieval_result, "token_count"):
            # VisualRetrievalResult object
            tokens_used = retrieval_result.token_count
            nodes_retrieved = retrieval_result.nodes_retrieved
            entities_found = len(getattr(retrieval_result, "entities", []))
            regions_accessed = len(getattr(retrieval_result, "regions", []))
        else:
            # Dictionary result
            tokens_used = retrieval_result["token_count"]
            nodes_retrieved = retrieval_result["nodes_retrieved"]
            entities_found = len(retrieval_result.get("entities", []))
            regions_accessed = len(retrieval_result.get("regions", []))

        result = {
            "question": question,
            "context": context,  # Use this with your LLM!
            "visual_embeddings": visual_embeddings,  # Optional visual context
            "query_level": classification["level"],
            "level_name": classification["level_name"],
            "strategy": classification["strategy"],
            "tokens_used": tokens_used,
            "token_budget": classification["max_tokens"],
            "nodes_retrieved": nodes_retrieved,
            "retrieval_time": f"{retrieval_time:.2f}s",
            "retrieval_time_seconds": retrieval_time,
            "entities_found": entities_found,
            "regions_accessed": regions_accessed,
            "visual_mode": visual_mode if is_visual_retrieval else False,
        }

        print("\n" + "-" * 60)
        print("CONTEXT RETRIEVED (Ready for your LLM)")
        print("-" * 60)
        print(f"Context length: {len(context)} characters")
        print(f"Entities: {entities_found}")
        print(f"Visual regions: {regions_accessed}")
        print(f"\n[Retrieval completed in {result['retrieval_time']}]")
        print(f"[Use 'context' field with your LLM for generation]")

        return result

    def batch_retrieve(
        self, questions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context for multiple queries

        Args:
            questions: List of questions

        Returns:
            List of retrieval results
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"Retrieval {i}/{len(questions)}")
            result = self.retrieve(question)
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
