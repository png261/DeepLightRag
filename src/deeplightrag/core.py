"""
DeepLightRAG: Document Indexing and Retrieval System
Focus: High-performance indexing and retrieval (NO generation)
Use with any LLM of your choice for generation
"""

import json
import logging
import os
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from .graph.dual_layer import DualLayerGraph
# from .ocr.deepseek_ocr import DeepSeekOCR  # Refactored to use factory
from .ocr.processor import PDFProcessor
from .retrieval.adaptive_retriever import AdaptiveRetriever
from .retrieval.query_classifier import QueryClassifier

# Setup logging
logger = logging.getLogger(__name__)


from .utils.config_manager import config_manager
from .utils.component_factory import component_factory

class DeepLightRAG:
    def __init__(self, config: Optional[Dict] = None, storage_dir: str = "./deeplightrag_data"):
        self.config = config or config_manager.get_default_config()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect GPU and optimize configuration
        self.device = config_manager.setup_gpu_optimization(self.config)
        self.gpu_name = self.device  # simplified for now, or fetch from device info if needed

        print("=" * 60)
        print("  DeepLightRAG: Indexing & Retrieval System")
        print("  High-performance indexing with vision-text compression")
        print("  Use with ANY LLM for generation")
        if self.device == "cuda":
            print(f"  🎮 GPU Acceleration: {self.device}")
        elif self.device == "mps":
             print(f"  🚀 Apple Silicon Acceleration: {self.device}")
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

    def _init_ocr(self):
        """Initialize OCR components with GPU support via Factory"""
        device_info = f" on {self.device}" if hasattr(self, "device") else ""
        print(f"\n[1/3] Initializing OCR/VLM Processor{device_info}...")
        
        # Use Factory to create the configured OCR processor
        self.ocr_model = component_factory.create_ocr_processor(self.config)
        self.pdf_processor = PDFProcessor(self.ocr_model)

        ocr_config = self.config.get("ocr", {})
        visual_status = (
            "enabled" if ocr_config.get(
                "enable_visual_embeddings", True) else "disabled"
        )
        print(
            f"  ✅ DeepSeek-OCR initialized (Visual embeddings: {visual_status})")

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

        # Try to load existing graph data from storage directory
        graph_data_path = self.storage_dir / "basic-text"
        if graph_data_path.exists():
            print(f"  Loading existing graph from {graph_data_path}")
            try:
                self.dual_layer_graph.load(str(graph_data_path))
                print(f"  ✅ Dual-Layer Graph loaded from storage")
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to load existing graph: {e}")
                print(f"  ✅ Dual-Layer Graph initialized (empty)")
        else:
            print("  ✅ Dual-Layer Graph initialized (empty)")

    def _init_retriever(self):
        """Initialize retrieval components with visual awareness"""
        print("\n[3/3] Initializing Visual-Aware Adaptive Retriever...")
        retrieval_config = self.config.get("retrieval", {})

        # Get model storage configuration
        models_config = self.config.get("models", {}).get("query_classifier", {})
        save_path = models_config.get("save_path", "query_classifier_model")
        
        # Check if we should use absolute path based on storage_dir
        import os
        if not os.path.isabs(save_path):
            # Determine base directory: storage_dir or config-defined base
            storage_conf = self.config.get("storage", {})
            base_dir = self.storage_dir or storage_conf.get("base_dir", "./deep_light_rag_data")
            model_dir = storage_conf.get("model_dir", "models")
            save_path = os.path.join(base_dir, model_dir, save_path)

        self.query_classifier = QueryClassifier(
            model_save_path=save_path,
            device=getattr(self, "device", "cpu")
        )

        # Use visual-aware retriever if visual embeddings are enabled
        if self.config.get("ocr", {}).get("enable_visual_embeddings", True):
            from .retrieval.visual_aware_retriever import VisualAwareRetriever

            self.retriever = VisualAwareRetriever(
                self.dual_layer_graph,
                self.query_classifier,
                visual_weight=retrieval_config.get("visual_weight", 0.3),
                config=self.config
            )
            print("  ✅ Visual-Aware Retriever initialized")
        else:
            self.retriever = AdaptiveRetriever(
                self.dual_layer_graph, self.query_classifier, config=self.config)
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
                    self.pdf_processor.save_results(
                        ocr_results, str(doc_dir / "ocr_results.json"))
                    logger.info("OCR results saved")
                except Exception as e:
                    logger.error(f"Failed to save OCR results: {e}")
                    raise

            # Calculate statistics
            total_pages = len(ocr_results)
            total_tokens = sum(r.total_tokens for r in ocr_results)
            estimated_original = total_pages * 2500
            compression_ratio = estimated_original / \
                total_tokens if total_tokens > 0 else 0

            indexing_time = time.time() - start_time

            # Update global stats
            self.stats["documents_indexed"] += 1
            self.stats["total_pages"] += total_pages
            self.stats["total_tokens_saved"] += estimated_original - \
                total_tokens

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

            print(f"Retrieved {retrieval_result.nodes_retrieved} nodes (Visual mode: {visual_mode})")
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
        # Safety check: ensure retrieval_result is not a string or other unexpected type
        if isinstance(retrieval_result, str):
            print(f"ERROR: retrieval_result is a string: {retrieval_result[:100]}")
            retrieval_result = {
                "context": retrieval_result,
                "entities": [],
                "regions": [],
                "token_count": len(retrieval_result) // 4,
                "nodes_retrieved": 0,
            }
        
        if hasattr(retrieval_result, "token_count"):
            # VisualRetrievalResult object
            tokens_used = retrieval_result.token_count
            nodes_retrieved = retrieval_result.nodes_retrieved
            entities_found = len(getattr(retrieval_result, "entities", []))
            regions_accessed = len(getattr(retrieval_result, "regions", []))
        else:
            # Dictionary result
            tokens_used = retrieval_result.get("token_count", 0)
            nodes_retrieved = retrieval_result.get("nodes_retrieved", 0)
            entities_found = len(retrieval_result.get("entities", []))
            regions_accessed = len(retrieval_result.get("regions", []))

        # Extract entities and regions from result
        if hasattr(retrieval_result, "entities"):
            entities_list = [e.to_dict() if hasattr(e, 'to_dict') else e for e in retrieval_result.entities]
            regions_list = [r.to_dict() if hasattr(r, 'to_dict') else r for r in retrieval_result.regions]
        else:
            entities_list = retrieval_result.get("entities", [])
            regions_list = retrieval_result.get("regions", [])
        
        result = {
            "question": question,
            "context": context,  # Raw markdown context
            "visual_embeddings": visual_embeddings,  # Optional visual context
            "entities": entities_list,  # Retrieved entities
            "regions": regions_list,  # Retrieved regions
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
        
        # Format with TOON for better LLM consumption
        from .retrieval.toon_formatter import format_toon_context
        result["formatted_context"] = format_toon_context(result, simple=True)

        print("\n" + "-" * 60)
        print("CONTEXT RETRIEVED (Ready for your LLM)")
        print("-" * 60)
        print(f"Raw context: {len(context)} characters")
        print(f"Formatted (TOON): {len(result['formatted_context'])} characters")
        print(f"Entities: {entities_found}")
        print(f"Visual regions: {regions_accessed}")
        print(f"Visual embeddings: {len(visual_embeddings)}")
        print(f"\n[Retrieval completed in {result['retrieval_time']}]")
        print(f"[Use 'formatted_context' field (TOON format) with your LLM]")

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

        state = {"config": self.config, "stats": self.stats,
                 "storage_dir": str(self.storage_dir)}

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

        # Reinitialize retriever with loaded graph (preserve visual awareness)
        retrieval_config = self.config.get("retrieval", {})
        if self.config.get("ocr", {}).get("enable_visual_embeddings", True):
            from .retrieval.visual_aware_retriever import VisualAwareRetriever
            
            self.retriever = VisualAwareRetriever(
                self.dual_layer_graph,
                self.query_classifier,
                visual_weight=retrieval_config.get("visual_weight", 0.3),
            )
        else:
            self.retriever = AdaptiveRetriever(
                self.dual_layer_graph, self.query_classifier)

        print(f"Document {document_id} loaded successfully")

    def extract_entities(self, text: str, visual_regions: List = None) -> List[Dict]:
        """
        Extract entities using GLiNER with optional visual context.

        Args:
            text: Input text to extract entities from
            visual_regions: Optional list of VisualRegion objects for visual context

        Returns:
            List of extracted entities with metadata
        """
        try:
            # Initialize GLiNER model
            from .ner.working_gliner_ner import GLiNERNERExtractor

            ner_extractor = GLiNERNERExtractor(
                model_name=self.config.get("ner", {}).get("model", "urchade/gliner_base-v2.1"),
                device=getattr(self, "device", "cpu"),
                confidence_threshold=self.config.get("ner", {}).get("threshold", 0.3)
            )

            # Extract entities
            if visual_regions:
                # Use visual context if available
                entities = ner_extractor.extract_entities_with_visual_context(text, visual_regions)
            else:
                # Standard entity extraction
                entities = ner_extractor.extract_entities(text)

            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def extract_relations(self, entities: List[Dict], text: str, visual_regions: List = None) -> List[Dict]:
        """
        Extract relations using GLiREL with optional visual context.

        Args:
            entities: List of extracted entities
            text: Original text for context
            visual_regions: Optional list of VisualRegion objects for layout context

        Returns:
            List of extracted relations with metadata
        """
        try:
            # Initialize GLiREL model
            from .ner.working_enhanced_ner import EnhancedNERPipeline

            relation_extractor = EnhancedNERPipeline(
                model_name=self.config.get("relation_extraction", {}).get("model", "knowledgator/GLiREL"),
                device=getattr(self, "device", "cpu"),
                confidence_threshold=self.config.get("relation_extraction", {}).get("threshold", 0.3)
            )

            # Extract relations
            if visual_regions:
                # Use visual/layout context if available
                relations = relation_extractor.extract_relations_with_layout(
                    entities, text, visual_regions
                )
            else:
                # Standard relation extraction
                relations = relation_extractor.extract_relations(entities, text)

            return relations

        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []

    def process_text_document(self, text: str, document_id: str = None, save_to_disk: bool = True) -> Dict:
        """
        Process a text document (not PDF) through the full pipeline.

        Args:
            text: Input text to process
            document_id: Optional document identifier
            save_to_disk: Whether to save results to disk

        Returns:
            Processing results with statistics
        """
        if document_id is None:
            document_id = f"text_doc_{int(time.time())}"

        try:
            # Phase 1: Create VisualRegions from text
            print("\n[PHASE 1] Converting text to Visual Regions...")
            logger.info("Creating visual regions from text")

            # Convert text to VisualRegions with mock spatial layout
            visual_regions = self._text_to_visual_regions(text)

            # Phase 2: Entity Extraction
            print("\n[PHASE 2] Extracting Entities...")
            logger.info("Extracting entities with GLiNER")
            entities = self.extract_entities(text, visual_regions)

            # Phase 3: Relation Extraction
            print("\n[PHASE 3] Extracting Relations...")
            logger.info("Extracting relations with GLiREL")
            relations = self.extract_relations(entities, text, visual_regions)

            # Phase 4: Graph Construction
            print("\n[PHASE 4] Building Graph...")
            logger.info("Building dual-layer graph")

            # Create OCR-like results for graph builder
            ocr_result = self._visual_regions_to_ocr_results(visual_regions)

            # Build graph from results
            self.dual_layer_graph.build_from_ocr_results([ocr_result])

            # Phase 5: Save to disk
            if save_to_disk:
                print("\n[PHASE 5] Saving to disk...")
                logger.info(f"Saving to {self.storage_dir}")
                doc_dir = self.storage_dir / document_id
                doc_dir.mkdir(parents=True, exist_ok=True)

                # Save graph
                self.dual_layer_graph.save(str(doc_dir))

                # Save processing results
                results = {
                    "document_id": document_id,
                    "text_length": len(text),
                    "entities": entities,
                    "relations": relations,
                    "visual_regions": len(visual_regions),
                    "processing_time": time.time()
                }

                import json
                with open(doc_dir / "processing_results.json", 'w') as f:
                    json.dump(results, f, indent=2)

            # Update statistics
            self.stats["documents_indexed"] += 1

            return {
                "document_id": document_id,
                "entities_found": len(entities),
                "relations_found": len(relations),
                "visual_regions": len(visual_regions),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                "document_id": document_id,
                "status": "failed",
                "error": str(e)
            }

    def _text_to_visual_regions(self, text: str) -> List:
        """Convert text to VisualRegions with spatial layout."""
        from .ocr.deepseek_ocr import VisualRegion, BoundingBox, VisualToken

        regions = []
        lines = text.split('\n')

        y_position = 50
        for line_num, line in enumerate(lines):
            if not line.strip():
                y_position += 30
                continue

            # Determine block type
            if line.strip().startswith('# '):
                block_type = "header"
                x1, x2 = 50, 600
            elif line.strip().startswith('## '):
                block_type = "subheader"
                x1, x2 = 70, 580
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                block_type = "list"
                x1, x2 = 100, 600
            else:
                block_type = "paragraph"
                x1, x2 = 70, 700

            y1 = y_position
            y2 = y1 + 25

            # Create VisualRegion with comprehensive features
            region = VisualRegion(
                region_id=f"region_{line_num}",
                page_num=1,
                block_type=block_type,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                compressed_tokens=self._create_tokens_from_text(line),
                text_content=line.strip(),
                markdown_content=line.strip(),
                token_count=len(line.split()) // 3,
                confidence=0.9,

                # Enhanced visual features
                region_embedding=self._create_embedding(line),
                visual_complexity=0.5 + (len(line) / 200),
                spatial_features={
                    "position_weight": 0.8,
                    "center_distance": abs(400 - (x1 + x2) / 2),
                    "page_position": y1 / 1000
                },
                layout_features={
                    "reading_order": line_num,
                    "text_block_type": block_type,
                    "is_title": block_type == "header"
                },
                quality_metrics={
                    "clarity_score": 0.9,
                    "contrast_score": 0.85
                }
            )

            regions.append(region)
            y_position = y2 + 15

        return regions

    def _create_tokens_from_text(self, text: str) -> List:
        """Create visual tokens from text."""
        from .ocr.deepseek_ocr import VisualToken
        import numpy as np

        words = text.split()
        tokens = []

        for i, word in enumerate(words):
            # Create simplified embedding
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            token = VisualToken(
                token_id=f"token_{i}",
                embedding=embedding,
                confidence=0.9,
                region_type="text"
            )
            tokens.append(token)

        return tokens

    def _create_embedding(self, text: str) -> np.ndarray:
        """Create a simple text embedding."""
        import numpy as np

        # Simple hash-based embedding (in real implementation, use proper model)
        embedding = np.zeros(768, dtype=np.float32)

        # Use text hash to create deterministic embedding
        text_hash = hash(text.strip())
        for i in range(768):
            embedding[i] = ((text_hash >> i) & 0xFF) / 255.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _visual_regions_to_ocr_results(self, visual_regions) -> "PageOCRResult":
        """Convert VisualRegions to PageOCRResult format."""
        from .ocr.deepseek_ocr import PageOCRResult

        return PageOCRResult(
            page_num=0,
            width=1000,
            height=1000,
            regions=visual_regions,
            total_tokens=sum(len(r.compressed_tokens) for r in visual_regions) if visual_regions else 0,
            processing_time=0.1
        )
