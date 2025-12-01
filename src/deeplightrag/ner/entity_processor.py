"""
Entity Processor for DeepLightRAG
High-level interface for entity processing and integration
"""

from typing import Dict, List, Optional, Any, Tuple
import time
from pathlib import Path

from .gliner_ner import GLiNERExtractor, ExtractedEntity
from .enhanced_ner_pipeline import EnhancedNERPipeline, NERProcessingResult
from ..ocr.deepseek_ocr import PageOCRResult, VisualRegion
from ..graph.entity_relationship import Entity, Relationship, EntityRelationshipGraph


class EntityProcessor:
    """
    High-level entity processor that integrates NER with DeepLightRAG
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_base",
        confidence_threshold: float = 0.3,
        enable_visual_grounding: bool = True,
        enable_coreference: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize Entity Processor

        Args:
            model_name: GLiNER model name
            confidence_threshold: Minimum confidence for entity extraction
            enable_visual_grounding: Whether to ground entities visually
            enable_coreference: Whether to resolve coreferences
            device: Device for inference
        """
        self.confidence_threshold = confidence_threshold
        self.enable_visual_grounding = enable_visual_grounding
        self.enable_coreference = enable_coreference

        # Initialize components
        print("Initializing Entity Processor...")

        self.gliner_extractor = GLiNERExtractor(
            model_name=model_name, confidence_threshold=confidence_threshold, device=device
        )

        self.ner_pipeline = EnhancedNERPipeline(
            gliner_extractor=self.gliner_extractor,
            enable_visual_grounding=enable_visual_grounding,
            enable_cross_region_coreference=enable_coreference,
            confidence_threshold=confidence_threshold,
        )

        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "processing_time_total": 0.0,
            "avg_entities_per_doc": 0.0,
            "entity_types_found": set(),
        }

        print("Entity Processor initialized successfully")

    def process_document(
        self,
        ocr_results: List[PageOCRResult],
        document_id: str = "unknown",
        entity_graph: Optional[EntityRelationshipGraph] = None,
    ) -> Dict[str, Any]:
        """
        Process a complete document for entity extraction

        Args:
            ocr_results: OCR results from document processing
            document_id: Document identifier
            entity_graph: Optional entity graph to populate

        Returns:
            Processing results with entities and relationships
        """
        start_time = time.time()

        print(f"\n🧠 Processing document '{document_id}' for entity extraction")
        print(f"   Pages: {len(ocr_results)}")

        # Collect all regions from all pages
        all_regions = []
        total_regions = 0

        for page_result in ocr_results:
            all_regions.extend(page_result.regions)
            total_regions += len(page_result.regions)

        print(f"   Total regions: {total_regions}")

        # Process regions for entity extraction
        ner_result = self.ner_pipeline.process_document_regions(
            regions=all_regions, document_id=document_id
        )

        # Convert to DeepLightRAG entities
        all_extracted_entities = []
        for region_entities in ner_result.entities_by_region.values():
            all_extracted_entities.extend(region_entities)

        deeplightrag_entities = self.ner_pipeline.convert_to_deeplightrag_entities(
            extracted_entities=all_extracted_entities, document_id=document_id
        )

        # Extract relationships
        relationships = self.ner_pipeline.extract_entity_relationships(
            entities=all_extracted_entities, regions=all_regions
        )

        # Add to entity graph if provided
        if entity_graph:
            self._add_to_entity_graph(deeplightrag_entities, relationships, entity_graph)

        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(
            num_entities=len(deeplightrag_entities),
            num_relationships=len(relationships),
            processing_time=processing_time,
            entity_types=[e.entity_type for e in deeplightrag_entities],
        )

        return {
            "document_id": document_id,
            "entities": deeplightrag_entities,
            "relationships": relationships,
            "ner_result": ner_result,
            "processing_time": processing_time,
            "summary": self._generate_processing_summary(
                ner_result, deeplightrag_entities, relationships
            ),
        }

    def process_text(
        self, text: str, text_id: str = "text_input", region_type: str = "paragraph"
    ) -> Dict[str, Any]:
        """
        Process raw text for entity extraction

        Args:
            text: Input text
            text_id: Text identifier
            region_type: Type of text region

        Returns:
            Processing results with entities
        """
        print(f"🧠 Processing text input for entity extraction")

        # Extract entities directly
        extracted_entities = self.gliner_extractor.extract_entities(
            text=text, region_type=region_type
        )

        # Convert to DeepLightRAG entities
        deeplightrag_entities = self.ner_pipeline.convert_to_deeplightrag_entities(
            extracted_entities=extracted_entities, document_id=text_id
        )

        return {
            "text_id": text_id,
            "entities": deeplightrag_entities,
            "extracted_entities": extracted_entities,
            "summary": f"Extracted {len(deeplightrag_entities)} entities from text",
        }

    def extract_entities_by_type(
        self, text: str, entity_types: List[str], region_type: str = "general"
    ) -> Dict[str, List[ExtractedEntity]]:
        """
        Extract specific entity types from text

        Args:
            text: Input text
            entity_types: List of entity types to extract
            region_type: Type of text region

        Returns:
            Dictionary mapping entity types to extracted entities
        """
        entities = self.gliner_extractor.extract_entities(
            text=text, entity_types=entity_types, region_type=region_type
        )

        # Group by entity type
        entities_by_type = {}
        for entity_type in entity_types:
            entities_by_type[entity_type] = [e for e in entities if e.label == entity_type]

        return entities_by_type

    def _add_to_entity_graph(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        entity_graph: EntityRelationshipGraph,
    ):
        """Add entities and relationships to the entity graph"""
        # Add entities
        for entity in entities:
            entity_graph.add_entity(entity)

        # Add relationships
        for relationship in relationships:
            entity_graph.add_relationship(relationship)

    def _update_stats(
        self,
        num_entities: int,
        num_relationships: int,
        processing_time: float,
        entity_types: List[str],
    ):
        """Update processing statistics"""
        self.stats["documents_processed"] += 1
        self.stats["total_entities"] += num_entities
        self.stats["total_relationships"] += num_relationships
        self.stats["processing_time_total"] += processing_time

        # Update average entities per document
        if self.stats["documents_processed"] > 0:
            self.stats["avg_entities_per_doc"] = (
                self.stats["total_entities"] / self.stats["documents_processed"]
            )

        # Update entity types found
        self.stats["entity_types_found"].update(entity_types)

    def _generate_processing_summary(
        self,
        ner_result: NERProcessingResult,
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> str:
        """Generate a processing summary"""
        entity_breakdown = "\n".join(
            [
                f"  {entity_type}: {count}"
                for entity_type, count in ner_result.entities_by_type.items()
            ]
        )

        return f"""
Entity Processing Summary:
========================
- Total entities: {len(entities)}
- Total relationships: {len(relationships)}
- Processing time: {ner_result.processing_time:.2f}s
- Average confidence: {ner_result.confidence_stats.get('mean', 0):.2f}
- Visual grounding: {ner_result.visual_grounding_success}/{ner_result.total_entities}

Entity Types:
{entity_breakdown}

Performance:
- Entities/second: {len(entities) / max(ner_result.processing_time, 0.1):.1f}
- Regions processed: {len(ner_result.entities_by_region)}
"""

    def get_supported_entity_types(self) -> Dict[str, Dict[str, Any]]:
        """Get supported entity types and their descriptions"""
        return self.gliner_extractor.get_supported_entities()

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats_copy = self.stats.copy()
        stats_copy["entity_types_found"] = list(self.stats["entity_types_found"])

        # Add derived statistics
        if self.stats["documents_processed"] > 0:
            stats_copy["avg_processing_time"] = (
                self.stats["processing_time_total"] / self.stats["documents_processed"]
            )
            stats_copy["avg_relationships_per_doc"] = (
                self.stats["total_relationships"] / self.stats["documents_processed"]
            )

        return stats_copy

    def validate_extraction_quality(
        self, entities: List[Entity], expected_entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate the quality of entity extraction

        Args:
            entities: List of extracted entities
            expected_entity_types: Expected entity types for validation

        Returns:
            Validation results and quality metrics
        """
        validation_results = {
            "total_entities": len(entities),
            "entity_types_found": list(set(e.entity_type for e in entities)),
            "confidence_distribution": {},
            "quality_score": 0.0,
            "issues": [],
        }

        if not entities:
            validation_results["issues"].append("No entities extracted")
            return validation_results

        # Analyze confidence distribution
        confidences = [e.confidence for e in entities]
        validation_results["confidence_distribution"] = {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "low_confidence_count": len([c for c in confidences if c < 0.5]),
        }

        # Check expected entity types
        if expected_entity_types:
            found_types = set(e.entity_type for e in entities)
            expected_types = set(expected_entity_types)
            missing_types = expected_types - found_types

            if missing_types:
                validation_results["issues"].append(
                    f"Missing expected entity types: {list(missing_types)}"
                )

        # Calculate quality score
        quality_score = 0.0

        # Confidence score (40%)
        avg_confidence = validation_results["confidence_distribution"]["mean"]
        quality_score += 0.4 * avg_confidence

        # Entity diversity (30%)
        unique_types = len(validation_results["entity_types_found"])
        max_expected_types = 12  # Based on our schema
        diversity_score = min(1.0, unique_types / max_expected_types)
        quality_score += 0.3 * diversity_score

        # Visual grounding (20%)
        grounded_entities = len([e for e in entities if e.grounding_boxes])
        grounding_score = grounded_entities / len(entities) if entities else 0
        quality_score += 0.2 * grounding_score

        # Issue penalty (10%)
        issue_penalty = len(validation_results["issues"]) * 0.1
        quality_score = max(0.0, quality_score - issue_penalty)

        validation_results["quality_score"] = quality_score

        return validation_results

    def benchmark_performance(self, test_texts: List[str]) -> Dict[str, Any]:
        """
        Benchmark NER performance on test texts

        Args:
            test_texts: List of test texts for benchmarking

        Returns:
            Benchmark results
        """
        print(f"🏃‍♂️ Benchmarking NER performance on {len(test_texts)} texts...")

        start_time = time.time()
        total_entities = 0
        total_chars = 0

        for text in test_texts:
            entities = self.gliner_extractor.extract_entities(text)
            total_entities += len(entities)
            total_chars += len(text)

        total_time = time.time() - start_time

        return {
            "total_texts": len(test_texts),
            "total_entities": total_entities,
            "total_characters": total_chars,
            "total_time": total_time,
            "texts_per_second": len(test_texts) / total_time,
            "entities_per_second": total_entities / total_time,
            "characters_per_second": total_chars / total_time,
            "avg_entities_per_text": total_entities / len(test_texts) if test_texts else 0,
        }
