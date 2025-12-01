"""
Enhanced NER Pipeline for DeepLightRAG
Integrates GLiNER with visual region processing and entity relationships
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
from collections import defaultdict

from .gliner_ner import GLiNERExtractor, ExtractedEntity
from ..ocr.deepseek_ocr import VisualRegion
from ..graph.entity_relationship import Entity, Relationship


@dataclass
class NERProcessingResult:
    """Result of NER processing on a document"""

    total_entities: int
    entities_by_type: Dict[str, int]
    entities_by_region: Dict[str, List[ExtractedEntity]]
    processing_time: float
    confidence_stats: Dict[str, float]
    visual_grounding_success: int

    def get_summary(self) -> str:
        """Get a summary of NER processing results"""
        return f"""
NER Processing Summary:
- Total entities extracted: {self.total_entities}
- Processing time: {self.processing_time:.2f}s
- Average confidence: {self.confidence_stats.get('mean', 0):.2f}
- Visual grounding success: {self.visual_grounding_success}/{self.total_entities}

Entity breakdown:
""" + "\n".join(
            [f"  {entity_type}: {count}" for entity_type, count in self.entities_by_type.items()]
        )


class EnhancedNERPipeline:
    """
    Enhanced NER pipeline that integrates with DeepLightRAG's visual processing
    """

    def __init__(
        self,
        gliner_extractor: Optional[GLiNERExtractor] = None,
        enable_visual_grounding: bool = True,
        enable_cross_region_coreference: bool = True,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize Enhanced NER Pipeline

        Args:
            gliner_extractor: GLiNER extractor instance
            enable_visual_grounding: Whether to ground entities to visual regions
            enable_cross_region_coreference: Whether to resolve entities across regions
            confidence_threshold: Minimum confidence threshold for entities
        """
        self.gliner_extractor = gliner_extractor or GLiNERExtractor()
        self.enable_visual_grounding = enable_visual_grounding
        self.enable_cross_region_coreference = enable_cross_region_coreference
        self.confidence_threshold = confidence_threshold

        # Processing statistics
        self.processing_stats = {
            "total_documents": 0,
            "total_regions_processed": 0,
            "total_entities_extracted": 0,
            "avg_processing_time": 0.0,
            "entities_by_region_type": defaultdict(int),
        }

    def process_document_regions(
        self, regions: List[VisualRegion], document_id: str = "unknown", full_text: str = ""
    ) -> Dict[str, Any]:
        """
        Process all regions in a document for entity extraction

        Args:
            regions: List of visual regions from OCR
            document_id: Document identifier
            full_text: Full document text (optional)

        Returns:
            Dictionary with entities, relationships, and metadata
        """
        start_time = time.time()

        all_entities = []
        entities_by_region = {}
        entities_by_type = defaultdict(int)
        visual_grounding_success = 0

        print(f"\n[NER] Processing {len(regions)} regions for entity extraction...")

        # Process each region
        for i, region in enumerate(regions):
            print(f"  Processing region {i+1}/{len(regions)} ({region.block_type})")

            # Extract entities from region text
            region_entities = self._extract_entities_from_region(region)

            # Store entities by region
            entities_by_region[region.region_id] = region_entities

            # Update statistics
            for entity in region_entities:
                entities_by_type[entity.label] += 1
                all_entities.append(entity)

                # Check visual grounding
                if self._has_visual_grounding(entity, region):
                    visual_grounding_success += 1

                # Update region type statistics
                self.processing_stats["entities_by_region_type"][region.block_type] += 1

        # Cross-region coreference resolution
        if self.enable_cross_region_coreference:
            all_entities = self._resolve_cross_region_coreference(all_entities, entities_by_region)

        # Extract relationships between entities
        print(f"\n[RE] Extracting relationships between {len(all_entities)} entities...")
        relationships = self.extract_entity_relationships(all_entities, regions)

        # Calculate confidence statistics
        confidences = [e.confidence for e in all_entities]
        confidence_stats = {
            "mean": sum(confidences) / len(confidences) if confidences else 0,
            "min": min(confidences) if confidences else 0,
            "max": max(confidences) if confidences else 0,
        }

        processing_time = time.time() - start_time

        # Update global statistics
        self._update_processing_stats(len(regions), len(all_entities), processing_time)

        # Return enhanced result format expected by test
        return {
            "entities": all_entities,
            "relationships": relationships,
            "metadata": {
                "total_entities": len(all_entities),
                "entities_by_type": dict(entities_by_type),
                "entities_by_region": entities_by_region,
                "processing_time": processing_time,
                "confidence_stats": confidence_stats,
                "visual_grounding_success": visual_grounding_success,
                "document_id": document_id,
            },
        }

    def _extract_entities_from_region(self, region: VisualRegion) -> List[ExtractedEntity]:
        """
        Extract entities from a single visual region

        Args:
            region: Visual region to process

        Returns:
            List of extracted entities
        """
        # Determine which entity types to focus on based on region type
        focused_entity_types = self._get_focused_entity_types(region.block_type)

        # Extract entities using GLiNER
        entities = self.gliner_extractor.extract_entities(
            text=region.text_content,
            entity_types=focused_entity_types,
            region_type=region.block_type,
        )

        # Filter by confidence threshold
        entities = [e for e in entities if e.confidence >= self.confidence_threshold]

        # Add visual grounding information
        if self.enable_visual_grounding:
            entities = self._add_visual_grounding(entities, region)

        return entities

    def _get_focused_entity_types(self, block_type: str) -> List[str]:
        """
        Get focused entity types based on region block type

        Args:
            block_type: Type of visual region

        Returns:
            List of relevant entity types
        """
        # Map block types to relevant entity types
        block_type_mapping = {
            "header": ["CONCEPT", "TECHNICAL_TERM", "PRODUCT"],
            "paragraph": [
                "PERSON",
                "ORGANIZATION",
                "LOCATION",
                "DATE_TIME",
                "TECHNICAL_TERM",
                "CONCEPT",
            ],
            "table": ["METRIC", "PERCENTAGE", "MONEY", "METRIC_RESULT", "DATE_TIME"],
            "figure": ["REFERENCE", "METRIC_RESULT", "TECHNICAL_TERM"],
            "caption": ["REFERENCE", "CONCEPT", "TECHNICAL_TERM"],
            "list": ["TECHNICAL_TERM", "PRODUCT", "METHOD", "CONCEPT"],
            "formula": ["TECHNICAL_TERM", "METHOD", "METRIC"],
        }

        # Get focused types, fallback to all types
        focused_types = block_type_mapping.get(block_type, None)

        return focused_types

    def _add_visual_grounding(
        self, entities: List[ExtractedEntity], region: VisualRegion
    ) -> List[ExtractedEntity]:
        """
        Add visual grounding information to entities

        Args:
            entities: List of extracted entities
            region: Visual region containing the entities

        Returns:
            Enhanced entities with visual grounding
        """
        for entity in entities:
            # Add region information
            entity.metadata.update(
                {
                    "region_id": region.region_id,
                    "page_num": region.page_num,
                    "block_type": region.block_type,
                    "bbox": region.bbox.to_list(),
                    "region_confidence": region.confidence,
                }
            )

            # Calculate relative position within region text
            if entity.start >= 0 and entity.end <= len(region.text_content):
                relative_start = entity.start / len(region.text_content)
                relative_end = entity.end / len(region.text_content)

                entity.metadata.update(
                    {
                        "relative_start": relative_start,
                        "relative_end": relative_end,
                        "text_position": (
                            "start"
                            if relative_start < 0.3
                            else ("end" if relative_start > 0.7 else "middle")
                        ),
                    }
                )

        return entities

    def _has_visual_grounding(self, entity: ExtractedEntity, region: VisualRegion) -> bool:
        """Check if entity has successful visual grounding"""
        return (
            "region_id" in entity.metadata
            and "bbox" in entity.metadata
            and entity.metadata["region_id"] == region.region_id
        )

    def _resolve_cross_region_coreference(
        self,
        all_entities: List[ExtractedEntity],
        entities_by_region: Dict[str, List[ExtractedEntity]],
    ) -> List[ExtractedEntity]:
        """
        Resolve entity coreferences across regions

        Args:
            all_entities: All extracted entities
            entities_by_region: Entities grouped by region

        Returns:
            Entities with coreference resolved
        """
        # Group entities by normalized text and type
        entity_groups = defaultdict(list)

        for entity in all_entities:
            key = (entity.normalized_form, entity.label)
            entity_groups[key].append(entity)

        # Resolve coreferences within groups
        resolved_entities = []

        for (normalized_text, entity_type), group in entity_groups.items():
            if len(group) == 1:
                # No coreference needed
                resolved_entities.extend(group)
            else:
                # Merge entities with same normalized form
                primary_entity = max(group, key=lambda e: e.confidence)

                # Add coreference information to primary entity
                primary_entity.metadata["coreferences"] = []
                primary_entity.metadata["mention_count"] = len(group)

                for other_entity in group:
                    if other_entity != primary_entity:
                        primary_entity.metadata["coreferences"].append(
                            {
                                "region_id": other_entity.metadata.get("region_id"),
                                "text": other_entity.text,
                                "confidence": other_entity.confidence,
                            }
                        )

                resolved_entities.append(primary_entity)

        return resolved_entities

    def _update_processing_stats(self, num_regions: int, num_entities: int, processing_time: float):
        """Update processing statistics"""
        self.processing_stats["total_documents"] += 1
        self.processing_stats["total_regions_processed"] += num_regions
        self.processing_stats["total_entities_extracted"] += num_entities

        # Update running average of processing time
        total_docs = self.processing_stats["total_documents"]
        current_avg = self.processing_stats["avg_processing_time"]
        self.processing_stats["avg_processing_time"] = (
            current_avg * (total_docs - 1) + processing_time
        ) / total_docs

    def convert_to_deeplightrag_entities(
        self, extracted_entities: List[ExtractedEntity], document_id: str = "unknown"
    ) -> List[Entity]:
        """
        Convert GLiNER entities to DeepLightRAG Entity objects

        Args:
            extracted_entities: List of extracted entities from GLiNER
            document_id: Document identifier

        Returns:
            List of DeepLightRAG Entity objects
        """
        deeplightrag_entities = []

        for i, entity in enumerate(extracted_entities):
            # Create entity ID
            entity_id = f"{document_id}_entity_{i}_{entity.label.lower()}"

            # Extract visual grounding information
            source_regions = [entity.metadata.get("region_id", "unknown")]
            grounding_boxes = [entity.metadata.get("bbox", [])]
            block_type_context = [entity.metadata.get("block_type", "unknown")]
            page_numbers = [entity.metadata.get("page_num", 0)]

            # Create DeepLightRAG Entity
            dlr_entity = Entity(
                entity_id=entity_id,
                name=entity.text,
                entity_type=entity.label,
                value=entity.normalized_form,
                description=f"{entity.label} entity: {entity.text}",
                source_regions=source_regions,
                grounding_boxes=grounding_boxes,
                block_type_context=block_type_context,
                confidence=entity.confidence,
                mention_count=entity.metadata.get("mention_count", 1),
                page_numbers=page_numbers,
                metadata=entity.metadata,
            )

            deeplightrag_entities.append(dlr_entity)

        return deeplightrag_entities

    def extract_entity_relationships(
        self, entities: List[ExtractedEntity], regions: List[VisualRegion]
    ) -> List[Relationship]:
        """
        Extract relationships between entities using OpenNRE + patterns

        Args:
            entities: List of extracted entities
            regions: List of visual regions

        Returns:
            List of entity relationships
        """
        relationships = []

        # NEW: Try DeBERTa-based relation extraction first (highest accuracy)
        try:
            from .deberta_relation_extractor import DeBERTaRelationExtractor

            if not hasattr(self, "_deberta_extractor"):
                print("    🤖 Initializing DeBERTa relation extraction (ACE05/TACRED)...")
                self._deberta_extractor = DeBERTaRelationExtractor(
                    confidence_threshold=0.5, device="auto"
                )

            # Extract relations using DeBERTa
            all_deberta_relations = []
            entities_by_region = defaultdict(list)

            for entity in entities:
                region_id = entity.metadata.get("region_id", "unknown")
                entities_by_region[region_id].append(entity)

            # Process each region
            for region_id, region_entities in entities_by_region.items():
                if len(region_entities) < 2:
                    continue

                region = next((r for r in regions if r.region_id == region_id), None)
                if not region:
                    continue

                region_relations = self._deberta_extractor.extract_relations_from_entities(
                    entities=region_entities, text=region.text_content, region=region
                )
                all_deberta_relations.extend(region_relations)

            # Convert to DeepLightRAG format
            if all_deberta_relations:
                relationships = self._deberta_extractor.convert_to_deeplightrag_relationships(
                    deberta_relations=all_deberta_relations, document_id="pipeline_extract"
                )

                print(
                    f"    ✅ DeBERTa extracted {len(relationships)} relationships ({len(all_deberta_relations)} total)"
                )
                return relationships
            else:
                print(f"    ⚠️ DeBERTa found no relationships, trying OpenNRE fallback...")

        except Exception as e:
            print(f"    ❌ DeBERTa extraction failed: {e}")
            print(f"    🔄 Falling back to OpenNRE...")

        # FALLBACK 1: Try OpenNRE-based relation extraction
        try:
            from .relation_extractor import RelationExtractionPipeline

            if not hasattr(self, "_relation_pipeline"):
                print("    🔗 Initializing OpenNRE relation extraction pipeline...")
                self._relation_pipeline = RelationExtractionPipeline()

            # Extract relations using OpenNRE + patterns
            result = self._relation_pipeline.process_entities_for_relations(
                entities=entities, regions=regions, document_id="pipeline_extract"
            )

            relationships = result["relationships"]

            if len(relationships) > 0:
                print(f"    ✅ OpenNRE extracted {len(relationships)} relationships")
                return relationships
            else:
                print(f"    ⚠️ OpenNRE found no relationships, trying pattern fallback...")

        except Exception as e:
            print(f"    ❌ OpenNRE extraction failed: {e}")
            print(f"    🔄 Falling back to LLM extraction...")

        # FALLBACK 2: Original co-occurrence relationships
        entities_by_region = defaultdict(list)
        for entity in entities:
            region_id = entity.metadata.get("region_id", "unknown")
            entities_by_region[region_id].append(entity)

        # Extract co-occurrence relationships within regions
        for region_id, region_entities in entities_by_region.items():
            if len(region_entities) < 2:
                continue

            # Find corresponding region
            region = next((r for r in regions if r.region_id == region_id), None)
            if not region:
                continue

            # Create co-occurrence relationships
            for i, entity1 in enumerate(region_entities):
                for entity2 in region_entities[i + 1 :]:
                    relationship = self._create_cooccurrence_relationship(entity1, entity2, region)
                    if relationship:
                        relationships.append(relationship)

        print(f"    📊 Co-occurrence extraction found {len(relationships)} relationships")
        return relationships

    def _create_cooccurrence_relationship(
        self, entity1: ExtractedEntity, entity2: ExtractedEntity, region: VisualRegion
    ) -> Optional[Relationship]:
        """Create a co-occurrence relationship between two entities"""

        # Calculate distance between entities in text
        distance = abs(entity1.start - entity2.start)
        max_distance = 200  # Maximum character distance for relationship

        if distance > max_distance:
            return None

        # Determine relationship strength based on distance and context
        weight = max(0.1, 1.0 - (distance / max_distance))

        # Create relationship
        relationship = Relationship(
            source_entity=f"entity_{entity1.text}_{entity1.label}",
            target_entity=f"entity_{entity2.text}_{entity2.label}",
            relationship_type="co_occurs_with",
            description=f"{entity1.text} co-occurs with {entity2.text} in {region.block_type}",
            weight=weight,
            spatial_cooccurrence=True,
            layout_aware_type=f"co_occurrence_{region.block_type}",
            source_regions=[region.region_id],
            evidence_text=region.text_content[
                min(entity1.start, entity2.start) : max(entity1.end, entity2.end)
            ],
            metadata={
                "distance": distance,
                "region_type": region.block_type,
                "page_num": region.page_num,
            },
        )

        return relationship

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return dict(self.processing_stats)

    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            "total_documents": 0,
            "total_regions_processed": 0,
            "total_entities_extracted": 0,
            "avg_processing_time": 0.0,
            "entities_by_region_type": defaultdict(int),
        }
