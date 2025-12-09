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
        confidence_threshold: float = 0.5,
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

        # Process all regions using batch extraction for better performance
        all_relationships = []
        
        # Prepare batch data
        texts = [region.text_content for region in regions]
        region_types = [region.block_type for region in regions]
        
        # Batch extraction using GLiNER2
        batch_results = self.gliner_extractor.batch_extract_entities_and_relationships(
            texts=texts,
            region_types=region_types,
            batch_size=8  # Process 8 regions at a time
        )

        # Process results and update metadata
        for i, (region, (region_entities, region_relationships)) in enumerate(zip(regions, batch_results)):
            # Update entities with region metadata
            for entity in region_entities:
                entity.metadata["region_id"] = region.region_id
                entity.metadata["block_type"] = region.block_type
                entity.metadata["page_num"] = region.page_num

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

            # Add region relationships
            all_relationships.extend(region_relationships)
            
        print(f"  ✅ Batch extraction complete: {len(all_entities)} entities, {len(all_relationships)} relationships")

        # Cross-region coreference resolution
        if self.enable_cross_region_coreference:
            all_entities = self._resolve_cross_region_coreference(all_entities, entities_by_region)

        # Extract additional cross-region relationships if needed
        if len(all_entities) > sum(len(entities) for entities in entities_by_region.values()):
            print(f"\n[RE] Extracting cross-region relationships between {len(all_entities)} entities using GLiNER2...")
            cross_region_relationships = self.extract_entity_relationships(all_entities, regions, relation_extractor="gliner2")
            all_relationships.extend(cross_region_relationships)

        relationships = all_relationships

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

        return None

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
                source_visual_regions=source_regions,
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
        self, entities: List[ExtractedEntity], regions: List[VisualRegion],
        relation_extractor: str = "gliner2"
    ) -> List[Relationship]:
        """
        Extract relationships between entities using GLiNER2

        Args:
            entities: List of extracted entities
            regions: List of visual regions
            relation_extractor: Use 'gliner2' for relationship extraction

        Returns:
            List of entity relationships extracted by GLiNER2
        """
        if relation_extractor != "gliner2":
            raise ValueError("Only GLiNER2 relation extractor is supported. Please set relation_extractor='gliner2'")

        relationships = []

        # Use GLiNER2 for relationship extraction
        all_gliner2_relations = []
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

            # Extract relationships using GLiNER2
            region_relations = self.gliner_extractor.extract_relationships(
                text=region.text_content,
                entities=region_entities,
                region_type=region.block_type
            )
            all_gliner2_relations.extend(region_relations)

        # Filter and convert GLiNER2 relations to DeepLightRAG format
        for relation in all_gliner2_relations:
            # Quality filter: Only keep high-confidence relations
            confidence = relation.get('confidence', 0.0)
            
            # Apply stricter thresholds based on extraction method
            extraction_method = relation.get('metadata', {}).get('extraction_method', 'unknown')
            
            if extraction_method == 'co-occurrence' and confidence < 0.6:
                continue  # Skip low-confidence co-occurrence relations
            elif extraction_method == 'gliner2-schema' and confidence < 0.75:
                continue  # Skip low-confidence schema relations
            elif confidence < 0.5:
                continue  # Skip very low confidence relations
            
            # Skip RELATED_TO relations if we have more specific relations
            if relation['relation_type'] == 'RELATED_TO' and confidence < 0.7:
                continue
            
            # Create relationship ID
            relation_id = f"rel_{relation['source_entity_id']}_{relation['target_entity_id']}_{relation['relation_type'].lower()}"

            # Create DeepLightRAG Relationship object
            dlr_relationship = Relationship(
                relationship_id=relation_id,
                source_entity_id=relation['source_entity_id'],
                target_entity_id=relation['target_entity_id'],
                relationship_type=relation['relation_type'],
                confidence=confidence,
                description=f"{relation['relation_type']}: {relation['source_entity']} -> {relation['target_entity']}",
                context=relation['context'],
                metadata=relation['metadata']
            )
            relationships.append(dlr_relationship)

        if relationships:
            print(f"    ✅ GLiNER2 schema extracted {len(relationships)} relationships")
        else:
            print(f"    ℹ️ GLiNER2 schema found no relationships")

        return relationships

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
