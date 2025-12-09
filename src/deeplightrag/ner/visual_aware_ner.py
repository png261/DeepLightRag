"""
Visual-Aware NER Pipeline
Enhanced Named Entity Recognition using full visual context from DeepSeek-OCR
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import time

from .gliner_ner import GLiNERExtractor, ExtractedEntity
from ..ocr.deepseek_ocr import VisualRegion
from ..ocr.visual_feature_extractor import ComprehensiveVisualExtractor
from ..graph.entity_relationship import Entity, Relationship


class VisualAwareNERPipeline:
    """
    Enhanced NER pipeline that fully utilizes DeepSeek-OCR's visual features
    """

    def __init__(
        self,
        gliner_extractor: Optional[GLiNERExtractor] = None,
        enable_visual_validation: bool = True,
        enable_spatial_reasoning: bool = True,
        confidence_threshold: float = 0.3,
        visual_weight: float = 0.4,  # Weight for visual features in confidence
    ):
        """
        Initialize Visual-Aware NER Pipeline

        Args:
            gliner_extractor: GLiNER extractor instance
            enable_visual_validation: Whether to use visual features for validation
            enable_spatial_reasoning: Whether to use spatial relationships
            confidence_threshold: Minimum confidence threshold
            visual_weight: Weight for visual features in final confidence
        """
        self.gliner_extractor = gliner_extractor or GLiNERExtractor()
        self.visual_extractor = ComprehensiveVisualExtractor()
        self.enable_visual_validation = enable_visual_validation
        self.enable_spatial_reasoning = enable_spatial_reasoning
        self.confidence_threshold = confidence_threshold
        self.visual_weight = visual_weight

        # Processing statistics
        self.stats = {
            "total_entities": 0,
            "visually_validated": 0,
            "spatially_resolved": 0,
            "confidence_boosted": 0,
        }

    def process_document_with_visual_context(
        self,
        image,
        regions: List[VisualRegion],
        document_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Process document with full visual context

        Args:
            image: Full page image
            regions: List of visual regions
            document_id: Document identifier

        Returns:
            Dictionary with entities, relationships, and visual features
        """
        start_time = time.time()

        print(f"\n[Visual-NER] Processing {len(regions)} regions with full visual context...")

        # 1. Extract comprehensive visual features
        visual_features = self.visual_extractor.extract_all_features(image, regions)

        # 2. Extract entities with visual enhancement
        all_entities = []
        entities_by_region = {}
        entities_by_type = defaultdict(int)

        for region in regions:
            # Standard NER extraction
            entities = self._extract_entities_with_visual_boost(region, visual_features)

            # Store by region
            entities_by_region[region.region_id] = entities
            all_entities.extend(entities)

            # Update statistics
            for entity in entities:
                entities_by_type[entity.label] += 1
                if entity.metadata.get("visually_validated", False):
                    self.stats["visually_validated"] += 1
                if entity.confidence > 0.7:  # High confidence after boost
                    self.stats["confidence_boosted"] += 1

        # 3. Apply spatial reasoning
        if self.enable_spatial_reasoning:
            all_entities = self._apply_spatial_reasoning(
                all_entities, regions, visual_features["spatial_relationships"]
            )

        # 4. Extract enhanced relationships
        relationships = self._extract_visual_enhanced_relationships(
            all_entities, regions, visual_features
        )

        # 5. Apply cross-region entity linking
        linked_entities = self._link_cross_region_entities(all_entities, visual_features)

        processing_time = time.time() - start_time

        # Update statistics
        self.stats["total_entities"] += len(all_entities)

        return {
            "entities": linked_entities,
            "relationships": relationships,
            "visual_features": visual_features,
            "metadata": {
                "total_entities": len(all_entities),
                "entities_by_type": dict(entities_by_type),
                "entities_by_region": entities_by_region,
                "processing_time": processing_time,
                "visual_validation_stats": dict(self.stats),
                "document_id": document_id,
            }
        }

    def _extract_entities_with_visual_boost(
        self, region: VisualRegion, visual_features: Dict[str, Any]
    ) -> List[ExtractedEntity]:
        """Extract entities with visual context enhancement"""
        # Get focused entity types based on region type
        focused_types = self._get_visual_focused_types(region)

        # Extract entities using GLiNER
        entities = self.gliner_extractor.extract_entities(
            text=region.text_content,
            entity_types=focused_types,
            region_type=region.block_type,
        )

        # Enhance with visual features
        enhanced_entities = []
        for entity in entities:
            # Get visual features for this region
            region_features = visual_features["region_features"].get(region.region_id, {})

            # Apply visual validation
            visual_score = self._validate_entity_visually(entity, region, region_features)

            # Calculate enhanced confidence
            enhanced_confidence = self._calculate_enhanced_confidence(
                entity.confidence, visual_score, region
            )

            # Update entity with visual information
            enhanced_entity = ExtractedEntity(
                text=entity.text,
                start=entity.start,
                end=entity.end,
                label=entity.label,
                confidence=enhanced_confidence,
                metadata={
                    **entity.metadata,
                    "visually_validated": visual_score > 0.7,
                    "visual_score": visual_score,
                    "region_features": region_features,
                    "spatial_context": self._extract_spatial_context(entity, region),
                    "visual_hierarchy": self._determine_visual_hierarchy(entity, region),
                }
            )

            enhanced_entities.append(enhanced_entity)

        return enhanced_entities

    def _get_visual_focused_types(self, region: VisualRegion) -> Optional[List[str]]:
        """Get focused entity types based on visual region analysis"""
        # Use visual features to determine relevant entity types
        region_features = getattr(region, 'layout_features', {})

        if region.block_type == "table":
            # Table regions: focus on data entities
            return ["METRIC", "PERCENTAGE", "MONEY", "DATE_TIME", "ORGANIZATION"]
        elif region.block_type == "figure":
            # Figure regions: focus on references and concepts
            return ["REFERENCE", "CONCEPT", "TECHNICAL_TERM", "PERSON"]
        elif region.block_type == "header":
            # Headers: titles and high-level concepts
            return ["CONCEPT", "TECHNICAL_TERM", "ORGANIZATION"]
        elif region.block_type == "formula":
            # Formulas: technical and scientific terms
            return ["TECHNICAL_TERM", "CONCEPT", "METHOD"]
        else:
            # Default types
            return None

    def _validate_entity_visually(
        self,
        entity: ExtractedEntity,
        region: VisualRegion,
        region_features: Dict[str, Any]
    ) -> float:
        """Validate entity using visual features"""
        if not region_features:
            return 0.5  # Default if no features

        visual_score = 0.5

        # 1. Quality-based validation
        quality = region_features.get("quality", {})
        overall_quality = quality.get("overall", 0.5)
        clarity = quality.get("clarity", 0.5)
        contrast = quality.get("contrast", 0.5)

        quality_score = (overall_quality + clarity + contrast) / 3
        visual_score += quality_score * 0.3

        # 2. Position-based validation
        spatial = region_features.get("spatial", {})
        position_score = spatial.get("position_score", 0.5)
        is_header = spatial.get("is_header", False)
        is_centered = spatial.get("is_centered", False)

        # Boost certain entities in specific positions
        if entity.label == "PERSON" and is_centered:
            position_score *= 1.2
        elif entity.label in ["ORGANIZATION", "CONCEPT"] and is_header:
            position_score *= 1.3

        visual_score += position_score * 0.2

        # 3. Readability validation
        readability = region_features.get("readability", {})
        text_density = readability.get("word_count", 0) / 100  # Normalized

        # Higher density for certain entity types
        if entity.label in ["METRIC", "PERCENTAGE", "MONEY"]:
            text_density = min(1.0, text_density * 1.5)

        visual_score += text_density * 0.2

        # 4. Content-specific validation
        content_specific = region_features.get("content_specific", {})
        if region.block_type == "table" and entity.label in ["METRIC", "PERCENTAGE"]:
            visual_score += 0.3
        elif region.block_type == "figure" and entity.label == "REFERENCE":
            visual_score += 0.3

        return min(1.0, visual_score)

    def _calculate_enhanced_confidence(
        self, base_confidence: float, visual_score: float, region: VisualRegion
    ) -> float:
        """Calculate confidence enhanced with visual features"""
        # Weighted combination of base confidence and visual validation
        enhanced = (
            base_confidence * (1 - self.visual_weight) +
            visual_score * self.visual_weight
        )

        # Apply region-specific boosts
        if region.embedding_confidence > 0.8:
            enhanced *= 1.1

        if region.visual_complexity > 0.7 and region.block_type in ["table", "figure"]:
            enhanced *= 1.05

        return min(1.0, enhanced)

    def _extract_spatial_context(
        self, entity: ExtractedEntity, region: VisualRegion
    ) -> Dict[str, Any]:
        """Extract spatial context for entity"""
        return {
            "region_position": {
                "x": (region.bbox.x1 + region.bbox.x2) / 2,
                "y": (region.bbox.y1 + region.bbox.y2) / 2,
                "width": region.bbox.width(),
                "height": region.bbox.height(),
            },
            "relative_position": {
                "start_ratio": entity.start / max(1, len(region.text_content)),
                "end_ratio": entity.end / max(1, len(region.text_content)),
            },
            "neighbor_regions": region.spatial_neighbors,
            "visual_hierarchy": region.visual_hierarchy,
        }

    def _determine_visual_hierarchy(
        self, entity: ExtractedEntity, region: VisualRegion
    ) -> str:
        """Determine visual hierarchy level for entity"""
        if region.block_type == "header":
            if entity.label in ["PERSON", "ORGANIZATION"]:
                return "primary_title"
            return "secondary_title"
        elif region.block_type == "paragraph":
            if entity.label == "PERSON" and entity.start == 0:
                return "topic_sentence"
            return "supporting_content"
        elif region.block_type == "table":
            if entity.label == "ORGANIZATION":
                return "table_header"
            return "table_data"
        else:
            return "standard"

    def _apply_spatial_reasoning(
        self,
        entities: List[ExtractedEntity],
        regions: List[VisualRegion],
        spatial_relationships: Dict[str, Any]
    ) -> List[ExtractedEntity]:
        """Apply spatial reasoning to enhance and validate entities"""
        enhanced_entities = []

        # Create spatial lookup
        region_lookup = {r.region_id: r for r in regions}

        # Process entities with spatial context
        for entity in entities:
            enhanced_entity = entity

            # Find related entities in neighboring regions
            related_entities = self._find_spatially_related_entities(
                entity, entities, spatial_relationships
            )

            if related_entities:
                # Update metadata with spatial relationships
                enhanced_entity.metadata["spatially_related"] = related_entities

                # Boost confidence for consistent entities
                consistent_types = [e.label for e in related_entities if e.label == entity.label]
                if len(consistent_types) > 1:
                    boost = min(0.2, len(consistent_types) * 0.05)
                    enhanced_entity.confidence = min(1.0, enhanced_entity.confidence + boost)
                    self.stats["spatially_resolved"] += 1

            enhanced_entities.append(enhanced_entity)

        return enhanced_entities

    def _find_spatially_related_entities(
        self,
        target_entity: ExtractedEntity,
        all_entities: List[ExtractedEntity],
        spatial_relationships: Dict[str, Any]
    ) -> List[ExtractedEntity]:
        """Find entities that are spatially related"""
        related = []

        # Get region ID from target entity metadata
        target_region_id = target_entity.metadata.get("region_id")
        if not target_region_id:
            return related

        # Find neighboring regions
        neighbors = spatial_relationships.get("neighbors", {}).get(target_region_id, [])

        # Find entities in neighboring regions
        for neighbor in neighbors:
            neighbor_region_id = neighbor["region_id"]
            distance = neighbor["distance"]

            # Find entities in neighbor region
            for entity in all_entities:
                if entity.metadata.get("region_id") == neighbor_region_id:
                    # Check if entities are similar
                    if self._are_entities_spatially_related(target_entity, entity, distance):
                        related.append(entity)

        return related

    def _are_entities_spatially_related(
        self, entity1: ExtractedEntity, entity2: ExtractedEntity, distance: float
    ) -> bool:
        """Check if two entities are spatially related"""
        # Same type and close together
        if entity1.label == entity2.label and distance < 0.1:
            return True

        # Certain types that often appear together
        type_pairs = [
            ("PERSON", "ORGANIZATION"),
            ("METRIC", "ORGANIZATION"),
            ("DATE_TIME", "PERSON"),
            ("LOCATION", "ORGANIZATION"),
        ]

        return (entity1.label, entity2.label) in type_pairs or \
               (entity2.label, entity1.label) in type_pairs

    def _extract_visual_enhanced_relationships(
        self,
        entities: List[ExtractedEntity],
        regions: List[VisualRegion],
        visual_features: Dict[str, Any]
    ) -> List[Relationship]:
        """Extract relationships using visual context"""
        relationships = []

        # 1. Layout-based relationships
        layout_rels = self._extract_layout_relationships(entities, regions)
        relationships.extend(layout_rels)

        # 2. Spatial proximity relationships
        spatial_rels = self._extract_spatial_relationships(
            entities, visual_features["spatial_relationships"]
        )
        relationships.extend(spatial_rels)

        # 3. Visual hierarchy relationships
        hierarchy_rels = self._extract_hierarchy_relationships(entities, visual_features)
        relationships.extend(hierarchy_rels)

        # 4. Content-specific relationships (e.g., figure-caption)
        content_rels = self._extract_content_specific_relationships(entities, regions)
        relationships.extend(content_rels)

        return relationships

    def _extract_layout_relationships(
        self, entities: List[ExtractedEntity], regions: List[VisualRegion]
    ) -> List[Relationship]:
        """Extract relationships based on layout patterns"""
        relationships = []

        # Sort regions by reading order
        sorted_regions = sorted(regions, key=lambda r: (r.page_num, r.bbox.y1, r.bbox.x1))

        # Create region to entities mapping
        region_entities = defaultdict(list)
        for entity in entities:
            region_id = entity.metadata.get("region_id")
            if region_id:
                region_entities[region_id].append(entity)

        # Find relationships between consecutive regions
        for i in range(len(sorted_regions) - 1):
            region1 = sorted_regions[i]
            region2 = sorted_regions[i + 1]

            entities1 = region_entities.get(region1.region_id, [])
            entities2 = region_entities.get(region2.region_id, [])

            # Create cross-region relationships
            for entity1 in entities1:
                for entity2 in entities2:
                    # Check if relationship makes sense
                    if self._is_valid_layout_relationship(entity1, entity2, region1, region2):
                        rel_type = self._determine_relationship_type(entity1, entity2, region1, region2)

                        relationship = Relationship(
                            relationship_id=f"{entity1.text}_{entity2.text}_{rel_type}",
                            source_entity_id=entity1.text,
                            target_entity_id=entity2.text,
                            relationship_type=rel_type,
                            confidence=min(entity1.confidence, entity2.confidence),
                            description=f"{rel_type}: {entity1.text} -> {entity2.text}",
                            context=f"Layout relationship between {region1.block_type} and {region2.block_type}",
                            metadata={
                                "source_region": region1.region_id,
                                "target_region": region2.region_id,
                                "relationship_type": "layout",
                                "distance": abs(region1.bbox.y2 - region2.bbox.y1),
                            }
                        )
                        relationships.append(relationship)

        return relationships

    def _extract_spatial_relationships(
        self,
        entities: List[ExtractedEntity],
        spatial_relationships: Dict[str, Any]
    ) -> List[Relationship]:
        """Extract relationships based on spatial proximity"""
        relationships = []

        # Group entities by region
        region_entities = defaultdict(list)
        for entity in entities:
            region_id = entity.metadata.get("region_id")
            if region_id:
                region_entities[region_id].append(entity)

        # Find relationships within regions
        for region_id, region_ents in region_entities.items():
            for i, entity1 in enumerate(region_ents):
                for entity2 in region_ents[i+1:]:
                    # Check if entities are close in text
                    if entity2.start - entity1.end < 50:  # Within 50 characters
                        rel_type = self._determine_proximity_relationship(entity1, entity2)

                        if rel_type:
                            relationship = Relationship(
                                relationship_id=f"{entity1.text}_{entity2.text}_{rel_type}_prox",
                                source_entity_id=entity1.text,
                                target_entity_id=entity2.text,
                                relationship_type=rel_type,
                                confidence=min(entity1.confidence, entity2.confidence) * 0.9,
                                description=f"{rel_type}: {entity1.text} -> {entity2.text}",
                                context=f"Proximity relationship in region {region_id}",
                                metadata={
                                    "source_region": region_id,
                                    "target_region": region_id,
                                    "relationship_type": "proximity",
                                    "text_distance": entity2.start - entity1.end,
                                }
                            )
                            relationships.append(relationship)

        return relationships

    def _extract_hierarchy_relationships(
        self,
        entities: List[ExtractedEntity],
        visual_features: Dict[str, Any]
    ) -> List[Relationship]:
        """Extract relationships based on visual hierarchy"""
        relationships = []

        # Find header-content relationships
        for entity in entities:
            hierarchy = entity.metadata.get("visual_hierarchy", "")

            if hierarchy == "primary_title":
                # Find supporting content entities
                for other in entities:
                    other_hierarchy = other.metadata.get("visual_hierarchy", "")
                    if other_hierarchy in ["secondary_title", "topic_sentence", "supporting_content"]:
                        if other.metadata.get("region_id") != entity.metadata.get("region_id"):
                            relationship = Relationship(
                                relationship_id=f"{entity.text}_{other.text}_describes",
                                source_entity_id=entity.text,
                                target_entity_id=other.text,
                                relationship_type="describes",
                                confidence=min(entity.confidence, other.confidence) * 0.85,
                                description=f"Describes: {entity.text} -> {other.text}",
                                context="Visual hierarchy relationship",
                                metadata={
                                    "relationship_type": "hierarchy",
                                    "source_hierarchy": hierarchy,
                                    "target_hierarchy": other_hierarchy,
                                }
                            )
                            relationships.append(relationship)

        return relationships

    def _extract_content_specific_relationships(
        self, entities: List[ExtractedEntity], regions: List[VisualRegion]
    ) -> List[Relationship]:
        """Extract content-specific relationships (e.g., figure-caption)"""
        relationships = []

        # Find figure-caption relationships
        for region in regions:
            if region.block_type == "figure":
                figure_entities = [e for e in entities if e.metadata.get("region_id") == region.region_id]

                # Look for caption regions nearby
                for other_region in regions:
                    if other_region.block_type == "caption":
                        # Check if caption is close to figure
                        if abs(region.bbox.y2 - other_region.bbox.y1) < 0.1:
                            caption_entities = [e for e in entities if e.metadata.get("region_id") == other_region.region_id]

                            for fig_ent in figure_entities:
                                for cap_ent in caption_entities:
                                    relationship = Relationship(
                                        relationship_id=f"{fig_ent.text}_{cap_ent.text}_caption",
                                        source_entity_id=fig_ent.text,
                                        target_entity_id=cap_ent.text,
                                        relationship_type="caption",
                                        confidence=min(fig_ent.confidence, cap_ent.confidence) * 0.9,
                                        description=f"Caption: {fig_ent.text} -> {cap_ent.text}",
                                        context="Figure-caption relationship",
                                        metadata={
                                            "relationship_type": "content_specific",
                                            "figure_region": region.region_id,
                                            "caption_region": other_region.region_id,
                                        }
                                    )
                                    relationships.append(relationship)

        return relationships

    def _link_cross_region_entities(
        self, entities: List[ExtractedEntity], visual_features: Dict[str, Any]
    ) -> List[Entity]:
        """Link entities across regions and convert to DeepLightRAG Entity objects"""
        # Group entities by normalized form and type
        entity_groups = defaultdict(list)

        for entity in entities:
            # Create normalized key
            normalized = entity.text.lower().strip()
            key = (normalized, entity.label)
            entity_groups[key].append(entity)

        # Resolve and create final entities
        final_entities = []

        for (normalized, entity_type), group in entity_groups.items():
            if len(group) == 1:
                # Single instance
                entity = group[0]
            else:
                # Multiple instances - merge
                entity = self._merge_entity_mentions(group)

            # Create DeepLightRAG Entity
            dlr_entity = Entity(
                entity_id=f"entity_{entity_type}_{normalized.replace(' ', '_')}",
                name=entity.text,
                entity_type=entity.label,
                value=normalized,
                description=f"{entity_type} entity: {entity.text}",
                source_regions=[entity.metadata.get("region_id", "unknown")],
                grounding_boxes=[],
                block_type_context=[entity.metadata.get("block_type", "unknown")],
                confidence=entity.confidence,
                mention_count=len(group),
                page_numbers=[entity.metadata.get("page_num", 0)],
                metadata={
                    **entity.metadata,
                    "visual_features": entity.metadata.get("region_features", {}),
                    "spatial_context": entity.metadata.get("spatial_context", {}),
                    "cross_region_mentions": len(group) - 1,
                }
            )
            final_entities.append(dlr_entity)

        return final_entities

    def _merge_entity_mentions(self, entities: List[ExtractedEntity]) -> ExtractedEntity:
        """Merge multiple mentions of the same entity"""
        # Select the highest confidence entity as primary
        primary = max(entities, key=lambda e: e.confidence)

        # Aggregate metadata
        all_regions = [e.metadata.get("region_id") for e in entities if e.metadata.get("region_id")]
        all_pages = [e.metadata.get("page_num") for e in entities if e.metadata.get("page_num")]

        # Create merged entity
        merged = ExtractedEntity(
            text=primary.text,
            start=primary.start,
            end=primary.end,
            label=primary.label,
            confidence=primary.confidence,
            metadata={
                **primary.metadata,
                "merged_mentions": len(entities),
                "all_regions": list(set(all_regions)),
                "all_pages": list(set(all_pages)),
                "original_confidences": [e.confidence for e in entities],
            }
        )

        return merged

    def _is_valid_layout_relationship(
        self,
        entity1: ExtractedEntity,
        entity2: ExtractedEntity,
        region1: VisualRegion,
        region2: VisualRegion
    ) -> bool:
        """Check if a layout relationship between entities is valid"""
        # Same region - don't create relationship
        if region1.region_id == region2.region_id:
            return False

        # Certain entity pairs don't make sense
        invalid_pairs = [
            ("DATE_TIME", "DATE_TIME"),
            ("PERCENTAGE", "PERCENTAGE"),
        ]

        return (entity1.label, entity2.label) not in invalid_pairs and \
               (entity2.label, entity1.label) not in invalid_pairs

    def _determine_relationship_type(
        self,
        entity1: ExtractedEntity,
        entity2: ExtractedEntity,
        region1: VisualRegion,
        region2: VisualRegion
    ) -> str:
        """Determine relationship type based on entities and regions"""
        # Check for specific patterns
        if entity1.label == "PERSON" and entity2.label == "ORGANIZATION":
            return "works_at"
        elif entity1.label == "ORGANIZATION" and entity2.label == "PERSON":
            return "employs"
        elif entity1.label == "LOCATION" and entity2.label == "ORGANIZATION":
            return "located_in"
        elif entity1.label == "DATE_TIME" and entity2.label in ["PERSON", "ORGANIZATION"]:
            return "published_on"
        elif region1.block_type == "table" and entity2.label in ["METRIC", "PERCENTAGE"]:
            return "contains_metric"
        else:
            return "related_to"

    def _determine_proximity_relationship(
        self, entity1: ExtractedEntity, entity2: ExtractedEntity
    ) -> Optional[str]:
        """Determine relationship based on proximity"""
        # Common proximity patterns
        if entity1.label == "PERSON" and entity2.label == "PERSON":
            return "and"
        elif entity1.label == "ORGANIZATION" and entity2.label == "ORGANIZATION":
            return "and"
        elif entity1.label == "LOCATION" and entity2.label == "LOCATION":
            return "and"
        elif entity1.label in ["METRIC", "MONEY"] and entity2.label in ["PERCENTAGE", "RATE"]:
            return "at_rate"
        elif entity1.label == "DATE_TIME" and entity2.label in ["MONEY", "METRIC"]:
            return "on_date"
        else:
            return None

    def get_visual_enhanced_stats(self) -> Dict[str, Any]:
        """Get statistics about visual enhancement"""
        return dict(self.stats)

    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            "total_entities": 0,
            "visually_validated": 0,
            "spatially_resolved": 0,
            "confidence_boosted": 0,
        }