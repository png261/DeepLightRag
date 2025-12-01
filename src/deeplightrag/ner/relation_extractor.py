"""
Advanced Relation Extraction for DeepLightRAG
OpenNRE-based relation extraction with custom DeepLightRAG relation schema
"""

import re
from typing import Dict, List, Tuple, Optional, Any, Set, TYPE_CHECKING
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import json
from pathlib import Path



try:
    import opennre

    HAS_OPENNRE = True
except ImportError:
    HAS_OPENNRE = False
    print("Warning: OpenNRE not installed. Install with: pip install opennre")

from .gliner_ner import ExtractedEntity
from ..graph.entity_relationship import Entity, Relationship
from ..ocr.deepseek_ocr import VisualRegion


@dataclass
class ExtractedRelation:
    """Extracted relation with metadata"""

    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    evidence_text: str
    source_positions: Tuple[int, int]  # start, end
    target_positions: Tuple[int, int]  # start, end
    visual_context: Dict[str, Any] = None
    extraction_method: str = "opennre"

    def __post_init__(self):
        if self.visual_context is None:
            self.visual_context = {}


class DeepLightRAGRelationSchema:
    """
    Custom relation schema optimized for document understanding
    """

    def __init__(self):
        self.relations = {
            # Spatial Relations (from visual layout)
            "LOCATED_IN": {
                "description": "Spatial containment or location relationship",
                "examples": ["New York located in USA", "Table 1 located in Section 2"],
                "patterns": [r"(.+)\s+(?:in|within|inside|located in)\s+(.+)"],
                "visual_context": True,
                "confidence_factors": ["spatial_distance", "layout_hierarchy"],
                "opennre_mapping": "per:city_of_birth",  # Closest OpenNRE relation
            },
            "ADJACENT_TO": {
                "description": "Physical or visual adjacency",
                "examples": ["Figure 1 adjacent to caption", "Header adjacent to paragraph"],
                "patterns": [r"(.+)\s+(?:adjacent to|next to|beside)\s+(.+)"],
                "visual_context": True,
                "confidence_factors": ["spatial_proximity", "visual_alignment"],
                "opennre_mapping": "org:located_in",
            },
            # Hierarchical Relations
            "PART_OF": {
                "description": "Part-whole relationships",
                "examples": ["Chapter 1 part of book", "Method part of algorithm"],
                "patterns": [r"(.+)\s+(?:part of|component of|belongs to)\s+(.+)"],
                "visual_context": False,
                "confidence_factors": ["textual_hierarchy", "semantic_containment"],
                "opennre_mapping": "org:subsidiaries",
            },
            "CONSISTS_OF": {
                "description": "Composition relationships",
                "examples": ["Algorithm consists of steps", "System consists of components"],
                "patterns": [r"(.+)\s+(?:consists of|comprises|contains|includes)\s+(.+)"],
                "visual_context": False,
                "confidence_factors": ["enumeration_patterns", "structural_indicators"],
                "opennre_mapping": "org:founded_by",
            },
            # Functional Relations
            "USES": {
                "description": "Usage or application relationship",
                "examples": ["Model uses dataset", "System uses GPU"],
                "patterns": [r"(.+)\s+(?:uses|utilizes|employs|applies)\s+(.+)"],
                "visual_context": False,
                "confidence_factors": ["action_verbs", "technical_context"],
                "opennre_mapping": "per:employee_of",
            },
            "PRODUCES": {
                "description": "Output or generation relationship",
                "examples": ["Model produces predictions", "Process produces results"],
                "patterns": [r"(.+)\s+(?:produces|generates|creates|outputs|yields)\s+(.+)"],
                "visual_context": False,
                "confidence_factors": ["causal_indicators", "output_patterns"],
                "opennre_mapping": "org:founded",
            },
            "ENABLES": {
                "description": "Enablement or facilitation",
                "examples": ["Technology enables innovation", "Method enables accuracy"],
                "patterns": [r"(.+)\s+(?:enables|facilitates|allows|permits)\s+(.+)"],
                "visual_context": False,
                "confidence_factors": ["causal_language", "benefit_indicators"],
                "opennre_mapping": "per:cause_of_death",
            },
            # Comparative Relations
            "COMPARED_TO": {
                "description": "Comparison relationships",
                "examples": ["Method A compared to Method B", "95% vs 87% accuracy"],
                "patterns": [r"(.+)\s+(?:compared to|versus|vs\.?|against)\s+(.+)"],
                "visual_context": True,
                "confidence_factors": ["comparison_indicators", "metric_proximity"],
                "opennre_mapping": "per:alternate_names",
            },
            "BETTER_THAN": {
                "description": "Performance superiority",
                "examples": ["New model better than baseline", "Accuracy improved over previous"],
                "patterns": [r"(.+)\s+(?:better than|superior to|outperforms|exceeds)\s+(.+)"],
                "visual_context": True,
                "confidence_factors": ["performance_metrics", "improvement_language"],
                "opennre_mapping": "per:other_family",
            },
            # Temporal Relations
            "FOLLOWS": {
                "description": "Temporal or procedural sequence",
                "examples": ["Step 2 follows Step 1", "2024 results follow 2023 baseline"],
                "patterns": [r"(.+)\s+(?:follows|comes after|succeeds)\s+(.+)"],
                "visual_context": True,
                "confidence_factors": ["sequence_indicators", "temporal_markers"],
                "opennre_mapping": "per:date_of_birth",
            },
            # Domain-Specific Relations
            "EVALUATED_ON": {
                "description": "Evaluation relationship",
                "examples": ["Model evaluated on dataset", "Method tested on benchmark"],
                "patterns": [r"(.+)\s+(?:evaluated on|tested on|benchmarked on)\s+(.+)"],
                "visual_context": False,
                "confidence_factors": ["evaluation_verbs", "dataset_mentions"],
                "opennre_mapping": "per:schools_attended",
            },
            "ACHIEVES": {
                "description": "Performance achievement",
                "examples": ["Model achieves 95% accuracy", "System achieves 10x speedup"],
                "patterns": [r"(.+)\s+(?:achieves|attains|reaches|obtains)\s+(.+)"],
                "visual_context": True,
                "confidence_factors": ["performance_claims", "metric_associations"],
                "opennre_mapping": "per:title",
            },
            # Co-occurrence Relations
            "CO_OCCURS_WITH": {
                "description": "Frequent co-occurrence in same context",
                "examples": ["Machine learning co-occurs with AI", "GPU co-occurs with training"],
                "patterns": [r"(.+)\s+(?:co-occurs with|appears with|found with)\s+(.+)"],
                "visual_context": True,
                "confidence_factors": ["spatial_proximity", "textual_frequency"],
                "opennre_mapping": "per:siblings",
            },
        }

    def get_relation_types(self) -> List[str]:
        """Get all relation type labels"""
        return list(self.relations.keys())

    def get_opennre_mapping(self, relation_type: str) -> Optional[str]:
        """Get OpenNRE relation mapping"""
        return self.relations.get(relation_type, {}).get("opennre_mapping")

    def get_patterns_for_relation(self, relation_type: str) -> List[str]:
        """Get regex patterns for a relation type"""
        return self.relations.get(relation_type, {}).get("patterns", [])


class OpenNREExtractor:
    """
    OpenNRE-based relation extractor for DeepLightRAG
    """

    def __init__(
        self,
        model_name: str = "wiki80_bert_softmax",
        confidence_threshold: float = 0.3,
        max_distance: int = 200,  # Max character distance between entities
        device: str = "cpu",
    ):
        """
        Initialize OpenNRE extractor

        Args:
            model_name: OpenNRE model name
            confidence_threshold: Minimum confidence for relation extraction
            max_distance: Maximum distance between entities to consider
            device: Device for inference
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_distance = max_distance
        self.device = device

        # Initialize relation schema
        self.schema = DeepLightRAGRelationSchema()

        # Load model
        self.model = None
        self._load_model()

        # Performance tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "total_relations": 0,
            "relations_by_type": defaultdict(int),
            "avg_confidence": 0.0,
        }

    def _load_model(self):
        """Load OpenNRE model"""
        if not HAS_OPENNRE:
            print("OpenNRE not available. Using pattern-based extractor.")
            return

        try:
            print(f"Loading OpenNRE model: {self.model_name}")
            self.model = opennre.get_model(self.model_name)
            print("OpenNRE model loaded successfully")
        except Exception as e:
            print(f"Failed to load OpenNRE model: {e}")
            print("Using pattern-based fallback")
            self.model = None

    def extract_relations_from_entities(
        self, entities: List[ExtractedEntity], text: str, region: Optional[VisualRegion] = None
    ) -> List[ExtractedRelation]:
        """
        Extract relations from entity pairs

        Args:
            entities: List of extracted entities
            text: Source text
            region: Optional visual region for spatial context

        Returns:
            List of extracted relations
        """
        relations = []

        # Generate entity pairs within distance threshold
        entity_pairs = self._generate_entity_pairs(entities, text)

        print(f"    🔗 Processing {len(entity_pairs)} entity pairs for relations...")

        # Extract relations using OpenNRE + patterns
        for pair in entity_pairs:
            pair_relations = self._extract_relation_from_pair(pair, text, region)
            relations.extend(pair_relations)

        # Post-process relations
        relations = self._post_process_relations(relations, entities, text)

        # Update statistics
        self._update_stats(relations)

        return relations

    def _generate_entity_pairs(
        self, entities: List[ExtractedEntity], text: str
    ) -> List[Tuple[ExtractedEntity, ExtractedEntity, int]]:
        """
        Generate valid entity pairs with distance filtering

        Returns:
            List of (entity1, entity2, distance) tuples
        """
        pairs = []

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                # Calculate distance between entities
                distance = abs(entity1.start - entity2.start)

                # Skip if too far apart
                if distance > self.max_distance:
                    continue

                # Skip if same entity type (in most cases)
                if entity1.label == entity2.label:
                    # Allow only for certain types that can relate to each other
                    if entity1.label not in ["TECHNICAL_TERM", "CONCEPT", "METHOD"]:
                        continue

                pairs.append((entity1, entity2, distance))

        # Sort by distance (closer pairs first)
        pairs.sort(key=lambda x: x[2])

        return pairs

    def _extract_relation_from_pair(
        self,
        pair: Tuple[ExtractedEntity, ExtractedEntity, int],
        text: str,
        region: Optional[VisualRegion] = None,
    ) -> List[ExtractedRelation]:
        """Extract relation from an entity pair"""
        entity1, entity2, distance = pair
        relations = []

        # Get text context around entities
        context_start = max(0, min(entity1.start, entity2.start) - 50)
        context_end = min(len(text), max(entity1.end, entity2.end) + 50)
        context = text[context_start:context_end]

        # Try OpenNRE extraction first
        if self.model:
            opennre_relations = self._extract_with_opennre(entity1, entity2, context)
            relations.extend(opennre_relations)

        # Try pattern-based extraction
        pattern_relations = self._extract_with_patterns(entity1, entity2, context, text)
        relations.extend(pattern_relations)

        # Try visual-spatial extraction if region available
        if region:
            spatial_relations = self._extract_spatial_relations(entity1, entity2, region)
            relations.extend(spatial_relations)

        # LLM fallback if no relations found and LLM is available
        if len(relations) == 0 and self.llm is not None:
            llm_relations = self._extract_with_llm(entity1, entity2, context, region)
            relations.extend(llm_relations)

        return relations

    def _extract_with_opennre(
        self, entity1: ExtractedEntity, entity2: ExtractedEntity, context: str
    ) -> List[ExtractedRelation]:
        """Extract relations using OpenNRE model"""
        relations = []

        try:
            # Prepare input for OpenNRE
            opennre_input = {
                "text": context,
                "h": {"name": entity1.text, "pos": [0, len(entity1.text)]},  # Simplified positions
                "t": {"name": entity2.text, "pos": [0, len(entity2.text)]},
            }

            # Get prediction
            result = self.model.infer(opennre_input)

            if result and result["score"] > self.confidence_threshold:
                # Map OpenNRE relation to our schema
                opennre_relation = result["relation"]
                mapped_relation = self._map_opennre_to_schema(opennre_relation)

                if mapped_relation:
                    relation = ExtractedRelation(
                        source_entity=entity1.text,
                        target_entity=entity2.text,
                        relation_type=mapped_relation,
                        confidence=result["score"],
                        evidence_text=context,
                        source_positions=(entity1.start, entity1.end),
                        target_positions=(entity2.start, entity2.end),
                        extraction_method="opennre",
                    )
                    relations.append(relation)

        except Exception as e:
            print(f"    OpenNRE extraction failed: {e}")

        return relations

    def _extract_with_patterns(
        self, entity1: ExtractedEntity, entity2: ExtractedEntity, context: str, full_text: str
    ) -> List[ExtractedRelation]:
        """Extract relations using regex patterns"""
        relations = []

        # Try each relation type's patterns
        for relation_type, relation_info in self.schema.relations.items():
            patterns = relation_info.get("patterns", [])

            for pattern in patterns:
                try:
                    # Try both entity orders
                    for e1, e2 in [(entity1, entity2), (entity2, entity1)]:
                        # Create pattern with entity placeholders
                        entity_pattern = pattern.replace("(.+)", f"({re.escape(e1.text)})")
                        entity_pattern = entity_pattern.replace(
                            "(.+)", f"({re.escape(e2.text)})", 1
                        )

                        match = re.search(entity_pattern, context, re.IGNORECASE)
                        if match:
                            confidence = 0.8  # High confidence for pattern matches

                            relation = ExtractedRelation(
                                source_entity=e1.text,
                                target_entity=e2.text,
                                relation_type=relation_type,
                                confidence=confidence,
                                evidence_text=match.group(0),
                                source_positions=(e1.start, e1.end),
                                target_positions=(e2.start, e2.end),
                                extraction_method="pattern",
                            )
                            relations.append(relation)
                            break  # Found relation, stop trying patterns

                except Exception as e:
                    continue  # Skip problematic patterns

        return relations

    def _extract_spatial_relations(
        self, entity1: ExtractedEntity, entity2: ExtractedEntity, region: VisualRegion
    ) -> List[ExtractedRelation]:
        """Extract spatial relations based on visual layout"""
        relations = []

        # Get entity positions in region
        e1_region_id = entity1.metadata.get("region_id")
        e2_region_id = entity2.metadata.get("region_id")

        # Same region - check for adjacency
        if e1_region_id == e2_region_id == region.region_id:
            # Calculate text distance
            text_distance = abs(entity1.start - entity2.start)

            # Very close entities might be adjacent
            if text_distance < 50:
                relation = ExtractedRelation(
                    source_entity=entity1.text,
                    target_entity=entity2.text,
                    relation_type="ADJACENT_TO",
                    confidence=0.6,
                    evidence_text=f"{entity1.text} and {entity2.text} appear close together",
                    source_positions=(entity1.start, entity1.end),
                    target_positions=(entity2.start, entity2.end),
                    visual_context={
                        "text_distance": text_distance,
                        "region_type": region.block_type,
                        "spatial_proximity": "high",
                    },
                    extraction_method="spatial",
                )
                relations.append(relation)

        # Different regions - check for hierarchical relations
        elif region.block_type in ["header", "title"]:
            # Headers might contain or describe entities in subsequent regions
            relation = ExtractedRelation(
                source_entity=entity1.text,
                target_entity=entity2.text,
                relation_type="RELATED_TO",
                confidence=0.4,
                evidence_text=f"{entity1.text} appears in header context with {entity2.text}",
                source_positions=(entity1.start, entity1.end),
                target_positions=(entity2.start, entity2.end),
                visual_context={
                    "region_type": region.block_type,
                    "hierarchical_context": "header_content",
                },
                extraction_method="spatial",
            )
            relations.append(relation)

        return relations

    def _extract_with_llm(
        self,
        entity1: ExtractedEntity,
        entity2: ExtractedEntity,
        context: str,
        region: Optional[VisualRegion] = None,
    ) -> List[ExtractedRelation]:
        """Extract relations using LLM as fallback"""
        relations = []

        try:
            # Create LLM prompt for relation extraction
            prompt = f"""
Analyze the following text to identify the relationship between two specific entities.

Context: "{context}"

Entity 1: "{entity1.text}" (Type: {entity1.label})
Entity 2: "{entity2.text}" (Type: {entity2.label})

Task: Determine if there is a meaningful relationship between these entities in the given context.

Possible relationship types:
- RELATED_TO: General semantic relation
- PART_OF: Entity1 is part of Entity2
- INFLUENCES: Entity1 affects or influences Entity2
- COMPARED_TO: Entities are compared or contrasted
- MENTIONS: Entity1 mentions or references Entity2
- CONTAINS: Entity1 contains or includes Entity2
- DEPENDS_ON: Entity1 depends on or requires Entity2
- USES: Entity1 uses or employs Entity2
- CAUSES: Entity1 causes or leads to Entity2

Respond with JSON format:
{{
  "has_relationship": true/false,
  "relationship_type": "TYPE_NAME",
  "confidence": 0.0-1.0,
  "explanation": "brief explanation"
}}

If no clear relationship exists, set has_relationship to false.
"""

            # Call LLM with low temperature for consistency
            response = self.llm.generate(prompt, temperature=0.2, max_tokens=256)

            # Parse LLM response
            import json

            try:
                result = json.loads(response.strip())

                if result.get("has_relationship", False):
                    rel_type = result.get("relationship_type", "RELATED_TO")
                    confidence = float(result.get("confidence", 0.5))
                    explanation = result.get("explanation", "")

                    # Validate relationship type
                    valid_types = {
                        "RELATED_TO",
                        "PART_OF",
                        "INFLUENCES",
                        "COMPARED_TO",
                        "MENTIONS",
                        "CONTAINS",
                        "DEPENDS_ON",
                        "USES",
                        "CAUSES",
                    }

                    if rel_type not in valid_types:
                        rel_type = "RELATED_TO"

                    # Only accept high-confidence predictions
                    if confidence >= 0.6:
                        relation = ExtractedRelation(
                            source_entity=entity1.text,
                            target_entity=entity2.text,
                            relation_type=rel_type,
                            confidence=confidence,
                            evidence_text=context,
                            source_positions=(entity1.start, entity1.end),
                            target_positions=(entity2.start, entity2.end),
                            extraction_method="llm_fallback",
                            visual_context={
                                "region_type": region.block_type if region else "unknown",
                                "llm_explanation": explanation,
                            },
                        )
                        relations.append(relation)

            except json.JSONDecodeError:
                print(
                    f"    ⚠️ LLM returned invalid JSON for entities: {entity1.text}, {entity2.text}"
                )

        except Exception as e:
            print(f"    ❌ LLM relation extraction failed: {e}")

        return relations

    def _map_opennre_to_schema(self, opennre_relation: str) -> Optional[str]:
        """Map OpenNRE relation to DeepLightRAG schema"""
        # Reverse mapping from schema to find matching relation
        for dlr_relation, info in self.schema.relations.items():
            if info.get("opennre_mapping") == opennre_relation:
                return dlr_relation

        # Fuzzy matching for unmapped relations
        relation_mappings = {
            "per:employee_of": "USES",
            "per:city_of_birth": "LOCATED_IN",
            "org:founded_by": "CONSISTS_OF",
            "per:cause_of_death": "CAUSES",
            "per:alternate_names": "COMPARED_TO",
            "org:subsidiaries": "PART_OF",
        }

        return relation_mappings.get(opennre_relation)

    def _post_process_relations(
        self, relations: List[ExtractedRelation], entities: List[ExtractedEntity], text: str
    ) -> List[ExtractedRelation]:
        """Post-process extracted relations"""
        # Remove duplicates
        relations = self._remove_duplicate_relations(relations)

        # Filter by confidence
        relations = [r for r in relations if r.confidence >= self.confidence_threshold]

        # Sort by confidence
        relations.sort(key=lambda x: x.confidence, reverse=True)

        return relations

    def _remove_duplicate_relations(
        self, relations: List[ExtractedRelation]
    ) -> List[ExtractedRelation]:
        """Remove duplicate relations"""
        seen = set()
        filtered = []

        for relation in relations:
            # Create unique key
            key = (relation.source_entity, relation.target_entity, relation.relation_type)

            if key not in seen:
                seen.add(key)
                filtered.append(relation)
            else:
                # Keep the one with higher confidence
                existing_idx = next(
                    i
                    for i, r in enumerate(filtered)
                    if (r.source_entity, r.target_entity, r.relation_type) == key
                )
                if relation.confidence > filtered[existing_idx].confidence:
                    filtered[existing_idx] = relation

        return filtered

    def _update_stats(self, relations: List[ExtractedRelation]):
        """Update extraction statistics"""
        self.extraction_stats["total_extractions"] += 1
        self.extraction_stats["total_relations"] += len(relations)

        for relation in relations:
            self.extraction_stats["relations_by_type"][relation.relation_type] += 1

        if relations:
            avg_conf = sum(r.confidence for r in relations) / len(relations)
            current_avg = self.extraction_stats["avg_confidence"]
            total_extractions = self.extraction_stats["total_extractions"]
            # Update running average
            self.extraction_stats["avg_confidence"] = (
                current_avg * (total_extractions - 1) + avg_conf
            ) / total_extractions

    def convert_to_deeplightrag_relationships(
        self, extracted_relations: List[ExtractedRelation], document_id: str = "unknown"
    ) -> List[Relationship]:
        """Convert extracted relations to DeepLightRAG Relationship objects"""
        relationships = []

        for i, relation in enumerate(extracted_relations):
            # Create relationship ID
            rel_id = f"{document_id}_rel_{i}_{relation.relation_type.lower()}"

            # Create DeepLightRAG Relationship
            dlr_relationship = Relationship(
                source_entity=relation.source_entity,
                target_entity=relation.target_entity,
                relationship_type=relation.relation_type,
                description=f"{relation.source_entity} {relation.relation_type.lower().replace('_', ' ')} {relation.target_entity}",
                weight=relation.confidence,
                evidence_text=relation.evidence_text,
                spatial_cooccurrence=relation.visual_context is not None,
                layout_aware_type=f"{relation.relation_type}_{relation.extraction_method}",
                source_regions=[],  # Will be populated from entity metadata
                metadata={
                    "extraction_method": relation.extraction_method,
                    "confidence": relation.confidence,
                    "source_positions": relation.source_positions,
                    "target_positions": relation.target_positions,
                    "visual_context": relation.visual_context,
                },
            )

            relationships.append(dlr_relationship)

        return relationships

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return dict(self.extraction_stats)

    def get_supported_relations(self) -> Dict[str, Dict[str, Any]]:
        """Get supported relation types"""
        return self.schema.relations


class RelationExtractionPipeline:
    """
    High-level relation extraction pipeline
    """

    def __init__(
        self,
        opennre_extractor: Optional[OpenNREExtractor] = None,
        enable_visual_relations: bool = True,
        enable_cross_region_relations: bool = True,
    ):
        self.extractor = opennre_extractor or OpenNREExtractor()
        self.enable_visual_relations = enable_visual_relations
        self.enable_cross_region_relations = enable_cross_region_relations

    def process_entities_for_relations(
        self,
        entities: List[ExtractedEntity],
        regions: List[VisualRegion],
        document_id: str = "unknown",
    ) -> Dict[str, Any]:
        """Process entities to extract relations"""
        all_relations = []

        # Group entities by region
        entities_by_region = defaultdict(list)
        for entity in entities:
            region_id = entity.metadata.get("region_id", "unknown")
            entities_by_region[region_id].append(entity)

        # Extract intra-region relations
        for region_id, region_entities in entities_by_region.items():
            if len(region_entities) < 2:
                continue

            # Find corresponding region
            region = next((r for r in regions if r.region_id == region_id), None)
            if not region:
                continue

            print(f"    🔗 Extracting relations in region {region_id} ({region.block_type})")

            # Extract relations within this region
            region_relations = self.extractor.extract_relations_from_entities(
                entities=region_entities, text=region.text_content, region=region
            )

            all_relations.extend(region_relations)

        # Extract cross-region relations if enabled
        if self.enable_cross_region_relations:
            cross_relations = self._extract_cross_region_relations(entities, regions)
            all_relations.extend(cross_relations)

        # Convert to DeepLightRAG format
        relationships = self.extractor.convert_to_deeplightrag_relationships(
            extracted_relations=all_relations, document_id=document_id
        )

        return {
            "relations": all_relations,
            "relationships": relationships,
            "stats": {
                "total_relations": len(all_relations),
                "relations_by_type": dict(self.extractor.extraction_stats["relations_by_type"]),
                "avg_confidence": self.extractor.extraction_stats["avg_confidence"],
            },
        }

    def _extract_cross_region_relations(
        self, entities: List[ExtractedEntity], regions: List[VisualRegion]
    ) -> List[ExtractedRelation]:
        """Extract relations across different regions"""
        cross_relations = []

        # Simple cross-region relation: header entities relate to content entities
        header_entities = []
        content_entities = []

        for entity in entities:
            region_id = entity.metadata.get("region_id")
            region = next((r for r in regions if r.region_id == region_id), None)

            if region and region.block_type in ["header", "title"]:
                header_entities.append(entity)
            elif region and region.block_type in ["paragraph", "table", "figure"]:
                content_entities.append(entity)

        # Create relations between header and content entities
        for header_entity in header_entities:
            for content_entity in content_entities:
                if header_entity.label == content_entity.label:  # Same entity type
                    relation = ExtractedRelation(
                        source_entity=header_entity.text,
                        target_entity=content_entity.text,
                        relation_type="RELATED_TO",
                        confidence=0.5,
                        evidence_text=f"Cross-region relation: {header_entity.text} in header relates to {content_entity.text} in content",
                        source_positions=(header_entity.start, header_entity.end),
                        target_positions=(content_entity.start, content_entity.end),
                        visual_context={
                            "cross_region": True,
                            "source_region_type": "header",
                            "target_region_type": "content",
                        },
                        extraction_method="cross_region",
                    )
                    cross_relations.append(relation)

        return cross_relations
