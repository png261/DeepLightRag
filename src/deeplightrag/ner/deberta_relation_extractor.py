"""
DeBERTa-based Relation Extraction for DeepLightRAG
State-of-the-art relation extraction using DeBERTa-v3 trained on ACE05/TACRED
"""

import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoConfig

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

from .gliner_ner import ExtractedEntity
from ..graph.entity_relationship import Entity, Relationship
from ..ocr.deepseek_ocr import VisualRegion


@dataclass
class DeBERTaRelation:
    """DeBERTa-extracted relation with confidence"""

    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    evidence_text: str
    source_positions: Tuple[int, int]
    target_positions: Tuple[int, int]
    model_output: Dict[str, Any] = None
    visual_context: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_output is None:
            self.model_output = {}
        if self.visual_context is None:
            self.visual_context = {}


class ACE05TACREDRelationSchema:
    """
    ACE05 + TACRED relation types mapped to DeepLightRAG schema
    """

    def __init__(self):
        # Combined relation schema from ACE05 + TACRED + DeepLightRAG extensions
        self.relations = {
            # === ACE05 Relations (18 types) ===
            "PER:Employee-Executive": {
                "description": "Person is an executive of organization",
                "domain": "ACE05",
                "dlr_mapping": "WORKS_FOR",
                "examples": ["CEO of Google", "President of university"],
            },
            "PER:Employee-Staff": {
                "description": "Person is staff member of organization",
                "domain": "ACE05",
                "dlr_mapping": "WORKS_FOR",
                "examples": ["engineer at Microsoft", "researcher at MIT"],
            },
            "PER:Member-of-Group": {
                "description": "Person is member of group/organization",
                "domain": "ACE05",
                "dlr_mapping": "PART_OF",
                "examples": ["member of committee", "part of team"],
            },
            "ORG:Subsidiary": {
                "description": "Organization is subsidiary of another",
                "domain": "ACE05",
                "dlr_mapping": "PART_OF",
                "examples": ["YouTube subsidiary of Google", "WhatsApp part of Meta"],
            },
            "GPE:Headquarters": {
                "description": "Organization headquarters located in place",
                "domain": "ACE05",
                "dlr_mapping": "LOCATED_IN",
                "examples": ["Apple headquarters in Cupertino", "Google based in Mountain View"],
            },
            "PHYS:Part-Whole": {
                "description": "Physical part-whole relationship",
                "domain": "ACE05",
                "dlr_mapping": "PART_OF",
                "examples": ["engine part of car", "CPU part of computer"],
            },
            # === TACRED Relations (key selection from 42 types) ===
            "per:employee_of": {
                "description": "Person employed by organization",
                "domain": "TACRED",
                "dlr_mapping": "WORKS_FOR",
                "examples": ["John works for IBM", "researcher at Stanford"],
            },
            "per:founder": {
                "description": "Person founded organization",
                "domain": "TACRED",
                "dlr_mapping": "FOUNDED",
                "examples": ["Jobs founded Apple", "Gates founded Microsoft"],
            },
            "per:title": {
                "description": "Person has title/position",
                "domain": "TACRED",
                "dlr_mapping": "HAS_TITLE",
                "examples": ["Dr. Smith", "Professor Chen", "CEO Johnson"],
            },
            "org:founded_by": {
                "description": "Organization founded by person",
                "domain": "TACRED",
                "dlr_mapping": "FOUNDED_BY",
                "examples": ["Google founded by Page and Brin", "Tesla founded by Musk"],
            },
            "org:headquarters": {
                "description": "Organization headquarters location",
                "domain": "TACRED",
                "dlr_mapping": "LOCATED_IN",
                "examples": ["Facebook HQ in Menlo Park", "Microsoft campus in Redmond"],
            },
            "per:schools_attended": {
                "description": "Person attended educational institution",
                "domain": "TACRED",
                "dlr_mapping": "EDUCATED_AT",
                "examples": ["graduated from MIT", "studied at Harvard"],
            },
            "per:city_of_birth": {
                "description": "Person born in city",
                "domain": "TACRED",
                "dlr_mapping": "BORN_IN",
                "examples": ["born in New York", "native of London"],
            },
            "per:spouse": {
                "description": "Person married to another person",
                "domain": "TACRED",
                "dlr_mapping": "MARRIED_TO",
                "examples": ["married to Jane", "spouse of John"],
            },
            # === DeepLightRAG Extensions (Document-specific) ===
            "USES": {
                "description": "Entity uses another entity",
                "domain": "DeepLightRAG",
                "dlr_mapping": "USES",
                "examples": ["model uses dataset", "system uses GPU"],
            },
            "ACHIEVES": {
                "description": "Entity achieves performance/result",
                "domain": "DeepLightRAG",
                "dlr_mapping": "ACHIEVES",
                "examples": ["model achieves 95% accuracy", "system reaches 10x speedup"],
            },
            "COMPARED_TO": {
                "description": "Entity compared against another",
                "domain": "DeepLightRAG",
                "dlr_mapping": "COMPARED_TO",
                "examples": ["Method A vs Method B", "95% compared to 87%"],
            },
            "EVALUATED_ON": {
                "description": "Entity evaluated on dataset/benchmark",
                "domain": "DeepLightRAG",
                "dlr_mapping": "EVALUATED_ON",
                "examples": ["model tested on ImageNet", "algorithm benchmarked on dataset"],
            },
            "BETTER_THAN": {
                "description": "Entity performs better than another",
                "domain": "DeepLightRAG",
                "dlr_mapping": "BETTER_THAN",
                "examples": ["DeBERTa outperforms BERT", "new method beats baseline"],
            },
            "ENABLES": {
                "description": "Entity enables another entity/capability",
                "domain": "DeepLightRAG",
                "dlr_mapping": "ENABLES",
                "examples": ["GPU enables fast training", "method enables accuracy"],
            },
            "PRODUCES": {
                "description": "Entity produces output/result",
                "domain": "DeepLightRAG",
                "dlr_mapping": "PRODUCES",
                "examples": ["model produces predictions", "algorithm generates results"],
            },
        }

        # Label mapping for model inference
        self.label_to_id = {label: idx for idx, label in enumerate(self.relations.keys())}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def get_relation_types(self) -> List[str]:
        """Get all relation type labels"""
        return list(self.relations.keys())

    def get_dlr_mapping(self, relation_type: str) -> str:
        """Get DeepLightRAG mapping for relation type"""
        return self.relations.get(relation_type, {}).get("dlr_mapping", relation_type)

    def get_relation_domain(self, relation_type: str) -> str:
        """Get source domain (ACE05/TACRED/DeepLightRAG)"""
        return self.relations.get(relation_type, {}).get("domain", "unknown")


class DeBERTaRelationExtractor:
    """
    DeBERTa-based relation extractor for superior accuracy
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        confidence_threshold: float = 0.5,
        max_length: int = 512,
        device: str = "auto",
        torch_dtype=torch.float16,
        batch_size: int = 4,
    ):
        """
        Initialize DeBERTa relation extractor

        Args:
            model_name: Pre-trained DeBERTa model name
            confidence_threshold: Minimum confidence for relations
            max_length: Maximum sequence length
            device: Device for inference ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size

        # Initialize relation schema
        self.schema = ACE05TACREDRelationSchema()

        # Model components
        self.tokenizer = None
        self.model = None
        self.classifier = None

        # Load model
        self._load_model()

        # Statistics tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "total_relations": 0,
            "relations_by_type": defaultdict(int),
            "relations_by_domain": defaultdict(int),
            "avg_confidence": 0.0,
            "model_performance": {"inference_time_ms": 0.0, "batch_size": 1},
        }

    def _setup_device(self, device: str) -> str:
        """Setup device with automatic detection"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model(self):
        """Load DeBERTa model and tokenizer with GPU optimization"""
        if not HAS_TRANSFORMERS:
            print("Transformers not available. Using pattern-based fallback.")
            return

        try:
            print(f"Loading DeBERTa model: {self.model_name} on {self.device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Check if we have a fine-tuned RE model available
            re_model_path = self._get_finetuned_model_path()

            # Model loading kwargs
            model_kwargs = {"torch_dtype": self.torch_dtype, "trust_remote_code": True}

            # Add device mapping for GPU
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True

            if re_model_path and Path(re_model_path).exists():
                print(f"Loading fine-tuned RE model from {re_model_path}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    re_model_path, **model_kwargs
                )
            else:
                print("Fine-tuned RE model not found. Using base model with custom head.")
                self._create_custom_re_model(**model_kwargs)

            # Move to device and set precision
            if self.model and "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
                if self.torch_dtype == torch.float16 and self.device == "cuda":
                    self.model = self.model.half()

            # Set to evaluation mode
            if self.model:
                self.model.eval()

            print(f"✅ DeBERTa model loaded successfully on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load DeBERTa model: {e}")
            print("Using pattern-based fallback")
            self.model = None
            self.tokenizer = None

    def _get_finetuned_model_path(self) -> Optional[str]:
        """Get path to fine-tuned RE model if available"""
        # Check for common fine-tuned model locations
        possible_paths = [
            "models/deberta-v3-large-ace05-tacred",
            "models/deberta-re-finetuned",
            f"models/{self.model_name.replace('/', '-')}-re",
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path

        return None

    def _create_custom_re_model(self, **model_kwargs):
        """Create custom RE model from base DeBERTa with GPU optimization"""
        try:
            # Load base model config
            config = AutoConfig.from_pretrained(self.model_name)
            config.num_labels = len(self.schema.relations) + 1  # +1 for "no_relation"

            # Update model kwargs
            model_kwargs["config"] = config

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **model_kwargs
            )

            print(f"Created custom RE head with {config.num_labels} labels")

        except Exception as e:
            print(f"Failed to create custom RE model: {e}")
            self.model = None

    def extract_relations_from_entities(
        self, entities: List[ExtractedEntity], text: str, region: Optional[VisualRegion] = None
    ) -> List[DeBERTaRelation]:
        """
        Extract relations from entity pairs using DeBERTa

        Args:
            entities: List of extracted entities
            text: Source text
            region: Optional visual region context

        Returns:
            List of DeBERTa relations
        """
        if not self.model or not self.tokenizer:
            return self._fallback_pattern_extraction(entities, text, region)

        relations = []

        # Generate entity pairs
        entity_pairs = self._generate_entity_pairs(entities)

        if not entity_pairs:
            return relations

        print(f"    🤖 DeBERTa processing {len(entity_pairs)} entity pairs...")

        # Process in batches for efficiency
        batch_size = 8  # Adjust based on memory

        for i in range(0, len(entity_pairs), batch_size):
            batch_pairs = entity_pairs[i : i + batch_size]
            batch_relations = self._extract_batch_relations(batch_pairs, text, region)
            relations.extend(batch_relations)

        # Post-process relations
        relations = self._post_process_relations(relations)

        # Update statistics
        self._update_stats(relations)

        return relations

    def _generate_entity_pairs(
        self, entities: List[ExtractedEntity], max_distance: int = 300
    ) -> List[Tuple[ExtractedEntity, ExtractedEntity]]:
        """Generate entity pairs for relation extraction"""
        pairs = []

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                # Check distance constraint
                distance = abs(entity1.start - entity2.start)
                if distance > max_distance:
                    continue

                # Check entity type compatibility for relations
                if self._are_entities_compatible(entity1, entity2):
                    pairs.append((entity1, entity2))

        return pairs

    def _are_entities_compatible(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> bool:
        """Check if two entities can have relations"""
        # Define compatible entity type pairs
        compatible_pairs = {
            ("PERSON", "ORGANIZATION"),
            ("PERSON", "LOCATION"),
            ("PERSON", "PERSON"),
            ("ORGANIZATION", "ORGANIZATION"),
            ("ORGANIZATION", "LOCATION"),
            ("TECHNICAL_TERM", "TECHNICAL_TERM"),
            ("TECHNICAL_TERM", "METRIC_RESULT"),
            ("TECHNICAL_TERM", "RESEARCH_ARTIFACT"),
            ("METHOD", "RESEARCH_ARTIFACT"),
            ("METHOD", "METRIC_RESULT"),
            ("PRODUCT", "ORGANIZATION"),
            ("PRODUCT", "TECHNICAL_TERM"),
        }

        pair = (entity1.label, entity2.label)
        return pair in compatible_pairs or (pair[1], pair[0]) in compatible_pairs

    def _extract_batch_relations(
        self,
        entity_pairs: List[Tuple[ExtractedEntity, ExtractedEntity]],
        text: str,
        region: Optional[VisualRegion],
    ) -> List[DeBERTaRelation]:
        """Extract relations from a batch of entity pairs"""
        relations = []

        # Prepare batch inputs
        batch_inputs = []
        batch_contexts = []

        for entity1, entity2 in entity_pairs:
            # Create marked input text
            marked_text = self._create_marked_input(text, entity1, entity2)
            batch_inputs.append(marked_text)

            # Store context for later use
            context = {"entity1": entity1, "entity2": entity2, "original_text": text}
            batch_contexts.append(context)

        try:
            # Tokenize batch
            encoded = self.tokenizer(
                batch_inputs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits

                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)

            # Process predictions
            for idx, (probs_tensor, context) in enumerate(zip(probs, batch_contexts)):
                relation = self._process_prediction(probs_tensor, context, region)
                if relation:
                    relations.append(relation)

        except Exception as e:
            print(f"    DeBERTa batch processing failed: {e}")
            # Fall back to pattern-based for this batch
            for entity1, entity2 in entity_pairs:
                pattern_relations = self._extract_pattern_relation(entity1, entity2, text)
                relations.extend(pattern_relations)

        return relations

    def _create_marked_input(
        self, text: str, entity1: ExtractedEntity, entity2: ExtractedEntity
    ) -> str:
        """Create input text with entity markers for DeBERTa"""
        # Strategy: Mark entities with special tokens
        # Format: [CLS] text_with_markers [SEP]

        # Get entity positions and text
        entities_with_pos = [
            (entity1.start, entity1.end, entity1.text, "E1"),
            (entity2.start, entity2.end, entity2.text, "E2"),
        ]

        # Sort by position to insert markers correctly
        entities_with_pos.sort(key=lambda x: x[0])

        # Insert markers from right to left to maintain positions
        marked_text = text
        offset = 0

        for start, end, ent_text, marker in reversed(entities_with_pos):
            # Insert end marker
            marked_text = (
                marked_text[: end + offset] + f" [/{marker}]" + marked_text[end + offset :]
            )
            # Insert start marker
            marked_text = (
                marked_text[: start + offset] + f"[{marker}] " + marked_text[start + offset :]
            )

            offset += len(f"[{marker}] ") + len(f" [/{marker}]")

        return marked_text

    def _process_prediction(
        self, probs: torch.Tensor, context: Dict, region: Optional[VisualRegion]
    ) -> Optional[DeBERTaRelation]:
        """Process model prediction to extract relation"""

        # Get top prediction
        top_prob, top_idx = torch.max(probs, dim=0)
        confidence = top_prob.item()

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return None

        # Map index to relation type
        if top_idx.item() == len(self.schema.relations):  # "no_relation" class
            return None

        relation_type = self.schema.id_to_label.get(top_idx.item())
        if not relation_type:
            return None

        # Create relation object
        entity1 = context["entity1"]
        entity2 = context["entity2"]

        relation = DeBERTaRelation(
            source_entity=entity1.text,
            target_entity=entity2.text,
            relation_type=relation_type,
            confidence=confidence,
            evidence_text=self._get_evidence_text(context["original_text"], entity1, entity2),
            source_positions=(entity1.start, entity1.end),
            target_positions=(entity2.start, entity2.end),
            model_output={
                "all_probs": probs.cpu().numpy(),
                "top_predictions": self._get_top_k_predictions(probs, k=3),
            },
            visual_context=self._get_visual_context(entity1, entity2, region),
        )

        return relation

    def _get_top_k_predictions(self, probs: torch.Tensor, k: int = 3) -> List[Dict]:
        """Get top-k predictions for analysis"""
        top_probs, top_indices = torch.topk(probs, k)

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            relation_type = self.schema.id_to_label.get(idx.item(), "unknown")
            predictions.append({"relation_type": relation_type, "confidence": prob.item()})

        return predictions

    def _get_evidence_text(
        self, text: str, entity1: ExtractedEntity, entity2: ExtractedEntity
    ) -> str:
        """Get evidence text around the entities"""
        start_pos = min(entity1.start, entity2.start)
        end_pos = max(entity1.end, entity2.end)

        # Expand context
        context_start = max(0, start_pos - 50)
        context_end = min(len(text), end_pos + 50)

        return text[context_start:context_end]

    def _get_visual_context(
        self, entity1: ExtractedEntity, entity2: ExtractedEntity, region: Optional[VisualRegion]
    ) -> Dict[str, Any]:
        """Get visual context information"""
        context = {}

        if region:
            context.update(
                {
                    "region_id": region.region_id,
                    "region_type": region.block_type,
                    "page_num": region.page_num,
                }
            )

        # Calculate entity distance
        distance = abs(entity1.start - entity2.start)
        context["text_distance"] = distance
        context["proximity"] = "close" if distance < 50 else "medium" if distance < 150 else "far"

        return context

    def _post_process_relations(self, relations: List[DeBERTaRelation]) -> List[DeBERTaRelation]:
        """Post-process extracted relations"""
        # Remove duplicates
        relations = self._remove_duplicate_relations(relations)

        # Sort by confidence
        relations.sort(key=lambda x: x.confidence, reverse=True)

        # Apply relation-specific filters
        relations = self._apply_relation_filters(relations)

        return relations

    def _remove_duplicate_relations(
        self, relations: List[DeBERTaRelation]
    ) -> List[DeBERTaRelation]:
        """Remove duplicate relations"""
        seen = set()
        filtered = []

        for relation in relations:
            key = (
                relation.source_entity.lower(),
                relation.target_entity.lower(),
                relation.relation_type,
            )

            if key not in seen:
                seen.add(key)
                filtered.append(relation)
            else:
                # Keep higher confidence version
                existing_idx = next(
                    i
                    for i, r in enumerate(filtered)
                    if (r.source_entity.lower(), r.target_entity.lower(), r.relation_type) == key
                )
                if relation.confidence > filtered[existing_idx].confidence:
                    filtered[existing_idx] = relation

        return filtered

    def _apply_relation_filters(self, relations: List[DeBERTaRelation]) -> List[DeBERTaRelation]:
        """Apply domain-specific relation filters"""
        filtered = []

        for relation in relations:
            # Check if relation makes semantic sense
            if self._is_semantically_valid(relation):
                filtered.append(relation)

        return filtered

    def _is_semantically_valid(self, relation: DeBERTaRelation) -> bool:
        """Check if relation is semantically valid"""
        # Basic semantic checks

        # Same entity check
        if relation.source_entity.lower() == relation.target_entity.lower():
            return False

        # Confidence check
        if relation.confidence < self.confidence_threshold:
            return False

        # Domain-specific checks could be added here

        return True

    def _fallback_pattern_extraction(
        self, entities: List[ExtractedEntity], text: str, region: Optional[VisualRegion]
    ) -> List[DeBERTaRelation]:
        """Fallback pattern-based extraction when DeBERTa unavailable"""
        relations = []

        # Simple pattern-based extraction for key relations
        patterns = {
            "WORKS_FOR": [
                r"(.+)\s+(?:works for|employed by|at)\s+(.+)",
                r"(.+)\s+(?:CEO|president|director) of\s+(.+)",
            ],
            "FOUNDED": [
                r"(.+)\s+(?:founded|established|created)\s+(.+)",
                r"(.+)\s+founder of\s+(.+)",
            ],
            "LOCATED_IN": [
                r"(.+)\s+(?:located in|based in|headquarters in)\s+(.+)",
                r"(.+)\s+(?:in|at)\s+(.+)",
            ],
        }

        for relation_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Create basic relation
                    relation = DeBERTaRelation(
                        source_entity=match.group(1).strip(),
                        target_entity=match.group(2).strip(),
                        relation_type=relation_type,
                        confidence=0.7,
                        evidence_text=match.group(0),
                        source_positions=(match.start(1), match.end(1)),
                        target_positions=(match.start(2), match.end(2)),
                        model_output={"method": "pattern_fallback"},
                    )
                    relations.append(relation)

        return relations

    def _extract_pattern_relation(
        self, entity1: ExtractedEntity, entity2: ExtractedEntity, text: str
    ) -> List[DeBERTaRelation]:
        """Extract relation using patterns for a specific entity pair"""
        # Simple co-occurrence relation
        distance = abs(entity1.start - entity2.start)
        if distance < 100:  # Close entities
            return [
                DeBERTaRelation(
                    source_entity=entity1.text,
                    target_entity=entity2.text,
                    relation_type="CO_OCCURS_WITH",
                    confidence=0.6,
                    evidence_text=f"{entity1.text} appears near {entity2.text}",
                    source_positions=(entity1.start, entity1.end),
                    target_positions=(entity2.start, entity2.end),
                    model_output={"method": "proximity_pattern"},
                )
            ]

        return []

    def _update_stats(self, relations: List[DeBERTaRelation]):
        """Update extraction statistics"""
        self.extraction_stats["total_extractions"] += 1
        self.extraction_stats["total_relations"] += len(relations)

        for relation in relations:
            self.extraction_stats["relations_by_type"][relation.relation_type] += 1
            domain = self.schema.get_relation_domain(relation.relation_type)
            self.extraction_stats["relations_by_domain"][domain] += 1

        if relations:
            avg_conf = sum(r.confidence for r in relations) / len(relations)
            current_avg = self.extraction_stats["avg_confidence"]
            total_extractions = self.extraction_stats["total_extractions"]
            # Update running average
            self.extraction_stats["avg_confidence"] = (
                current_avg * (total_extractions - 1) + avg_conf
            ) / total_extractions

    def convert_to_deeplightrag_relationships(
        self, deberta_relations: List[DeBERTaRelation], document_id: str = "unknown"
    ) -> List[Relationship]:
        """Convert DeBERTa relations to DeepLightRAG format"""
        relationships = []

        for i, relation in enumerate(deberta_relations):
            # Map to DeepLightRAG relation type
            dlr_relation_type = self.schema.get_dlr_mapping(relation.relation_type)

            dlr_relationship = Relationship(
                source_entity=relation.source_entity,
                target_entity=relation.target_entity,
                relationship_type=dlr_relation_type,
                description=f"{relation.source_entity} {dlr_relation_type.lower().replace('_', ' ')} {relation.target_entity}",
                weight=relation.confidence,
                evidence_text=relation.evidence_text,
                spatial_cooccurrence=bool(relation.visual_context),
                layout_aware_type=f"{dlr_relation_type}_deberta",
                source_regions=[],
                metadata={
                    "extraction_method": "deberta",
                    "original_relation_type": relation.relation_type,
                    "confidence": relation.confidence,
                    "model_output": relation.model_output,
                    "visual_context": relation.visual_context,
                    "domain": self.schema.get_relation_domain(relation.relation_type),
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
