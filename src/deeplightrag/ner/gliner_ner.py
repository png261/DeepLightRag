"""
GLiNER2-based Named Entity Recognition for DeepLightRAG
Pure GLiNER2 entity extraction - no fallbacks
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    from gliner2 import GLiNER2
    HAS_GLINER2 = True
except ImportError:
    try:
        from gliner import GLiNER
        # Use GLiNER as fallback for GLiNER2
        GLiNER2 = GLiNER 
        HAS_GLINER2 = False
        print("⚠️ GLiNER2 not found, falling back to GLiNER")
    except ImportError:
        raise ImportError("GLiNER/GLiNER2 not installed. Install with: pip install gliner")

# Import entity schema
from .entity_schema import EntitySchema
from .text_classifier import DocumentClassifier
from .entity_type_mapper import EntityTypeMapper


@dataclass
class ExtractedEntity:
    """Extracted entity with metadata"""

    text: str
    label: str
    start: int
    end: int
    confidence: float
    context: str = ""
    normalized_form: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.normalized_form:
            self.normalized_form = self.text.lower().strip()




class GLiNERExtractor:
    """
    GLiNER2 entity extractor with intelligent label selection
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_small-v2.1",
        device: str = "auto",
    ):
        """
        Initialize GLiNER2 extractor

        Args:
            model_name: GLiNER2 model name
            device: Device for inference (auto/cpu/cuda)
        """
        self.model_name = model_name
        self.device = device

        # Initialize components
        self.schema = EntitySchema()
        self.classifier = DocumentClassifier()

        # Load model with better error handling
        print(f"Loading GLiNER2 model: {self.model_name}")
        try:
            # Try loading without authentication requirements
            self.model = GLiNER2.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=False
            )
            if self.device and self.device != "auto":
                try:
                    self.model.to(self.device)
                    print(f"  Moved GLiNER2 model to {self.device}")
                except Exception as e:
                    print(f"  ⚠️ Failed to move model to {self.device}: {e}")

            print(f"✅ GLiNER2 model loaded successfully")
        except Exception as e:
            print(f"⚠️ Primary model loading failed: {e}")
            # Try alternative model
            try:
                alternative_model = "urchade/gliner_small-v2.1"
                print(f"Trying alternative model: {alternative_model}")
                self.model = GLiNER2.from_pretrained(alternative_model)
                print(f"✅ Alternative model loaded successfully")
            except Exception as e2:
                print(f"⚠️ Alternative model failed: {e2}")
                # Initialize without model for neural network fallback
                self.model = None
                print(f"📝 Will use neural network fallback extraction")

        # Performance tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "total_entities": 0,
            "total_relations": 0,
            "entities_by_type": defaultdict(int),
            "relations_by_type": defaultdict(int),
            "classification_stats": defaultdict(int),
            "unified_extractions": 0,
            "fallback_extractions": 0,
        }
        self.last_schema_results = None

    
    def batch_extract_entities_and_relationships(
        self, 
        texts: List[str], 
        entity_types: Optional[List[str]] = None, 
        region_types: Optional[List[str]] = None,
        batch_size: int = 8
    ) -> List[Tuple[List[ExtractedEntity], List[Dict[str, Any]]]]:
        """
        Batch extract entities and relationships using GLiNER2 for improved performance
        
        This method processes multiple texts in batches, significantly improving throughput
        compared to single-text processing.

        Args:
            texts: List of input texts from DeepSeek OCR
            entity_types: Specific entity types to extract (None for auto-detection)
            region_types: List of document region types (one per text, or None)
            batch_size: Number of texts to process in each batch

        Returns:
            List of tuples (entities, relationships) for each input text
        """
        if not texts:
            return []
        
        if region_types is None:
            region_types = ["general"] * len(texts)
        
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"  [GLiNER2 Batch] Processing {len(texts)} texts in {total_batches} batches...")
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            batch_region_types = region_types[batch_idx:batch_idx + batch_size]
            
            # Process batch
            batch_results = []
            for text, region_type in zip(batch_texts, batch_region_types):
                result = self.extract_entities_and_relationships(text, entity_types, region_type)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if (batch_idx // batch_size + 1) % 5 == 0:
                print(f"    Processed batch {batch_idx // batch_size + 1}/{total_batches}")
        
        return results

    def extract_entities_and_relationships(
        self, text: str, entity_types: Optional[List[str]] = None, region_type: str = "general"
    ) -> Tuple[List[ExtractedEntity], List[Dict[str, Any]]]:
        """
        Extract entities and relationships using GLiNER2's unified multi-task approach
        
        This method leverages GLiNER2's native capability to extract both entities
        and relationships in a single pass, which is 2-3x faster than separate calls.

        Args:
            text: Input text from DeepSeek OCR
            entity_types: Specific entity types to extract (None for auto-detection)
            region_type: Type of document region (for context)

        Returns:
            Tuple of (entities, relationships)
        """
        try:
            # Preprocess text to improve entity extraction
            processed_text = self._preprocess_text(text)

            # Optional: Classify document type for context-aware extraction
            doc_type_name = self._classify_document_type(processed_text)

            # Get relevant entity and relation labels based on document type
            entity_labels = entity_types or self._get_gliner_labels_for_document(doc_type_name)
            relation_labels = self._get_relation_labels_for_document(doc_type_name)

            # Build unified schema for multi-task extraction
            if hasattr(self.model, "create_schema"):
                schema = (self.model.create_schema()
                    .entities(entity_labels)
                    .relations(relation_labels)
                )

                # Single unified extraction call - entities AND relations together!
                result = self.model.extract(processed_text, schema)
            else:
                 # Standard GLiNER fallback (predict_entities)
                # print(f"DEBUG: Extracting from text length {len(processed_text)}")
                # print(f"DEBUG: Text sample: {processed_text[:100]}")
                # print(f"DEBUG: Labels: {entity_labels[:5]}...")
                _entities = self.model.predict_entities(processed_text, entity_labels)
                # print(f"DEBUG: Raw entities found: {len(_entities)}")
                result = {
                    "entities": {label: [] for label in entity_labels},
                    "relation_extraction": {}
                }
                # Group by label
                for ent in _entities:
                    label = ent['label']
                    if label not in result["entities"]:
                        result["entities"][label] = []
                    result["entities"][label].append(ent)
            
            # Extract entities from unified result
            entities = self._convert_entities_from_unified_result(
                result.get('entities', {}),
                text,  # Use original text for position finding
                region_type,
                doc_type_name,
                processed_text  # Pass processed text for reference
            )

            # Extract relationships from unified result
            relationships = self._convert_relationships_from_unified_result(
                result.get('relation_extraction', {}),
                entities,
                text,  # Use original text for context
                region_type,
                doc_type_name
            )

            # Remove duplicate entities
            entities = self._remove_duplicate_entities(entities)

            # Update statistics
            self._update_stats(entities, relationships)
            self.extraction_stats["classification_stats"][doc_type_name] += 1
            self.extraction_stats["unified_extractions"] += 1

            # print(f"    ✅ GLiNER unified extraction: {len(entities)} entities, {len(relationships)} relations")

            return entities, relationships

        except Exception as e:
            # Debug the actual error
            print(f"    ❌ GLiNER2 unified extraction failed: {str(e)}")
            print(f"    📝 Text length: {len(text)}, Sample: {repr(text[:100])}")

            # Fallback to entity-only extraction
            self.extraction_stats["fallback_extractions"] += 1
            entities = self._fallback_entity_extraction(text, entity_types, region_type)
            relationships = self._fallback_relationship_extraction(text, entities, region_type)
            self._update_stats(entities, relationships)
            return entities, relationships

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text to improve entity extraction

        Args:
            text: Raw text from OCR

        Returns:
            Preprocessed text optimized for GLiNER
        """
        import re

        # Clean up common OCR issues
        processed = text

        # Replace common OCR patterns that confuse entity extraction
        replacements = {
            # Fix spacing issues around punctuation
            r'\s+([.,;:!?)])': r'\1',
            r'([({])\s+': r'\1',
            # Fix multiple spaces
            r'\s{2,}': ' ',
            # Ensure proper spacing around bullets and numbers
            r'(\d+)\.(\S)': r'\1. \2',
            r'([•*])(\S)': r'\1 \2',
            # Normalize dashes
            r'–+': '-',
            r'—+': '-',
        }

        for pattern, replacement in replacements.items():
            processed = re.sub(pattern, replacement, processed)

        # Add markup for document structure to help GLiNER
        lines = processed.split('\n')
        marked_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Heuristic markup removed for generic data support
            marked_lines.append(line)

        # Rejoin with proper spacing
        processed = '\n'.join(marked_lines)

        return processed

    def _classify_document_type(self, text: str) -> str:
        """
        Classify document type using GLiNER2 text classification

        Args:
            text: Input text to classify

        Returns:
            Document type label
        """
        try:
            if hasattr(self.model, "classify_text"):
                classification_result = self.model.classify_text(
                    text,
                    {
                        "document_type": {
                            "labels": ["academic", "business", "technical", "legal", "medical", "news", "general"],
                            "multi_label": False,
                            "cls_threshold": 0.3
                        }
                    }
                )
                return classification_result.get("document_type", "general")
            return "general"
        except Exception as e:
            print(f"    ⚠️ Document classification failed, using 'general': {e}")
            return "general"

    def _convert_entities_from_unified_result(
        self,
        entities_dict: Dict[str, List],
        text: str,
        region_type: str,
        doc_type: str,
        processed_text: Optional[str] = None
    ) -> List[ExtractedEntity]:
        """
        Convert GLiNER2 unified result entities to ExtractedEntity objects

        Args:
            entities_dict: Entity dictionary from GLiNER2 result
            text: Original source text for position finding
            region_type: Document region type
            doc_type: Classified document type
            processed_text: Processed text that was actually fed to GLiNER

        Returns:
            List of ExtractedEntity objects
        """
        entities = []
        
        for label, entity_list in entities_dict.items():

            for entity_item in entity_list:
                # Handle both string and dict formats
                if isinstance(entity_item, str):
                    entity_text = entity_item
                    confidence = 0.8  # Default confidence
                elif isinstance(entity_item, dict):
                    entity_text = entity_item.get('text', entity_item.get('entity', ''))
                    confidence = entity_item.get('score', entity_item.get('confidence', 0.8))
                else:
                    continue

                # Clean entity text by removing markup
                clean_entity_text = entity_text
                if entity_text.startswith('[HEADING] '):
                    clean_entity_text = entity_text[10:]  # Remove [HEADING] prefix
                elif entity_text.startswith('[LIST] '):
                    clean_entity_text = entity_text[7:]   # Remove [LIST] prefix

                # Find entity position in original text
                entity_start = text.find(clean_entity_text)
                if entity_start == -1:
                    # Try with processed text if not found in original
                    entity_start = processed_text.find(clean_entity_text) if processed_text else -1

                    # If found in processed text, try to map back to original
                    if entity_start != -1 and processed_text:
                        # Simple heuristic: look for the entity near the same position
                        # in the original text (accounting for markup differences)
                        markup_length = len(entity_text) - len(clean_entity_text)
                        estimated_start = max(0, entity_start - markup_length)
                        entity_start = text.find(clean_entity_text, estimated_start, estimated_start + len(clean_entity_text) + 50)

                if entity_start == -1:
                    continue

                entity_end = entity_start + len(clean_entity_text)
                context = self._get_entity_context(text, entity_start, entity_end)

                # Initialize mapper if not already done
                if not hasattr(self, "_entity_type_mapper"):
                    self._entity_type_mapper = EntityTypeMapper()

                # Map entity type with context
                mapped_type = self._map_label_to_entity_type(label, clean_entity_text, context)

                # Get confidence boost based on entity characteristics
                confidence_boost = self._entity_type_mapper.get_confidence_boost(
                    clean_entity_text, mapped_type, context
                )

                entity = ExtractedEntity(
                    text=clean_entity_text,  # Use cleaned entity text
                    label=mapped_type,
                    start=entity_start,
                    end=entity_end,
                    confidence=min(confidence + confidence_boost, 1.0),  # Cap at 1.0
                    context=context,
                    metadata={
                        "region_type": region_type,
                        "original_label": label,
                        "extraction_method": "gliner2-unified",
                        "document_type": doc_type,
                        "confidence_boost": confidence_boost,
                    },
                )
                entities.append(entity)
        
        return entities

    def _convert_relationships_from_unified_result(
        self,
        relations_dict: Dict[str, List[Tuple]],
        entities: List[ExtractedEntity],
        text: str,
        region_type: str,
        doc_type: str
    ) -> List[Dict[str, Any]]:
        """
        Convert GLiNER2 unified result relations to relationship dictionaries
        
        Args:
            relations_dict: Relations dictionary from GLiNER2 result
            entities: List of extracted entities
            text: Source text
            region_type: Document region type
            doc_type: Classified document type
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        for relation_type, entity_pairs in relations_dict.items():
            # Map to DeepLightRAG relation type
            mapped_relation = self._map_relation_label(relation_type)
            
            for pair in entity_pairs:
                if len(pair) < 2:
                    continue
                
                source_text = pair[0]
                target_text = pair[1]
                
                # Get confidence if provided (GLiNER2 may include it)
                confidence = pair[2] if len(pair) > 2 else 0.75
                
                # Find matching entities
                source_entity = self._find_matching_entity(source_text, entities)
                target_entity = self._find_matching_entity(target_text, entities)
                
                if source_entity and target_entity:
                    relationships.append({
                        'source_entity': source_entity.text,
                        'target_entity': target_entity.text,
                        'relation_type': mapped_relation,
                        'confidence': confidence,
                        'context': self._get_relation_context(text, source_entity, target_entity),
                        'source_entity_id': f"{source_entity.text}_{source_entity.start}",
                        'target_entity_id': f"{target_entity.text}_{target_entity.start}",
                        'metadata': {
                            'region_type': region_type,
                            'extraction_method': 'gliner2-unified',
                            'document_type': doc_type,
                            'original_relation_type': relation_type,
                        }
                    })
        
        return relationships

    def _get_gliner_labels_for_document(self, doc_type: str) -> List[str]:
        """Get GLiNER labels for document type - enhanced with more specific types"""
        label_mapping = {
            "academic": [
                "person", "organization", "location", "date", "concept", "method",
                "technical", "reference", "product", "technology", "framework",
                "model", "algorithm", "conference", "university", "research"
            ],
            "business": [
                "person", "organization", "location", "money", "product", "date",
                "event", "technology", "service", "platform", "company", "startup"
            ],
            "technical": [
                "person", "organization", "product", "technical", "method", "tool",
                "framework", "technology", "algorithm", "model", "system", "platform",
                "library", "api", "language", "software", "application"
            ],
            "legal": [
                "person", "organization", "location", "date", "document", "legal",
                "case", "court", "law", "regulation", "contract"
            ],
            "medical": [
                "person", "organization", "location", "date", "condition", "treatment",
                "medication", "procedure", "symptom", "disease", "hospital"
            ],
            "news": [
                "person", "organization", "location", "date", "event", "money",
                "product", "technology", "government", "country", "city"
            ],
        }

        return label_mapping.get(doc_type, self._get_comprehensive_labels())

    def _get_relation_labels_for_document(self, doc_type: str) -> List[str]:
        """
        Get relevant relation labels for document type
        These are used by GLiNER2's unified extraction
        """
        relation_mapping = {
            "academic": [
                "authored", "cites", "references", "builds_on", "improves", 
                "evaluates", "compares_to", "uses", "extends", "affiliated_with"
            ],
            "business": [
                "works_for", "founded", "acquired", "invests_in", "competes_with", 
                "partners_with", "reports_to", "located_in", "manages"
            ],
            "technical": [
                "uses", "depends_on", "implements", "extends", "configures", 
                "deploys", "requires", "validates", "produces", "enables"
            ],
            "legal": [
                "represented_by", "filed_by", "decided_by", "appealed_by", 
                "cites", "testified_in", "located_in"
            ],
            "medical": [
                "diagnosed_with", "treated_with", "prescribed_for", "tested_for", 
                "operated_by", "monitored_by", "causes", "prevents"
            ],
            "news": [
                "reported_by", "quoted_by", "mentioned_in", "located_at", 
                "happened_on", "commented_on", "works_for"
            ],
            "general": [
                "related_to", "associated_with", "part_of", "located_in", 
                "works_for", "founded", "uses", "produces"
            ]
        }

        return relation_mapping.get(doc_type, relation_mapping["general"])

    def _get_comprehensive_labels(self) -> List[str]:
        """Get comprehensive GLiNER labels for all entity types"""
        return [
            "person", "organization", "location", "date", "money", "product",
            "event", "concept", "method", "technical", "tool", "framework",
            "reference", "document", "legal", "condition", "treatment", "medication",
            "metric", "percentage", "country", "city", "company", "university",
            "technology", "system", "process", "algorithm", "model", "data"
        ]

    def extract_entities(
        self, text: str, entity_types: Optional[List[str]] = None, region_type: str = "general"
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text (wrapper for backward compatibility)
        """
        entities, _ = self.extract_entities_and_relationships(text, entity_types, region_type)
        return entities

    def extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        relation_types: Optional[List[str]] = None,
        region_type: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using GLiNER2 schema results

        Args:
            text: Input text containing entities
            entities: List of entities extracted from the text
            relation_types: Specific relation types to extract (None for auto-detection)
            region_type: Type of document region (for context)

        Returns:
            List of extracted relationships
        """
        relationships = []

        # If we have recent schema results, use them for relationship extraction
        if self.last_schema_results and 'relation_extraction' in self.last_schema_results:
            relation_results = self.last_schema_results['relation_extraction']

            for relation_type, entity_pairs in relation_results.items():
                mapped_relation = self._map_relation_label(relation_type)

                for pair in entity_pairs:
                    if len(pair) >= 2:
                        source_text = pair[0]
                        target_text = pair[1]

                        # Find matching entities
                        source_entity = self._find_matching_entity(source_text, entities)
                        target_entity = self._find_matching_entity(target_text, entities)

                        if source_entity and target_entity:
                            relationships.append({
                                'source_entity': source_entity.text,
                                'target_entity': target_entity.text,
                                'relation_type': mapped_relation,
                                'confidence': 0.80,  # Raised from 0.75 for better quality
                                'context': self._get_relation_context(text, source_entity, target_entity),
                                'source_entity_id': f"{source_entity.text}_{source_entity.start}",
                                'target_entity_id': f"{target_entity.text}_{target_entity.start}",
                                'metadata': {
                                    'region_type': region_type,
                                    'extraction_method': 'gliner2-schema',
                                    'document_type': self.last_schema_results.get('document_type', 'general')
                                }
                            })

        # Fallback: Extract relationships using GLiNER2 with targeted approach
        if not relationships and len(entities) >= 2:
            relationships = self._fallback_relationship_extraction(text, entities, region_type)

        return relationships

    def _get_relation_context(self, text: str, entity1: ExtractedEntity, entity2: ExtractedEntity, window: int = 100) -> str:
        """Get context between two entities"""
        start = min(entity1.start, entity2.start)
        end = max(entity1.end, entity2.end)

        # Expand window to include context
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        return text[context_start:context_end]

    def _get_relation_labels_for_document_type(self, doc_type: str) -> List[str]:
        """Get relevant relation labels for document type"""
        relation_mapping = {
            "academic": ["references", "cites", "builds_on", "improves", "evaluates", "compares_to", "uses", "extends"],
            "business": ["acquired_by", "invests_in", "competes_with", "partners_with", "reports_to", "located_in", "affiliated_with"],
            "technical": ["uses", "depends_on", "implements", "extends", "configures", "deploys", "requires", "validates"],
            "legal": ["represented_by", "against", "testified_in", "filed_by", "decided_by", "appealed_by", "cites"],
            "medical": ["diagnosed_with", "treated_with", "prescribed_for", "tested_for", "operated_by", "monitored_by"],
            "news": ["reported_by", "quoted_by", "mentioned_in", "located_at", "happened_on", "commented_on"],
        }

        return relation_mapping.get(doc_type, ["related_to", "associated_with", "connected_to"])

    def _map_relation_label(self, gliner_label: str) -> str:
        """
        Map GLiNER2 relation label to DeepLightRAG relation type
        Supports comprehensive relation types from GLiNER2
        """
        label_lower = gliner_label.lower().replace(" ", "_")

        # Comprehensive relation mappings
        relation_mapping = {
            # Core relations
            "is_a": "IS_A",
            "has_property": "HAS_PROPERTY",
            
            # Spatial relations
            "located_in": "LOCATED_IN",
            "located_at": "LOCATED_AT",
            "near": "NEAR",
            "adjacent_to": "ADJACENT_TO",
            
            # Hierarchical relations
            "part_of": "PART_OF",
            "contains": "CONTAINS",
            "consists_of": "CONSISTS_OF",
            "subclass_of": "SUBCLASS_OF",
            "instance_of": "INSTANCE_OF",
            
            # Functional relations
            "uses": "USES",
            "enables": "ENABLES",
            "produces": "PRODUCES",
            "generates": "GENERATES",
            "depends_on": "DEPENDS_ON",
            "requires": "REQUIRES",
            "supports": "SUPPORTS",
            
            # Semantic relations
            "describes": "DESCRIBES",
            "explains": "EXPLAINS",
            "defines": "DEFINES",
            "exemplifies": "EXEMPLIFIES",
            "represents": "REPRESENTS",
            
            # Comparative relations
            "compares_to": "COMPARED_TO",
            "similar_to": "SIMILAR_TO",
            "different_from": "DIFFERENT_FROM",
            
            # Temporal relations
            "before": "BEFORE",
            "after": "AFTER",
            "during": "DURING",
            
            # Causal relations
            "causes": "CAUSES",
            "prevents": "PREVENTS",
            "leads_to": "LEADS_TO",
            "results_in": "RESULTS_IN",
            
            # Document relations
            "references": "REFERENCES",
            "cites": "REFERENCES",
            "mentions": "MENTIONS",
            "discusses": "DISCUSSES",
            "introduces": "INTRODUCES",
            
            # Research relations
            "achieves": "ACHIEVES",
            "improves": "IMPROVES",
            "evaluates": "EVALUATES",
            "validates": "VALIDATES",
            "demonstrates": "DEMONSTRATES",
            
            # Organizational relations
            "works_for": "WORKS_FOR",
            "founded": "FOUNDED",
            "authored": "AUTHORED_BY",
            "created_by": "CREATED_BY",
            "affiliated_with": "AFFILIATED_WITH",
            "collaborates_with": "COLLABORATES_WITH",
            
            # Business relations
            "acquired": "ACQUIRED_BY",
            "invests_in": "INVESTS_IN",
            "competes_with": "COMPETES_WITH",
            "partners_with": "PARTNERS_WITH",
            "reports_to": "REPORTS_TO",
            "manages": "MANAGES",
            
            # Technical relations
            "implements": "IMPLEMENTS",
            "extends": "EXTENDS",
            "configures": "CONFIGURES",
            "deploys": "DEPLOYS",
            
            # Legal relations
            "represented_by": "REPRESENTED_BY",
            "filed_by": "FILED_BY",
            "decided_by": "DECIDED_BY",
            
            # Medical relations
            "diagnosed_with": "DIAGNOSED_WITH",
            "treated_with": "TREATED_WITH",
            "prescribed_for": "PRESCRIBED_FOR",
            
            # News relations
            "reported_by": "REPORTED_BY",
            "quoted_by": "QUOTED_BY",
            "mentioned_in": "MENTIONED_IN",
            "happened_on": "HAPPENED_ON",
        }

        return relation_mapping.get(label_lower, label_lower.upper())

    def _get_entity_description(self, entity_type: str) -> str:
        """Get description for entity type"""
        descriptions = {
            "PERSON": "Names of people, individuals, or persons",
            "ORGANIZATION": "Company names, organizations, institutions",
            "LOCATION": "Locations, places, addresses, geographical entities",
            "DATE_TIME": "Dates, times, periods, temporal expressions",
            "MONEY": "Monetary values, prices, costs, financial amounts",
            "PERCENTAGE": "Percentages, rates, proportions",
            "METRIC": "Measurements, quantities, metrics, numerical values",
            "TECHNICAL_TERM": "Technical terms, jargon, specialized vocabulary",
            "PRODUCT": "Products, services, offerings, items",
            "CONCEPT": "Concepts, ideas, abstract notions",
            "EVENT": "Events, meetings, conferences, occurrences",
            "DOCUMENT": "Documents, papers, reports, written materials"
        }
        return descriptions.get(entity_type, f"{entity_type} entity")

    def _get_entity_types_for_document(self, doc_type: str) -> Dict[str, str]:
        """Get relevant entity types for document type"""
        entity_mapping = {
            "academic": {
                "PERSON": "Researchers, authors, academics",
                "ORGANIZATION": "Universities, institutions, publishers",
                "LOCATION": "Conference venues, institutions",
                "DATE_TIME": "Publication dates, conference dates",
                "TECHNICAL_TERM": "Technical terms, methodologies",
                "CONCEPT": "Theories, concepts, research areas",
                "DOCUMENT": "Papers, publications, articles"
            },
            "business": {
                "PERSON": "Executives, employees, stakeholders",
                "ORGANIZATION": "Companies, corporations, businesses",
                "LOCATION": "Offices, branches, facilities",
                "MONEY": "Revenue, costs, financial figures",
                "DATE_TIME": "Fiscal periods, reporting dates",
                "PRODUCT": "Products, services, offerings",
                "EVENT": "Meetings, conferences, events"
            },
            "technical": {
                "PERSON": "Developers, engineers, designers",
                "ORGANIZATION": "Tech companies, open source projects",
                "TECHNICAL_TERM": "API names, frameworks, technologies",
                "PRODUCT": "Software, tools, platforms",
                "CONCEPT": "Architectures, patterns, methodologies",
                "DOCUMENT": "Documentation, specifications"
            },
            "legal": {
                "PERSON": "Judges, lawyers, parties",
                "ORGANIZATION": "Law firms, courts, institutions",
                "LOCATION": "Courts, jurisdictions, venues",
                "DATE_TIME": "Court dates, filing dates",
                "DOCUMENT": "Contracts, agreements, rulings",
                "EVENT": "Hearings, trials, proceedings"
            },
            "medical": {
                "PERSON": "Doctors, patients, researchers",
                "ORGANIZATION": "Hospitals, clinics, institutions",
                "LOCATION": "Medical facilities, locations",
                "DATE_TIME": "Treatment dates, appointment times",
                "METRIC": "Dosages, measurements, vital signs",
                "PRODUCT": "Medications, treatments, devices",
                "CONCEPT": "Conditions, symptoms, diagnoses"
            },
            "news": {
                "PERSON": "Public figures, officials, sources",
                "ORGANIZATION": "Governments, companies, organizations",
                "LOCATION": "Places, venues, locations",
                "DATE_TIME": "Event dates, publication dates",
                "EVENT": "Incidents, announcements, occurrences",
                "MONEY": "Financial figures, amounts"
            }
        }

        return entity_mapping.get(doc_type, {
            "PERSON": "People, individuals",
            "ORGANIZATION": "Organizations, companies",
            "LOCATION": "Locations, places",
            "CONCEPT": "Concepts, ideas"
        })

    def _find_matching_entity(self, text: str, entities: List[ExtractedEntity]) -> Optional[ExtractedEntity]:
        """Find entity that matches the given text"""
        text_lower = text.lower().strip()

        for entity in entities:
            if entity.text.lower().strip() == text_lower:
                return entity

        # Fuzzy matching for partial matches
        for entity in entities:
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity

        return None

    def _fallback_entity_extraction(self, text: str, entity_types: Optional[List[str]], region_type: str) -> List[ExtractedEntity]:
        """Neural network fallback extraction without rule-based patterns"""
        try:
            # Try using GLiNER if available
            if self.model is not None:
                # Use basic labels for fallback
                labels = ["person", "organization", "location", "date", "money", "product", "concept"]

                result = self.model.extract_entities(text, labels)
                entities_dict = result.get('entities', {})

                # Convert to ExtractedEntity objects
                entities = []
                for label, entity_list in entities_dict.items():
                    for entity_text in entity_list:
                        entity_start = text.find(entity_text)
                        if entity_start == -1:
                            continue

                        entity_end = entity_start + len(entity_text)
                        context = self._get_entity_context(text, entity_start, entity_end)
                        mapped_type = self._map_label_to_entity_type(label)

                        # Initialize mapper if not already done
                        if not hasattr(self, "_entity_type_mapper"):
                            self._entity_type_mapper = EntityTypeMapper()

                        # Get confidence boost
                        confidence_boost = self._entity_type_mapper.get_confidence_boost(
                            entity_text, mapped_type, context
                        )

                        entity = ExtractedEntity(
                            text=entity_text,
                            label=mapped_type,
                            start=entity_start,
                            end=entity_end,
                            confidence=min(0.7 + confidence_boost, 1.0),
                            context=context,
                            metadata={
                                "region_type": region_type,
                                "original_label": label,
                                "extraction_method": "gliner2-fallback",
                                "confidence_boost": confidence_boost,
                            },
                        )
                        entities.append(entity)

                return self._remove_duplicate_entities(entities)
            else:
                # If no model available, return empty list (no rule-based extraction)
                return []
        except Exception:
            return []

    def _fallback_relationship_extraction(self, text: str, entities: List[ExtractedEntity], region_type: str) -> List[Dict[str, Any]]:
        """Fallback relationship extraction with improved quality filtering"""
        relationships = []
        doc_type_name = "general"  # Skip classification for relationships

        # More conservative co-occurrence based relationships
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if entities appear close to each other
                distance = abs(entity1.start - entity2.start)
                
                # Stricter proximity and confidence requirements
                if distance < 100:  # Reduced from 200 to 100 characters
                    # Only create relationships between high-confidence entities
                    if entity1.confidence >= 0.6 and entity2.confidence >= 0.6:
                        relationships.append({
                            'source_entity': entity1.text,
                            'target_entity': entity2.text,
                            'relation_type': 'RELATED_TO',
                            'confidence': 0.5,
                        'context': self._get_relation_context(text, entity1, entity2),
                        'source_entity_id': f"{entity1.text}_{entity1.start}",
                        'target_entity_id': f"{entity2.text}_{entity2.start}",
                        'metadata': {
                            'region_type': region_type,
                            'extraction_method': 'co-occurrence',
                            'document_type': doc_type_name if doc_type else 'general'
                        }
                    })

        return relationships

    def _map_label_to_entity_type(self, label: str, entity_text: str = "", context: str = "") -> str:
        """Map GLiNER label to DeepLightRAG entity type using improved mapper"""
        from .entity_type_mapper import EntityTypeMapper

        # Initialize mapper if not already done
        if not hasattr(self, "_entity_type_mapper"):
            self._entity_type_mapper = EntityTypeMapper()

        # Use the improved mapper with entity text and context
        return self._entity_type_mapper.map_entity_type(label, entity_text, context)

    def _get_entity_context(self, text: str, start: int, end: int, context_window: int = 50) -> str:
        """Get surrounding context for an entity"""
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        return text[context_start:context_end]

    def _remove_duplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity.text.lower(), entity.label)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _update_stats(self, entities: List[ExtractedEntity], relationships: List[Dict[str, Any]] = None):
        """Update extraction statistics"""
        self.extraction_stats["total_extractions"] += 1
        self.extraction_stats["total_entities"] += len(entities)

        for entity in entities:
            self.extraction_stats["entities_by_type"][entity.label] += 1
        
        if relationships:
            self.extraction_stats["total_relations"] += len(relationships)
            for rel in relationships:
                self.extraction_stats["relations_by_type"][rel.get("relation_type", "UNKNOWN")] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        return {
            "total_extractions": self.extraction_stats["total_extractions"],
            "total_entities": self.extraction_stats["total_entities"],
            "total_relations": self.extraction_stats["total_relations"],
            "unified_extractions": self.extraction_stats["unified_extractions"],
            "fallback_extractions": self.extraction_stats["fallback_extractions"],
            "entities_by_type": dict(self.extraction_stats["entities_by_type"]),
            "relations_by_type": dict(self.extraction_stats["relations_by_type"]),
            "classification_stats": dict(self.extraction_stats["classification_stats"]),
        }
