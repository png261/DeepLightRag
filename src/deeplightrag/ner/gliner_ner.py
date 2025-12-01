"""
GLiNER-based Named Entity Recognition for DeepLightRAG
Zero-shot entity extraction with custom entity types
"""

import re
import torch
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

try:
    from gliner import GLiNER

    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False
    print("Warning: GLiNER not installed. Install with: pip install gliner")


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


class DeepLightRAGEntitySchema:
    """
    Custom entity schema optimized for document understanding
    """

    def __init__(self):
        # Core entity types for document analysis
        self.entity_types = {
            # Basic entities
            "PERSON": {
                "description": "Person names, authors, researchers, experts",
                "examples": ["John Smith", "Dr. Watson", "Prof. Chen", "Einstein"],
                "prompts": ["person", "researcher", "author", "scientist", "expert", "name"],
            },
            "ORGANIZATION": {
                "description": "Companies, institutions, research groups, agencies",
                "examples": ["Google", "MIT", "WHO", "Stanford University", "OpenAI"],
                "prompts": ["organization", "company", "institution", "university", "agency"],
            },
            "LOCATION": {
                "description": "Geographic locations, addresses, regions",
                "examples": ["New York", "Laboratory 3", "Silicon Valley", "Room 101"],
                "prompts": ["location", "place", "country", "city", "region", "address"],
            },
            # Temporal entities
            "DATE_TIME": {
                "description": "Dates, times, temporal expressions, periods",
                "examples": ["2023", "January 15th", "last week", "Q3 2024", "2019-2021"],
                "prompts": ["date", "time", "year", "period", "duration", "when"],
            },
            # Quantitative entities
            "MONEY": {
                "description": "Monetary amounts, currencies, costs, budgets",
                "examples": ["$100", "€50M", "1.2 billion dollars", "cost of $5K"],
                "prompts": ["money", "cost", "price", "budget", "funding", "currency"],
            },
            "PERCENTAGE": {
                "description": "Percentage values, ratios, rates, proportions",
                "examples": ["25%", "0.95", "50 percent", "ratio of 1:3"],
                "prompts": ["percentage", "percent", "ratio", "rate", "proportion"],
            },
            "METRIC": {
                "description": "Measurements, quantities, units, dimensions",
                "examples": ["5.2kg", "100MB", "95°F", "3.5 meters", "1024x768"],
                "prompts": ["measurement", "quantity", "unit", "size", "dimension", "metric"],
            },
            # Technical entities
            "TECHNICAL_TERM": {
                "description": "Domain-specific technical terms, jargon, acronyms",
                "examples": ["machine learning", "OCR", "API", "neural network", "GPU"],
                "prompts": ["technical term", "technology", "method", "technique", "algorithm"],
            },
            "PRODUCT": {
                "description": "Product names, software, tools, systems, models",
                "examples": ["iPhone", "TensorFlow", "DeepLightRAG", "GPT-4", "Windows"],
                "prompts": ["product", "software", "tool", "system", "application", "model"],
            },
            "CONCEPT": {
                "description": "Abstract concepts, theories, methodologies, ideas",
                "examples": [
                    "artificial intelligence",
                    "sustainability",
                    "efficiency",
                    "democracy",
                ],
                "prompts": ["concept", "idea", "theory", "principle", "methodology", "approach"],
            },
            # Research-specific entities
            "METHOD": {
                "description": "Research methods, algorithms, procedures, protocols",
                "examples": [
                    "k-means clustering",
                    "gradient descent",
                    "BERT fine-tuning",
                    "cross-validation",
                ],
                "prompts": [
                    "method",
                    "algorithm",
                    "procedure",
                    "protocol",
                    "technique",
                    "approach",
                ],
            },
            "RESEARCH_ARTIFACT": {
                "description": "Papers, datasets, models, experiments, studies",
                "examples": ["ImageNet", "BERT model", "Study #1", "Dataset A", "Experiment 2"],
                "prompts": ["dataset", "model", "paper", "study", "experiment", "research"],
            },
            "METRIC_RESULT": {
                "description": "Performance metrics, results, scores, evaluations",
                "examples": ["F1-score of 0.95", "accuracy 87%", "BLEU score", "top-1 accuracy"],
                "prompts": ["performance", "result", "score", "metric", "evaluation", "accuracy"],
            },
            # Document structure entities
            "REFERENCE": {
                "description": "Citations, references, bibliography, sources",
                "examples": ["Figure 1", "Table 2", "Section 3.1", "Equation (5)", "[Smith, 2023]"],
                "prompts": ["reference", "citation", "figure", "table", "section", "equation"],
            },
            "KEYWORD": {
                "description": "Keywords, tags, important terms, key phrases",
                "examples": ["deep learning", "natural language processing", "computer vision"],
                "prompts": ["keyword", "key term", "important term", "tag", "subject"],
            },
        }

    def get_entity_labels(self) -> List[str]:
        """Get all entity type labels"""
        return list(self.entity_types.keys())

    def get_prompts_for_entity(self, entity_type: str) -> List[str]:
        """Get prompt variations for an entity type"""
        return self.entity_types.get(entity_type, {}).get("prompts", [entity_type.lower()])

    def get_all_prompts(self) -> List[str]:
        """Get all entity prompts for GLiNER"""
        all_prompts = []
        for entity_data in self.entity_types.values():
            all_prompts.extend(entity_data.get("prompts", []))
        return list(set(all_prompts))  # Remove duplicates


class GLiNERExtractor:
    """
    GLiNER-based entity extractor for DeepLightRAG
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_base",
        confidence_threshold: float = 0.3,
        max_length: int = 512,
        device: str = "auto",
        torch_dtype=None,
        batch_size: int = 8,
    ):
        """
        Initialize GLiNER extractor

        Args:
            model_name: GLiNER model name
            confidence_threshold: Minimum confidence for entity extraction
            max_length: Maximum text length for processing
            device: Device for inference (cpu/cuda)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size

        # Initialize entity schema
        self.schema = DeepLightRAGEntitySchema()

        # Load model
        self.model = None
        self._load_model()

        # Performance tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "total_entities": 0,
            "entities_by_type": defaultdict(int),
            "avg_confidence": 0.0,
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
        """Load GLiNER model with GPU support"""
        if not HAS_GLINER:
            print("GLiNER not available. Using mock extractor.")
            return

        try:
            print(f"Loading GLiNER model: {self.model_name} on {self.device}")

            # Load model
            self.model = GLiNER.from_pretrained(self.model_name, trust_remote_code=True)

            # Move to device and set dtype if specified
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                if self.torch_dtype == torch.float16:
                    self.model = self.model.half()
            elif self.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")

            # Set model to evaluation mode
            self.model.eval()

            print(f"✅ GLiNER model loaded on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load GLiNER model: {e}")
            self.model = None

    def extract_entities(
        self, text: str, entity_types: Optional[List[str]] = None, region_type: str = "general"
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text using GLiNER

        Args:
            text: Input text
            entity_types: Specific entity types to extract (None for all)
            region_type: Type of document region (for context)

        Returns:
            List of extracted entities
        """
        if not self.model:
            return self._mock_extract_entities(text, entity_types)

        # Get entity labels to search for
        if entity_types:
            labels = []
            for ent_type in entity_types:
                labels.extend(self.schema.get_prompts_for_entity(ent_type))
        else:
            labels = self.schema.get_all_prompts()

        # Extract entities
        entities = []

        # Split long text into chunks
        chunks = self._split_text(text)

        for chunk_start, chunk_text in chunks:
            try:
                # GLiNER prediction
                predictions = self.model.predict_entities(
                    chunk_text, labels, threshold=self.confidence_threshold
                )

                # Convert predictions to ExtractedEntity objects
                for pred in predictions:
                    entity = ExtractedEntity(
                        text=pred["text"],
                        label=self._map_label_to_entity_type(pred["label"]),
                        start=pred["start"] + chunk_start,
                        end=pred["end"] + chunk_start,
                        confidence=pred["score"],
                        context=self._get_entity_context(
                            text, pred["start"] + chunk_start, pred["end"] + chunk_start
                        ),
                        metadata={
                            "region_type": region_type,
                            "original_label": pred["label"],
                            "extraction_method": "gliner",
                        },
                    )
                    entities.append(entity)

            except Exception as e:
                print(f"GLiNER extraction failed for chunk: {e}")
                continue

        # Post-process entities
        entities = self._post_process_entities(entities, text)

        # Update statistics
        self._update_stats(entities)

        return entities

    def extract_entities_batch(
        self, texts: List[str], entity_types: Optional[List[str]] = None
    ) -> List[List[ExtractedEntity]]:
        """
        Extract entities from multiple texts in batch

        Args:
            texts: List of input texts
            entity_types: Entity types to extract

        Returns:
            List of entity lists for each text
        """
        results = []

        # Process each text (GLiNER doesn't have native batch processing)
        for text in texts:
            entities = self.extract_entities(text, entity_types)
            results.append(entities)

        return results

    def _split_text(self, text: str) -> List[Tuple[int, str]]:
        """Split long text into processable chunks"""
        if len(text) <= self.max_length:
            return [(0, text)]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.max_length, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within last 100 chars
                sentence_end = text.rfind(".", start, end)
                if sentence_end > start + self.max_length // 2:
                    end = sentence_end + 1

            chunks.append((start, text[start:end]))
            start = end

        return chunks

    def _map_label_to_entity_type(self, label: str) -> str:
        """Map GLiNER label to DeepLightRAG entity type"""
        label_lower = label.lower()

        # Direct mappings
        label_to_type = {
            "person": "PERSON",
            "researcher": "PERSON",
            "author": "PERSON",
            "scientist": "PERSON",
            "organization": "ORGANIZATION",
            "company": "ORGANIZATION",
            "institution": "ORGANIZATION",
            "university": "ORGANIZATION",
            "location": "LOCATION",
            "place": "LOCATION",
            "date": "DATE_TIME",
            "time": "DATE_TIME",
            "year": "DATE_TIME",
            "money": "MONEY",
            "cost": "MONEY",
            "percentage": "PERCENTAGE",
            "percent": "PERCENTAGE",
            "measurement": "METRIC",
            "quantity": "METRIC",
            "technical term": "TECHNICAL_TERM",
            "technology": "TECHNICAL_TERM",
            "product": "PRODUCT",
            "software": "PRODUCT",
            "concept": "CONCEPT",
            "method": "METHOD",
            "algorithm": "METHOD",
            "dataset": "RESEARCH_ARTIFACT",
            "model": "RESEARCH_ARTIFACT",
            "performance": "METRIC_RESULT",
            "result": "METRIC_RESULT",
            "reference": "REFERENCE",
            "keyword": "KEYWORD",
        }

        return label_to_type.get(label_lower, "MISC")

    def _get_entity_context(self, text: str, start: int, end: int, context_window: int = 50) -> str:
        """Get surrounding context for an entity"""
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        return text[context_start:context_end]

    def _post_process_entities(
        self, entities: List[ExtractedEntity], text: str
    ) -> List[ExtractedEntity]:
        """Post-process extracted entities"""
        # Remove duplicates
        entities = self._remove_duplicate_entities(entities)

        # Enhance with pattern-based extraction
        entities = self._enhance_with_patterns(entities, text)

        # Normalize entity text
        for entity in entities:
            entity.normalized_form = self._normalize_entity_text(entity.text, entity.label)

        # Sort by start position
        entities.sort(key=lambda x: x.start)

        return entities

    def _remove_duplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities based on text and position overlap"""
        if not entities:
            return entities

        # Sort by start position
        entities.sort(key=lambda x: (x.start, x.end))

        filtered = [entities[0]]

        for entity in entities[1:]:
            last_entity = filtered[-1]

            # Check for overlap
            if (
                entity.start < last_entity.end
                and entity.end > last_entity.start
                and entity.text.lower() == last_entity.text.lower()
            ):
                # Keep the one with higher confidence
                if entity.confidence > last_entity.confidence:
                    filtered[-1] = entity
            else:
                filtered.append(entity)

        return filtered

    def _enhance_with_patterns(
        self, entities: List[ExtractedEntity], text: str
    ) -> List[ExtractedEntity]:
        """Enhance entity extraction with pattern-based rules"""
        # Pattern-based extraction for specific entity types
        pattern_entities = []

        # Money patterns
        money_patterns = [
            r"\$[\d,]+(?:\.\d+)?(?:[BMK])?",
            r"€[\d,]+(?:\.\d+)?(?:[BMK])?",
            r"[\d,]+(?:\.\d+)?\s*(?:dollars?|euros?|USD|EUR)",
        ]

        for pattern in money_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Check if already extracted
                if not self._overlaps_with_existing(match.start(), match.end(), entities):
                    entity = ExtractedEntity(
                        text=match.group(),
                        label="MONEY",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        context=self._get_entity_context(text, match.start(), match.end()),
                        metadata={"extraction_method": "pattern", "pattern": "money"},
                    )
                    pattern_entities.append(entity)

        # Percentage patterns
        percentage_patterns = [
            r"\d+(?:\.\d+)?%",
            r"\d+(?:\.\d+)?\s*percent",
        ]

        for pattern in percentage_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if not self._overlaps_with_existing(match.start(), match.end(), entities):
                    entity = ExtractedEntity(
                        text=match.group(),
                        label="PERCENTAGE",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                        context=self._get_entity_context(text, match.start(), match.end()),
                        metadata={"extraction_method": "pattern", "pattern": "percentage"},
                    )
                    pattern_entities.append(entity)

        # Reference patterns
        reference_patterns = [
            r"Figure\s+\d+(?:\.\d+)?",
            r"Table\s+\d+(?:\.\d+)?",
            r"Section\s+\d+(?:\.\d+)*",
            r"Equation\s+\(?(\d+)\)?",
            r"\[[\w\s,]+\s+\d{4}\]",  # Citations like [Smith et al., 2023]
        ]

        for pattern in reference_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if not self._overlaps_with_existing(match.start(), match.end(), entities):
                    entity = ExtractedEntity(
                        text=match.group(),
                        label="REFERENCE",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        context=self._get_entity_context(text, match.start(), match.end()),
                        metadata={"extraction_method": "pattern", "pattern": "reference"},
                    )
                    pattern_entities.append(entity)

        return entities + pattern_entities

    def _overlaps_with_existing(
        self, start: int, end: int, entities: List[ExtractedEntity]
    ) -> bool:
        """Check if position overlaps with existing entities"""
        for entity in entities:
            if start < entity.end and end > entity.start:
                return True
        return False

    def _normalize_entity_text(self, text: str, entity_type: str) -> str:
        """Normalize entity text based on type"""
        normalized = text.strip()

        if entity_type == "PERSON":
            # Standardize person names
            normalized = " ".join(word.capitalize() for word in normalized.split())
        elif entity_type == "ORGANIZATION":
            # Keep original case for organizations
            pass
        elif entity_type in ["TECHNICAL_TERM", "CONCEPT", "METHOD"]:
            # Lowercase for technical terms
            normalized = normalized.lower()
        elif entity_type == "PRODUCT":
            # Keep original case for products
            pass

        return normalized

    def _mock_extract_entities(
        self, text: str, entity_types: Optional[List[str]] = None
    ) -> List[ExtractedEntity]:
        """Mock entity extraction when GLiNER is not available"""
        entities = []

        # Simple pattern-based mock extraction
        if "PERSON" in (entity_types or []):
            # Mock person extraction
            person_patterns = [r"Dr\.\s+\w+", r"Prof\.\s+\w+", r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"]
            for pattern in person_patterns:
                for match in re.finditer(pattern, text):
                    entities.append(
                        ExtractedEntity(
                            text=match.group(),
                            label="PERSON",
                            start=match.start(),
                            end=match.end(),
                            confidence=0.7,
                            metadata={"extraction_method": "mock"},
                        )
                    )

        return entities

    def _update_stats(self, entities: List[ExtractedEntity]):
        """Update extraction statistics"""
        self.extraction_stats["total_extractions"] += 1
        self.extraction_stats["total_entities"] += len(entities)

        for entity in entities:
            self.extraction_stats["entities_by_type"][entity.label] += 1

        if entities:
            avg_conf = sum(e.confidence for e in entities) / len(entities)
            current_avg = self.extraction_stats["avg_confidence"]
            total_extractions = self.extraction_stats["total_extractions"]
            # Update running average
            self.extraction_stats["avg_confidence"] = (
                current_avg * (total_extractions - 1) + avg_conf
            ) / total_extractions

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return dict(self.extraction_stats)

    def get_supported_entities(self) -> Dict[str, Dict[str, Any]]:
        """Get supported entity types and their descriptions"""
        return self.schema.entity_types
