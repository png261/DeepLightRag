"""
BERT/DeBERTa-based NER for DeepLightRAG
High-performance NER without LLM dependency for intermediate processing
"""

import re
import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        pipeline,
        TokenClassificationPipeline,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

from .gliner_ner import ExtractedEntity


@dataclass
class BERTNERResult:
    """BERT NER extraction result"""

    entities: List[ExtractedEntity]
    processing_time: float
    model_confidence: float
    tokens_processed: int


class BERTNERExtractor:
    """
    BERT/DeBERTa-based NER extractor optimized for research documents
    Uses pre-trained models fine-tuned on scientific/technical datasets
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        aggregation_strategy: str = "average",
        confidence_threshold: float = 0.5,
        device: str = "auto",
    ):
        """
        Initialize BERT NER extractor

        Args:
            model_name: Pre-trained NER model
            aggregation_strategy: Token aggregation strategy
            confidence_threshold: Minimum confidence for entities
            device: Device for inference
        """
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        # Load model
        self._load_model()

        # Statistics
        self.stats = {
            "total_extractions": 0,
            "total_entities": 0,
            "entities_by_type": defaultdict(int),
            "avg_confidence": 0.0,
        }

    def _load_model(self):
        """Load BERT/DeBERTa model for NER"""
        if not HAS_TRANSFORMERS:
            print("Transformers not available. Using pattern fallback.")
            return

        try:
            print(f"Loading BERT NER model: {self.model_name}")

            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Try to load as NER pipeline first
            try:
                self.pipeline = pipeline(
                    "ner",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    aggregation_strategy=self.aggregation_strategy,
                    device=0 if self.device == "cuda" else -1,
                )
                print(f"✅ BERT NER pipeline loaded on {self.device}")
                return
            except:
                print("NER pipeline failed, trying manual model loading...")

            # Manual loading
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

            if self.device == "cuda":
                self.model = self.model.cuda()

            self.model.eval()
            print(f"✅ BERT NER model loaded manually on {self.device}")

        except Exception as e:
            print(f"Failed to load BERT NER model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None

    def extract_entities(self, text: str, max_length: int = 512) -> List[ExtractedEntity]:
        """
        Extract entities using BERT/DeBERTa

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            List of extracted entities
        """
        if not self.pipeline and not self.model:
            return self._pattern_fallback(text)

        entities = []

        # Split long text
        chunks = self._split_text(text, max_length)

        for chunk_start, chunk_text in chunks:
            try:
                if self.pipeline:
                    # Use pipeline
                    predictions = self.pipeline(chunk_text)
                else:
                    # Manual inference
                    predictions = self._manual_inference(chunk_text)

                # Convert predictions to entities
                chunk_entities = self._convert_predictions(predictions, chunk_text, chunk_start)
                entities.extend(chunk_entities)

            except Exception as e:
                print(f"BERT NER failed for chunk: {e}")
                continue

        # Post-process
        entities = self._post_process_entities(entities, text)

        # Update stats
        self._update_stats(entities)

        return entities

    def _split_text(self, text: str, max_length: int) -> List[Tuple[int, str]]:
        """Split text into processable chunks"""
        if len(text) <= max_length:
            return [(0, text)]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_length, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind(".", start, end)
                if sentence_end > start + max_length // 2:
                    end = sentence_end + 1

            chunks.append((start, text[start:end]))
            start = end

        return chunks

    def _manual_inference(self, text: str) -> List[Dict]:
        """Manual model inference when pipeline unavailable"""
        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to pipeline-like format
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        results = []

        for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
            if token.startswith("##") or token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            # Get max prediction
            max_prob, max_idx = torch.max(pred, dim=0)
            label = self.model.config.id2label.get(max_idx.item(), "O")

            if label != "O" and max_prob.item() > self.confidence_threshold:
                results.append(
                    {
                        "entity": label,
                        "score": max_prob.item(),
                        "word": token,
                        "start": i,  # Token index, will need conversion
                        "end": i + 1,
                    }
                )

        return results

    def _convert_predictions(
        self, predictions: List[Dict], text: str, chunk_start: int
    ) -> List[ExtractedEntity]:
        """Convert model predictions to ExtractedEntity objects"""
        entities = []

        for pred in predictions:
            if pred.get("score", 0) < self.confidence_threshold:
                continue

            # Extract entity information
            entity_text = pred.get("word", "")
            entity_label = pred.get("entity", "")
            confidence = pred.get("score", 0.0)

            # Handle character positions
            start_pos = pred.get("start", 0) + chunk_start
            end_pos = pred.get("end", len(entity_text)) + chunk_start

            # Clean up label (remove B-/I- prefixes)
            clean_label = self._clean_label(entity_label)

            # Create entity
            entity = ExtractedEntity(
                text=entity_text.replace("##", ""),  # Remove subword markers
                label=clean_label,
                start=start_pos,
                end=end_pos,
                confidence=confidence,
                metadata={
                    "extraction_method": "bert_ner",
                    "model_name": self.model_name,
                    "original_label": entity_label,
                },
            )

            entities.append(entity)

        return entities

    def _clean_label(self, label: str) -> str:
        """Clean BIO labels to entity types"""
        # Remove B-/I- prefixes
        if label.startswith(("B-", "I-")):
            label = label[2:]

        # Map common NER labels to our schema
        label_mapping = {
            "PER": "PERSON",
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "ORGANIZATION": "ORGANIZATION",
            "LOC": "LOCATION",
            "LOCATION": "LOCATION",
            "MISC": "CONCEPT",
            "GPE": "LOCATION",  # Geo-political entity
            "DATE": "DATE_TIME",
            "TIME": "DATE_TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENTAGE",
            "CARDINAL": "METRIC",
            "ORDINAL": "METRIC",
        }

        return label_mapping.get(label.upper(), label.upper())

    def _post_process_entities(
        self, entities: List[ExtractedEntity], text: str
    ) -> List[ExtractedEntity]:
        """Post-process extracted entities"""
        # Merge adjacent entities of same type
        entities = self._merge_adjacent_entities(entities)

        # Remove duplicates
        entities = self._remove_duplicates(entities)

        # Enhance with patterns
        pattern_entities = self._extract_pattern_entities(text, entities)
        entities.extend(pattern_entities)

        # Sort by position
        entities.sort(key=lambda x: x.start)

        return entities

    def _merge_adjacent_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Merge adjacent entities of the same type"""
        if not entities:
            return entities

        entities.sort(key=lambda x: x.start)
        merged = [entities[0]]

        for entity in entities[1:]:
            last = merged[-1]

            # Check if adjacent and same type
            if entity.start <= last.end + 2 and entity.label == last.label:
                # Merge entities
                merged_text = last.text + " " + entity.text
                merged_entity = ExtractedEntity(
                    text=merged_text.strip(),
                    label=last.label,
                    start=last.start,
                    end=entity.end,
                    confidence=max(last.confidence, entity.confidence),
                    metadata={**last.metadata, "merged": True},
                )
                merged[-1] = merged_entity
            else:
                merged.append(entity)

        return merged

    def _remove_duplicates(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities"""
        seen = set()
        filtered = []

        for entity in entities:
            key = (entity.text.lower(), entity.label, entity.start)
            if key not in seen:
                seen.add(key)
                filtered.append(entity)

        return filtered

    def _extract_pattern_entities(
        self, text: str, existing_entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Extract additional entities using patterns"""
        pattern_entities = []
        existing_spans = {(e.start, e.end) for e in existing_entities}

        # Technical term patterns
        tech_patterns = [
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]*)*\b",  # CamelCase
            r"\b[A-Z]{2,}\b",  # Acronyms
            r"\b\w+[-_]\w+\b",  # Hyphen/underscore terms
        ]

        for pattern in tech_patterns:
            for match in re.finditer(pattern, text):
                span = (match.start(), match.end())
                if span not in existing_spans and len(match.group()) > 2:
                    entity = ExtractedEntity(
                        text=match.group(),
                        label="TECHNICAL_TERM",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.7,
                        metadata={"extraction_method": "pattern", "pattern": "technical"},
                    )
                    pattern_entities.append(entity)
                    existing_spans.add(span)

        return pattern_entities

    def _pattern_fallback(self, text: str) -> List[ExtractedEntity]:
        """Pattern-based fallback when models unavailable"""
        entities = []

        # Basic patterns
        patterns = {
            "PERSON": [r"Dr\.\s+\w+", r"Prof\.\s+\w+", r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"],
            "ORGANIZATION": [r"\b[A-Z][a-z]+\s+(?:Inc|Corp|LLC|University|Institute)\b"],
            "PERCENTAGE": [r"\d+(?:\.\d+)?%"],
            "MONEY": [r"\$[\d,]+(?:\.\d+)?"],
        }

        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                for match in re.finditer(pattern, text):
                    entities.append(
                        ExtractedEntity(
                            text=match.group(),
                            label=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.6,
                            metadata={"extraction_method": "pattern_fallback"},
                        )
                    )

        return entities

    def _update_stats(self, entities: List[ExtractedEntity]):
        """Update extraction statistics"""
        self.stats["total_extractions"] += 1
        self.stats["total_entities"] += len(entities)

        for entity in entities:
            self.stats["entities_by_type"][entity.label] += 1

        if entities:
            avg_conf = sum(e.confidence for e in entities) / len(entities)
            total_extractions = self.stats["total_extractions"]
            current_avg = self.stats["avg_confidence"]
            self.stats["avg_confidence"] = (
                current_avg * (total_extractions - 1) + avg_conf
            ) / total_extractions

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return dict(self.stats)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "aggregation_strategy": self.aggregation_strategy,
            "model_loaded": self.model is not None or self.pipeline is not None,
        }
