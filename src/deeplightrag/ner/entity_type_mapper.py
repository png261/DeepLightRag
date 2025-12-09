"""
Entity Type Mapper for GLiNER Labels
Maps GLiNER output labels to standardized DeepLightRAG entity types
"""

from typing import Dict, Set, Optional
import re


class EntityTypeMapper:
    """
    Maps GLiNER labels to standardized entity types with high accuracy
    """

    def __init__(self):
        # GLiNER label to DeepLightRAG entity type mapping
        # Optimized for GLiNER2's default labels
        self.gliner_to_entity_type = {
            # Core entity mappings
            "person": "PERSON",
            "people": "PERSON",
            "individual": "PERSON",
            "human": "PERSON",
            "name": "PERSON",
            "author": "PERSON",
            "researcher": "PERSON",
            "scientist": "PERSON",

            # Organization mappings
            "organization": "ORGANIZATION",
            "company": "ORGANIZATION",
            "institution": "ORGANIZATION",
            "agency": "ORGANIZATION",
            "business": "ORGANIZATION",
            "corporation": "ORGANIZATION",
            "university": "ORGANIZATION",
            "startup": "ORGANIZATION",

            # Location mappings
            "location": "LOCATION",
            "place": "LOCATION",
            "geographic": "LOCATION",
            "address": "LOCATION",
            "country": "LOCATION",
            "city": "LOCATION",
            "state": "LOCATION",
            "region": "LOCATION",

            # Date/Time mappings
            "date": "DATE_TIME",
            "time": "DATE_TIME",
            "year": "DATE_TIME",
            "period": "DATE_TIME",
            "duration": "DATE_TIME",
            "when": "DATE_TIME",
            "timestamp": "DATE_TIME",
            "datetime": "DATE_TIME",

            # Money mappings
            "money": "MONEY",
            "cost": "MONEY",
            "price": "MONEY",
            "budget": "MONEY",
            "currency": "MONEY",
            "value": "MONEY",
            "amount": "MONEY",
            "expense": "MONEY",

            # Percentage mappings
            "percentage": "PERCENTAGE",
            "percent": "PERCENTAGE",
            "rate": "PERCENTAGE",
            "proportion": "PERCENTAGE",
            "ratio": "PERCENTAGE",

            # Technical term mappings
            "technical": "TECHNICAL_TERM",
            "technology": "TECHNICAL_TERM",
            "method": "TECHNICAL_TERM",
            "technique": "TECHNICAL_TERM",
            "algorithm": "TECHNICAL_TERM",
            "framework": "TECHNICAL_TERM",
            "model": "TECHNICAL_TERM",
            "system": "TECHNICAL_TERM",
            "architecture": "TECHNICAL_TERM",
            "approach": "TECHNICAL_TERM",

            # Product mappings
            "product": "PRODUCT",
            "software": "PRODUCT",
            "tool": "PRODUCT",
            "application": "PRODUCT",
            "platform": "PRODUCT",
            "service": "PRODUCT",
            "device": "PRODUCT",
            "equipment": "PRODUCT",
            "instrument": "PRODUCT",
            "technology": "PRODUCT",
            "system": "PRODUCT",
            "framework": "PRODUCT",
            "library": "PRODUCT",
            "api": "PRODUCT",
            "language": "PRODUCT",
            "model": "PRODUCT",
            "algorithm": "PRODUCT",

            # Concept mappings
            "concept": "CONCEPT",
            "idea": "CONCEPT",
            "theory": "CONCEPT",
            "principle": "CONCEPT",
            "notion": "CONCEPT",
            "paradigm": "CONCEPT",
            "methodology": "CONCEPT",
            "strategy": "CONCEPT",

            # Event mappings
            "event": "EVENT",
            "meeting": "EVENT",
            "conference": "EVENT",
            "launch": "EVENT",
            "release": "EVENT",
            "announcement": "EVENT",
            "presentation": "EVENT",
            "workshop": "EVENT",

            # Document mappings
            "document": "DOCUMENT",
            "report": "DOCUMENT",
            "paper": "DOCUMENT",
            "article": "DOCUMENT",
            "reference": "DOCUMENT",
            "publication": "DOCUMENT",
            "study": "DOCUMENT",
            "research": "DOCUMENT",

            # Metric mappings
            "measurement": "METRIC",
            "quantity": "METRIC",
            "metric": "METRIC",
            "size": "METRIC",
            "dimension": "METRIC",
            "parameter": "METRIC",
            "statistic": "METRIC",
            "figure": "METRIC",

            # Additional mappings for better coverage
            "project": "PRODUCT",
            "initiative": "EVENT",
            "program": "PRODUCT",
            "standard": "TECHNICAL_TERM",
            "specification": "TECHNICAL_TERM",
            "protocol": "TECHNICAL_TERM",
            "procedure": "TECHNICAL_TERM",
            "process": "TECHNICAL_TERM",
            "language": "TECHNICAL_TERM",
            "library": "PRODUCT",
            "database": "PRODUCT",
            "api": "PRODUCT",
            "interface": "TECHNICAL_TERM",
            "conference": "EVENT",
            "university": "ORGANIZATION",
            "research": "CONCEPT",
            "reference": "DOCUMENT",
            "company": "ORGANIZATION",
            "startup": "ORGANIZATION",
            "government": "ORGANIZATION",
        }

        # Pattern-based mappings for unstructured labels
        self.pattern_mappings = [
            # Year patterns
            (r"^\d{4}$", "DATE_TIME", lambda x: 1900 <= int(x) <= 2100),
            # Percentage patterns
            (r"^\d+\.?\d*%$", "PERCENTAGE", None),
            (r"^\d+\.?\d*\s*percent$", "PERCENTAGE", None),
            # Money patterns
            (r"^\$[\d,]+\.?\d*$", "MONEY", None),
            (r"^\$[\d,]+\.?\d*\s*[mb]?(?:illion|illion)?$", "MONEY", None),
            # Number ranges
            (r"^\d+-\d+$", "METRIC", None),
            # Version numbers
            (r"^v?\d+\.\d+(\.\d+)?$", "PRODUCT", None),
        ]

        # Common word mappings (case-insensitive)
        self.word_mappings = {
            "google": "ORGANIZATION",
            "microsoft": "ORGANIZATION",
            "apple": "ORGANIZATION",
            "amazon": "ORGANIZATION",
            "tesla": "ORGANIZATION",
            "openai": "ORGANIZATION",
            "facebook": "ORGANIZATION",
            "meta": "ORGANIZATION",
            "tensorflow": "PRODUCT",
            "pytorch": "PRODUCT",
            "keras": "PRODUCT",
            "scikit-learn": "PRODUCT",
            "python": "PRODUCT",
            "javascript": "PRODUCT",
            "java": "PRODUCT",
            "c++": "PRODUCT",
            "machine learning": "TECHNICAL_TERM",
            "deep learning": "TECHNICAL_TERM",
            "neural network": "TECHNICAL_TERM",
            "ai": "CONCEPT",
            "artificial intelligence": "CONCEPT",
            "tensorflow": "PRODUCT",
            "pytorch": "PRODUCT",
            "keras": "PRODUCT",
            "opencv": "PRODUCT",
            "huggingface": "PRODUCT",
            "github": "PRODUCT",
            "docker": "PRODUCT",
            "kubernetes": "PRODUCT",
            "aws": "PRODUCT",
            "azure": "PRODUCT",
            "gcp": "PRODUCT",
        }

    def map_entity_type(self, gliner_label: str, entity_text: str, context: str = "") -> str:
        """
        Map GLiNER label to standardized entity type

        Args:
            gliner_label: Label from GLiNER output
            entity_text: The actual entity text
            context: Surrounding context for better classification

        Returns:
            Standardized entity type
        """
        # Clean the label
        label = gliner_label.lower().strip()
        text = entity_text.lower().strip()

        # 1. Direct mapping
        if label in self.gliner_to_entity_type:
            return self.gliner_to_entity_type[label]

        # 2. Pattern-based classification
        for pattern, entity_type, validator in self.pattern_mappings:
            if re.match(pattern, entity_text):
                if validator is None or validator(entity_text):
                    return entity_type

        # 3. Word-based classification for known entities
        if text in self.word_mappings:
            return self.word_mappings[text]

        # 4. Heuristic classification based on label patterns
        if "person" in label or "name" in label:
            return "PERSON"
        elif any(org_word in label for org_word in ["org", "company", "business", "corp", "university", "government", "startup"]):
            return "ORGANIZATION"
        elif any(loc_word in label for loc_word in ["loc", "place", "address", "country", "city"]):
            return "LOCATION"
        elif any(date_word in label for date_word in ["date", "time", "year", "when"]):
            return "DATE_TIME"
        elif any(money_word in label for money_word in ["money", "cost", "price", "$"]):
            return "MONEY"
        elif any(tech_word in label for tech_word in ["tech", "method", "algorithm", "system", "technical"]):
            return "PRODUCT"  # Changed from TECHNICAL_TERM - most tech should be products
        elif any(product_word in label for product_word in ["product", "tool", "software", "app", "platform", "framework", "library", "api"]):
            return "PRODUCT"
        elif label in ["conference", "event", "workshop", "meeting"]:
            return "EVENT"
        elif label in ["reference", "document", "paper", "article"]:
            return "DOCUMENT"

        # 5. Context-based classification
        if context:
            # Check for title case (often names)
            if entity_text.istitle() and len(entity_text.split()) <= 3:
                return "PERSON"

            # Check for all caps (often acronyms for organizations)
            if entity_text.isupper() and len(entity_text) <= 10:
                return "ORGANIZATION"

        # 6. Default fallback based on text content
        if self._looks_like_date(entity_text):
            return "DATE_TIME"
        elif self._looks_like_money(entity_text):
            return "MONEY"
        elif self._looks_like_percentage(entity_text):
            return "PERCENTAGE"

        # Final fallback
        return "CONCEPT"  # Generic type for unknown entities

    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date"""
        date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",
            r"^\d{1,2}/\d{1,2}/\d{2,4}$",
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}",
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns)

    def _looks_like_money(self, text: str) -> bool:
        """Check if text looks like money"""
        return bool(re.match(r"^\$[\d,]+\.?\d*|[\d,]+\.?\d*\s*(USD|EUR|GBP|dollars?|euros?)$", text, re.IGNORECASE))

    def _looks_like_percentage(self, text: str) -> bool:
        """Check if text looks like a percentage"""
        return bool(re.match(r"^\d+\.?\d*%|^\d+\.?\d*\s*percent$", text, re.IGNORECASE))

    def get_confidence_boost(self, entity_text: str, entity_type: str, context: str = "") -> float:
        """
        Get confidence boost based on entity characteristics

        Args:
            entity_text: The entity text
            entity_type: Mapped entity type
            context: Surrounding context

        Returns:
            Confidence boost (0.0 to 0.2)
        """
        boost = 0.0

        # Boost for capitalized entities in appropriate types
        if entity_text.istitle() and entity_type in ["PERSON", "ORGANIZATION", "PRODUCT"]:
            boost += 0.1

        # Boost for all-caps acronyms
        if entity_text.isupper() and 2 <= len(entity_text) <= 6:
            if entity_type in ["ORGANIZATION", "TECHNICAL_TERM"]:
                boost += 0.1

        # Boost for entities in headings/titles
        if any(keyword in context.lower() for keyword in ["#", "##", "title", "heading"]):
            boost += 0.05

        # Boost for common patterns
        if entity_type == "DATE_TIME" and self._looks_like_date(entity_text):
            boost += 0.1
        elif entity_type == "MONEY" and self._looks_like_money(entity_text):
            boost += 0.1
        elif entity_type == "PERCENTAGE" and self._looks_like_percentage(entity_text):
            boost += 0.1

        return min(boost, 0.3)  # Cap the boost at 0.3 for better recall