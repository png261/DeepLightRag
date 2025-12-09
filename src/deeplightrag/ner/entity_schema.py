"""
Comprehensive Entity Schema for DeepLightRAG
Defines entities and their patterns for better extraction
"""

from typing import Dict, List, Optional, Set, Tuple
import re


class EntitySchema:
    """GLiNER2-optimized entity schema"""

    def __init__(self):
        # Entity types optimized for GLiNER2 labels
        self.entity_types = {
            "PERSON": {
                "description": "Person names, titles, roles",
                "gliner_labels": ["person", "people", "individual", "human"],
                "examples": ["John Smith", "Dr. Watson", "Tim Cook"],
            },

            "ORGANIZATION": {
                "description": "Companies, institutions, organizations",
                "gliner_labels": ["organization", "company", "institution", "agency", "business"],
                "examples": ["Google", "MIT", "OpenAI", "Apple"],
            },

            "LOCATION": {
                "description": "Geographic locations, places",
                "gliner_labels": ["location", "place", "geographic", "address", "country", "city"],
                "examples": ["New York", "Cupertino", "Silicon Valley"],
            },

            "DATE_TIME": {
                "description": "Dates, times, temporal expressions",
                "gliner_labels": ["date", "time", "year", "period", "duration", "when"],
                "examples": ["2023", "January 15", "yesterday"],
            },

            "MONEY": {
                "description": "Monetary amounts and values",
                "gliner_labels": ["money", "cost", "price", "budget", "currency", "value"],
                "examples": ["$100", "€50M", "1 billion dollars"],
            },

            "PERCENTAGE": {
                "description": "Percentages and ratios",
                "gliner_labels": ["percentage", "percent", "rate", "proportion"],
                "examples": ["25%", "50 percent"],
            },

            "TECHNICAL_TERM": {
                "description": "Technical and scientific terms",
                "gliner_labels": ["technical", "technology", "method", "technique", "algorithm"],
                "examples": ["machine learning", "OCR", "API", "GPU", "deep learning"],
            },

            "PRODUCT": {
                "description": "Products, software, tools",
                "gliner_labels": ["product", "software", "tool", "system", "application"],
                "examples": ["iPhone", "TensorFlow", "Windows", "ImageNet"],
            },

            "CONCEPT": {
                "description": "Abstract concepts and ideas",
                "gliner_labels": ["concept", "idea", "approach", "method", "theory"],
                "examples": ["innovation", "performance", "methodology", "architecture"],
            },

            "EVENT": {
                "description": "Events, meetings, conferences",
                "gliner_labels": ["event", "meeting", "conference", "launch", "release"],
                "examples": ["Conference 2024", "Product Launch", "announcement"],
            },

            "DOCUMENT": {
                "description": "Document types and references",
                "gliner_labels": ["document", "report", "paper", "article", "reference"],
                "examples": ["Report 123", "Table 1", "Figure 2", "paper"],
            },

            "METRIC": {
                "description": "Measurements and metrics",
                "gliner_labels": ["measurement", "quantity", "metric", "value", "size"],
                "examples": ["100GB", "5kg", "95%", "accuracy"],
            },
        }

        # Create reverse mapping for patterns
        self.pattern_to_type = {}
        for entity_type, info in self.entity_types.items():
            for pattern in info.get("patterns", []):
                self.pattern_to_type[pattern] = entity_type

    def get_entity_type_by_text(self, text: str, context: Optional[str] = None) -> Optional[str]:
        """Determine entity type based on text patterns"""
        text_lower = text.lower().strip()

        # Check exact patterns
        for entity_type, info in self.entity_types.items():
            for pattern in info.get("patterns", []):
                if re.search(pattern, text, re.IGNORECASE):
                    return entity_type

        # Check for GLiNER labels (would need actual GLiNER output)
        # This is simplified - in practice, you'd use GLiNER's output

        return None

    def get_gliner_labels_for_type(self, entity_type: str) -> List[str]:
        """Get GLiNER labels for a given entity type"""
        return self.entity_types.get(entity_type, {}).get("gliner_labels", [])

    def get_all_gliner_labels(self) -> List[str]:
        """Get all GLiNER labels"""
        all_labels = []
        for info in self.entity_types.values():
            all_labels.extend(info.get("gliner_labels", []))
        return list(set(all_labels))

    def get_entity_types_for_context(self, context: str) -> Optional[List[str]]:
        """Get entity types that are relevant for a given context"""
        context_mapping = {
            "header": ["PERSON", "ORGANIZATION", "DATE", "EVENT"],
            "paragraph": ["PERSON", "ORGANIZATION", "CONCEPT", "TECHNICAL_TERM"],
            "table": ["METRIC", "MONEY", "DATE_TIME", "PRODUCT"],
            "list": ["CONCEPT", "PRODUCT", "EVENT", "METHOD"],
            "figure": ["PRODUCT", "TECHNICAL_TERM", "CONCEPT"],
            "caption": ["DOCUMENT", "PERSON", "ORGANIZATION"],
            "quote": ["PERSON", "CONCEPT", "EVENT"],
        }

        return context_mapping.get(context, None)

    def normalize_entity_name(self, text: str, entity_type: str) -> str:
        """Normalize entity name based on type"""
        text = text.strip()

        if entity_type == "PERSON":
            # Keep proper casing for names
            words = text.split()
            if len(words) > 0:
                words[0] = words[0].title()
            text = " ".join(words)
        elif entity_type == "ORGANIZATION":
            # Keep original case for organizations
            pass
        elif entity_type in ["DATE_TIME", "MONEY", "PERCENTAGE", "METRIC"]:
            # Normalize numbers
            text = text.replace(",", "")  # Remove comma separators

        return text

    def extract_with_patterns(self, text: str, entity_types: Optional[List[str]] = None) -> List[Dict]:
        """Extract entities using pattern matching"""
        entities = []

        for entity_type, info in self.entity_types.items():
            if entity_types and entity_type not in entity_types:
                continue

            for pattern in info.get("patterns", []):
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group()
                    start = match.start()
                    end = match.end()

                    entity = {
                        "text": entity_text,
                        "type": entity_type,
                        "start": start,
                        "end": end,
                        "confidence": 0.9,  # High confidence for pattern matches
                        "source": "pattern"
                    }
                    entities.append(entity)

        # Remove duplicates (same text and type)
        unique_entities = []
        seen = set()

        for entity in entities:
            key = (entity["text"].lower(), entity["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities