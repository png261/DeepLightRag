"""
Entity Quality Filter
Filters out low-quality entities extracted by GLiNER2
"""

import re
import string
from typing import List, Set


class EntityFilter:
    """Filter low-quality entities based on various heuristics"""

    def __init__(
        self,
        min_length: int = 2,
        max_number_length: int = 4,
        confidence_threshold: float = 0.2,  # Even lower threshold for better recall
    ):
        """
        Initialize entity filter

        Args:
            min_length: Minimum entity name length
            max_number_length: Maximum length for standalone numbers
            confidence_threshold: Minimum confidence score
        """
        self.min_length = min_length
        self.max_number_length = max_number_length
        self.confidence_threshold = confidence_threshold

        # Common stop words that shouldn't be entities
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
        }

        # Entity types that should be filtered more strictly
        self.strict_types = {"DATE_TIME", "NUMBER", "QUANTITY"}

    def is_valid_entity(
        self,
        entity_name: str,
        entity_type: str,
        confidence: float,
        block_type: str = "",
        context_text: str = "",
    ) -> bool:
        """
        Check if entity meets quality criteria
        NOTE: Disabled filtering to accept all extracted entities

        Args:
            entity_name: Entity name/value
            entity_type: Entity type (PERSON, ORGANIZATION, etc.)
            confidence: Extraction confidence score
            block_type: Type of text block (heading, paragraph, etc.)
            context_text: Surrounding text context

        Returns:
            True if entity should be kept (always true now)
        """
        name = entity_name.strip()

        # Only filter out completely empty entities
        if not name:
            return False

        # Accept all other entities regardless of confidence, type, etc.
        return True

    def _is_valid_strict_entity(
        self, name: str, entity_type: str, block_type: str
    ) -> bool:
        """
        Stricter validation for DATE_TIME and NUMBER entities

        Args:
            name: Entity name
            entity_type: Entity type
            block_type: Block type context

        Returns:
            True if valid date/number entity
        """
        # Allow dates with clear date patterns
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # 2024-01-15
            r"\d{1,2}/\d{1,2}/\d{2,4}",  # 01/15/2024
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}",
            r"\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
        ]

        for pattern in date_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return True

        # Filter standalone small numbers (likely page numbers or list numbers)
        if name.isdigit():
            num = int(name)
            # Keep year-like numbers
            if 1900 <= num <= 2100:
                return True
            # Keep version numbers
            if len(name) <= 3 and block_type in ["heading", "title"]:
                return True
            # Filter small numbers
            if len(name) <= self.max_number_length:
                return False

        # Allow numbers with units or context
        if re.search(r"\d+\s*(ms|s|sec|min|hour|MB|GB|KB|%|percent)", name, re.IGNORECASE):
            return True

        return False

    def _is_ocr_error(self, name: str) -> bool:
        """
        Detect likely OCR errors

        Args:
            name: Entity name

        Returns:
            True if likely an OCR error
        """
        # Mixed language fragments (common OCR error)
        # Check for Vietnamese mixed with English in weird ways
        vietnamese_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
        
        has_vietnamese = any(c.lower() in vietnamese_chars for c in name)
        has_ascii = any(c.isascii() and c.isalpha() for c in name)

        # If mixed Vietnamese and ASCII in very short string, likely OCR error
        if has_vietnamese and has_ascii and len(name) < 20:
            # Unless it's a proper name
            if not name[0].isupper():
                return True

        # Excessive special characters (OCR artifacts)
        special_char_ratio = sum(c in string.punctuation for c in name) / len(name)
        if special_char_ratio > 0.3:
            return True

        # Random concatenated text without spaces
        if len(name) > 30 and " " not in name and not name.isupper():
            # Check if it looks like concatenated words
            capital_count = sum(c.isupper() for c in name)
            if capital_count > len(name) * 0.3:
                return True

        return False

    def filter_entities(
        self, entities: List[dict], region_context: dict = None
    ) -> List[dict]:
        """
        Filter a list of entities

        Args:
            entities: List of entity dictionaries with name, type, confidence
            region_context: Optional context about the text region

        Returns:
            Filtered list of entities
        """
        filtered = []

        block_type = ""
        context_text = ""
        if region_context:
            block_type = region_context.get("block_type", "")
            context_text = region_context.get("text", "")

        for entity in entities:
            if self.is_valid_entity(
                entity_name=entity.get("name", ""),
                entity_type=entity.get("entity_type", ""),
                confidence=entity.get("confidence", 0.0),
                block_type=block_type,
                context_text=context_text,
            ):
                filtered.append(entity)

        return filtered

    def deduplicate_entities(
        self, entities: List[dict], similarity_threshold: float = 0.8
    ) -> List[dict]:
        """
        Remove duplicate entities with fuzzy matching

        Args:
            entities: List of entities
            similarity_threshold: Similarity threshold for matching

        Returns:
            Deduplicated entities
        """
        if not entities:
            return []

        # Sort by confidence (keep higher confidence versions)
        sorted_entities = sorted(
            entities, key=lambda e: e.get("confidence", 0), reverse=True
        )

        kept = []
        kept_names_lower = []

        for entity in sorted_entities:
            name = entity.get("name", "")
            name_lower = name.lower().strip()

            # Check if similar entity already kept
            is_duplicate = False
            for kept_name in kept_names_lower:
                if self._are_similar(name_lower, kept_name, similarity_threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(entity)
                kept_names_lower.append(name_lower)

        return kept

    def _are_similar(self, str1: str, str2: str, threshold: float) -> bool:
        """
        Check if two strings are similar (simple character overlap)

        Args:
            str1: First string
            str2: Second string
            threshold: Similarity threshold

        Returns:
            True if strings are similar enough
        """
        # Exact match
        if str1 == str2:
            return True

        # One is substring of other
        if str1 in str2 or str2 in str1:
            return True

        # Simple character overlap (Jaccard similarity)
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return False

        similarity = intersection / union
        return similarity >= threshold
