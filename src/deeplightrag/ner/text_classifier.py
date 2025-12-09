"""
Text Classifier for Dynamic Label Selection
Identifies document types to optimize entity and relationship extraction
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass


@dataclass
class DocumentType:
    """Document type classification result"""
    name: str
    confidence: float
    dominant_labels: List[str]
    metadata: Dict[str, any]


class DocumentClassifier:
    """
    Classifies text to determine optimal extraction labels
    """

    def __init__(self):
        # Define document type patterns and characteristics
        self.document_patterns = {
            "academic": {
                "keywords": [
                    "abstract", "introduction", "methodology", "results", "discussion",
                    "conclusion", "references", "bibliography", "doi", "arxiv",
                    "journal", "conference", "paper", "research", "study",
                    "experiment", "hypothesis", "theorem", "proof", "algorithm",
                    "dataset", "benchmark", "evaluation", "performance", "accuracy",
                    "precision", "recall", "f1-score"
                ],
                "patterns": [
                    r'\b(?:Fig\.|Figure)\s+\d+',
                    r'\b(?:Tab\.|Table)\s+\d+',
                    r'\b(?:Eq\.|Equation)\s*\(\d+)',
                    r'\b\d{4}\.\s*[A-Za-z]+',
                    r'\barxiv:\d+\.\d+',
                    r'\bdoi:\s*10\.\d+/\S+'
                ],
                "entity_labels": ["person", "organization", "location", "date", "product",
                                "technical", "concept", "method", "metric", "document"],
                "relation_labels": ["authored_by", "published_by", "references", "achieves",
                                   "improves", "outperforms", "uses", "evaluates", "proves"]
            },

            "business": {
                "keywords": [
                    "revenue", "profit", "loss", "market", "sales", "customer",
                    "client", "company", "corporation", "CEO", "CFO", "CTO", "executive",
                    "quarter", "annual", "report", "earnings", "stock", "share",
                    "investment", "funding", "merger", "acquisition", "partnership",
                    "budget", "cost", "expense", "strategy", "growth", "performance"
                ],
                "patterns": [
                    r'\$\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:million|billion|trillion)',
                    r'\b\d+(?:\.\d+)?%\s*(?:growth|increase|decrease)',
                    r'\bQ[1-4]\s+20\d{2}',
                    r'\bFY\s+20\d{2}'
                ],
                "entity_labels": ["organization", "person", "location", "date", "money",
                                "percentage", "product", "event"],
                "relation_labels": ["acquired_by", "merged_with", "invests_in", "competes_with",
                                   "partners_with", "reports_to", "located_in", "affiliated_with"]
            },

            "technical": {
                "keywords": [
                    "API", "SDK", "framework", "library", "module", "function",
                    "class", "method", "algorithm", "data structure", "database",
                    "server", "client", "frontend", "backend", "architecture",
                    "deployment", "configuration", "installation", "version",
                    "bug", "error", "issue", "fix", "patch", "update", "release",
                    "code", "programming", "development", "testing", "debugging"
                ],
                "patterns": [
                    r'\b[A-Z][a-z]*[A-Z][a-z]*\b',  # CamelCase
                    r'\b\w+\(\.\w+)+\b',  # Dotted notation
                    r'function\s+\w+\s*\(',
                    r'class\s+\w+\s*:',
                    r'def\s+\w+\s*\(',
                    r'<[^>]+>',
                    r'www\.\S+\.\S+',
                    r'https?://\S+'
                ],
                "entity_labels": ["technical", "product", "organization", "person",
                                "document", "metric"],
                "relation_labels": ["uses", "depends_on", "implements", "extends",
                                   "configures", "deploys", "requires", "validates"]
            },

            "legal": {
                "keywords": [
                    "court", "judge", "lawyer", "attorney", "plaintiff", "defendant",
                    "case", "lawsuit", "verdict", "ruling", "judgment", "decision",
                    "contract", "agreement", "clause", "regulation", "statute",
                    "legal", "evidence", "testimony", "witness", "objection",
                    "sustained", "overruled", "appealed", "filed", "motion"
                ],
                "patterns": [
                    r'\b\d+:\d+\s*(?:AM|PM|am|pm)',
                    r'\bCourt\s+of\s+\w+',
                    r'\bNo\.\s+\d+-\d+',
                    r'\b§\s*\d+',
                    r'\b\d+\s+U\.S\.C\.'
                ],
                "entity_labels": ["person", "organization", "location", "date", "document",
                                "event"],
                "relation_labels": ["represented_by", "against", "testified_in", "filed_by",
                                   "decided_by", "appealed_by", "cites", "references"]
            },

            "medical": {
                "keywords": [
                    "patient", "doctor", "physician", "nurse", "hospital", "clinic",
                    "diagnosis", "treatment", "therapy", "medication", "drug", "medicine",
                    "symptom", "disease", "condition", "procedure", "surgery", "test",
                    "lab", "result", "blood", "pressure", "temperature", "dose",
                    "prescription", "medical", "health", "clinical"
                ],
                "patterns": [
                    r'\b\d+\s*mg\b',
                    r'\b\d+\s*ml\b',
                    r'\b\d+/\d+/\d{4}',
                    r'\bBP:\s*\d+/\d+',
                    r'\bHR:\s*\d+',
                    r'\bTemp:\s*\d+\.?\d*°(?:C|F)'
                ],
                "entity_labels": ["person", "organization", "location", "date", "metric",
                                "product"],
                "relation_labels": ["diagnosed_with", "treated_with", "prescribed_for",
                                   "tested_for", "operated_by", "monitored_by"]
            },

            "news": {
                "keywords": [
                    "reported", "announced", "said", "according to", "sources",
                    "journalist", "reporter", "editor", "breaking", "update",
                    "story", "article", "headline", "press", "media", "interview",
                    "commentary", "opinion", "statement", "confirmed", "denied"
                ],
                "patterns": [
                    r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+\s+\-\-',
                    r'\b\d+\s+(?:years?|months?|weeks?|days?)\s+ago',
                    r'\b[A-Z][a-z]+\s+\(AP\)',
                    r'\b\d+:\d+\s*[APM|ampm]'
                ],
                "entity_labels": ["person", "organization", "location", "date", "event",
                                "money"],
                "relation_labels": ["reported_by", "quoted_by", "mentioned_in",
                                   "located_at", "happened_on", "commented_on"]
            }
        }

    def classify_text(self, text: str) -> DocumentType:
        """
        Classify text into document type and return optimal labels

        Args:
            text: Input text to classify

        Returns:
            DocumentType classification result
        """
        text_lower = text.lower()

        # Score each document type
        scores = {}
        for doc_type, config in self.document_patterns.items():
            score = 0

            # Count keyword matches
            keyword_matches = 0
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    keyword_matches += 1

            # Count pattern matches
            pattern_matches = 0
            for pattern in config["patterns"]:
                if re.search(pattern, text):
                    pattern_matches += 1

            # Calculate score (weighted)
            scores[doc_type] = (keyword_matches * 1) + (pattern_matches * 2)

        # Determine dominant type
        if not scores or max(scores.values()) == 0:
            # Default to general if no specific patterns found
            dominant_type = "general"
            confidence = 0.5
        else:
            dominant_type = max(scores, key=scores.get)
            total_score = sum(scores.values())
            confidence = scores[dominant_type] / total_score if total_score > 0 else 1.0

        # Get labels for dominant type
        config = self.document_patterns.get(dominant_type, self.document_patterns["academic"])
        entity_labels = config["entity_labels"]
        relation_labels = config["relation_labels"]

        return DocumentType(
            name=dominant_type,
            confidence=confidence,
            dominant_labels=entity_labels,
            metadata={
                "keyword_score": scores.get(dominant_type, 0),
                "all_scores": scores
            }
        )

    def classify_by_region_type(self, text: str, region_type: str) -> DocumentType:
        """
        Classify text considering region type for better context

        Args:
            text: Input text
            region_type: Type of visual region (header, paragraph, table, etc.)

        Returns:
            DocumentType classification result
        """
        # Get base classification
        doc_type = self.classify_text(text)

        # Adjust based on region type
        region_adjustments = {
            "header": {
                "entity_labels": ["person", "organization", "location", "date"],
                "relation_labels": ["authored_by", "published_on", "affiliated_with"]
            },
            "caption": {
                "entity_labels": ["document", "metric", "product"],
                "relation_labels": ["describes", "references", "shows"]
            },
            "table": {
                "entity_labels": ["metric", "money", "percentage", "date"],
                "relation_labels": ["compares_to", "achieves", "contains"]
            },
            "list": {
                "entity_labels": ["concept", "product", "organization"],
                "relation_labels": ["includes", "supports", "related_to"]
            },
            "paragraph": {
                "entity_labels": doc_type.dominant_labels,
                "relation_labels": doc_type.dominant_labels
            }
        }

        # Apply region-specific adjustments
        if region_type in region_adjustments:
            adjustment = region_adjustments[region_type]
            doc_type.dominant_labels = list(set(doc_type.dominant_labels + adjustment["entity_labels"]))

        return doc_type

    def get_optimal_labels(self, text: str, region_type: str = "general") -> Tuple[List[str], List[str]]:
        """
        Get optimal labels for entity and relationship extraction

        Args:
            text: Input text
            region_type: Type of visual region

        Returns:
            Tuple of (entity_labels, relation_labels)
        """
        doc_type = self.classify_by_region_type(text, region_type)

        # Get additional GLiNER2 labels based on text
        base_labels = {
            "person": ["person", "people"],
            "organization": ["organization", "company", "institution"],
            "location": ["location", "place"],
            "date": ["date", "time", "year"],
            "money": ["money", "cost", "price"],
            "percentage": ["percentage", "percent"],
            "technical": ["technical", "technology", "method"],
            "product": ["product", "software", "tool"],
            "concept": ["concept", "idea", "approach"],
            "event": ["event", "meeting", "conference"],
            "document": ["document", "report", "paper"],
            "metric": ["measurement", "quantity", "metric"]
        }

        # Combine base labels with document-type specific labels
        all_entity_labels = doc_type.dominant_labels.copy()

        # Add corresponding GLiNER2 labels
        gliner_labels = []
        for entity_type in all_entity_labels:
            if entity_type in base_labels:
                gliner_labels.extend(base_labels[entity_type])

        # Remove duplicates
        entity_labels = list(set(gliner_labels))
        relation_labels = doc_type.dominant_labels.copy()

        return entity_labels, relation_labels