"""
Query Classification for Adaptive Retrieval
Classifies queries into 4 levels based on complexity
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class QueryLevel:
    """Query complexity level"""

    level: int
    name: str
    max_tokens: int
    max_nodes: int
    strategy: str
    description: str


class QueryClassifier:
    """
    Adaptive Query Routing
    Classifies queries into 4 complexity levels
    """

    def __init__(self):
        self.levels = {
            1: QueryLevel(
                level=1,
                name="simple",
                max_tokens=2000,
                max_nodes=5,
                strategy="entity_lookup",
                description="Simple factual queries",
            ),
            2: QueryLevel(
                level=2,
                name="complex",
                max_tokens=6000,
                max_nodes=20,
                strategy="hybrid",
                description="Complex reasoning queries",
            ),
            3: QueryLevel(
                level=3,
                name="multi_doc",
                max_tokens=10000,
                max_nodes=50,
                strategy="hierarchical",
                description="Multi-document synthesis",
            ),
            4: QueryLevel(
                level=4,
                name="visual",
                max_tokens=12000,
                max_nodes=30,
                strategy="visual_fusion",
                description="Visual-semantic fusion queries",
            ),
        }

        # Keywords for classification
        self.visual_keywords = [
            "chart",
            "figure",
            "table",
            "image",
            "diagram",
            "graph",
            "plot",
            "visualization",
            "picture",
            "photo",
            "page",
            "show",
            "display",
            "look",
            "see",
        ]

        self.comparison_keywords = [
            "compare",
            "versus",
            "vs",
            "difference",
            "between",
            "contrast",
            "similar",
            "different",
            "both",
            "multiple",
        ]

        self.reasoning_keywords = [
            "why",
            "how",
            "explain",
            "analyze",
            "because",
            "correlate",
            "relationship",
            "impact",
            "effect",
            "cause",
        ]

        self.simple_keywords = [
            "what",
            "which",
            "when",
            "where",
            "who",
            "name",
            "list",
            "define",
            "is",
            "are",
        ]

    def classify(self, query: str) -> QueryLevel:
        """
        Classify query into complexity level

        Args:
            query: User query string

        Returns:
            QueryLevel with appropriate settings
        """
        query_lower = query.lower()

        # Check for visual queries (Level 4)
        if self._is_visual_query(query_lower):
            return self.levels[4]

        # Check for multi-document queries (Level 3)
        if self._is_multi_doc_query(query_lower):
            return self.levels[3]

        # Check for complex reasoning (Level 2)
        if self._is_complex_query(query_lower):
            return self.levels[2]

        # Default to simple factual (Level 1)
        return self.levels[1]

    def _is_visual_query(self, query: str) -> bool:
        """Check if query requires visual information"""
        # Check for visual keywords
        for keyword in self.visual_keywords:
            if keyword in query:
                return True

        # Check for page references
        if re.search(r"page\s+\d+", query):
            return True

        # Check for specific visual references
        if re.search(r"figure\s+\d+", query):
            return True

        if re.search(r"table\s+\d+", query):
            return True

        return False

    def _is_multi_doc_query(self, query: str) -> bool:
        """Check if query requires multi-document analysis"""
        # Check for comparison keywords
        comparison_count = sum(1 for keyword in self.comparison_keywords if keyword in query)
        if comparison_count >= 2:
            return True

        # Check for multiple entity mentions
        # Crude heuristic: multiple capitalized words
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", query)
        if len(capitalized) >= 3:
            return True

        # Check for aggregation terms
        aggregation_terms = ["all", "every", "each", "across", "throughout", "summary"]
        for term in aggregation_terms:
            if term in query:
                return True

        return False

    def _is_complex_query(self, query: str) -> bool:
        """Check if query requires complex reasoning"""
        # Check for reasoning keywords
        for keyword in self.reasoning_keywords:
            if keyword in query:
                return True

        # Check query length (longer = more complex)
        word_count = len(query.split())
        if word_count > 15:
            return True

        # Check for multiple questions
        if query.count("?") > 1:
            return True

        # Check for conditional phrases
        conditional_patterns = [
            r"\bif\b.*\bthen\b",
            r"\bwhen\b.*\bhow\b",
            r"\bwhat\b.*\band\b.*\bwhy\b",
        ]
        for pattern in conditional_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        return False

    def get_strategy_description(self, level: int) -> str:
        """Get description of retrieval strategy for a level"""
        strategies = {
            1: "Entity Lookup: Direct entity retrieval with minimal context",
            2: "Hybrid Search: Entity + relationship traversal (2-hop)",
            3: "Hierarchical: Cross-document entity linking and aggregation",
            4: "Visual Fusion: Combine visual regions with entity context",
        }
        return strategies.get(level, "Unknown strategy")

    def analyze_query(self, query: str) -> Dict:
        """
        Provide detailed analysis of query classification

        Returns:
            Dictionary with classification details
        """
        level = self.classify(query)

        # Count keyword matches
        visual_matches = [kw for kw in self.visual_keywords if kw in query.lower()]
        reasoning_matches = [kw for kw in self.reasoning_keywords if kw in query.lower()]
        comparison_matches = [kw for kw in self.comparison_keywords if kw in query.lower()]

        return {
            "query": query,
            "level": level.level,
            "level_name": level.name,
            "max_tokens": level.max_tokens,
            "max_nodes": level.max_nodes,
            "strategy": level.strategy,
            "description": level.description,
            "visual_keywords_found": visual_matches,
            "reasoning_keywords_found": reasoning_matches,
            "comparison_keywords_found": comparison_matches,
            "word_count": len(query.split()),
            "token_budget": f"{level.max_tokens} tokens (vs 30K fixed in LightRAG)",
        }
