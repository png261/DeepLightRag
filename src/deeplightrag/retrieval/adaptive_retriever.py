"""
Adaptive Retrieval System
Uses query complexity to determine optimal retrieval strategy
Enhanced with structured output, relationship search, and better ranking
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. Hybrid search will be disabled.")

from ..graph.dual_layer import DualLayerGraph
from .query_classifier import QueryClassifier, QueryLevel
from ..utils.component_factory import component_factory


@dataclass
class RetrievedEntity:
    """Entity with retrieval context"""

    entity_id: str
    name: str
    entity_type: str
    value: Any
    description: str
    confidence: float
    mention_count: int
    page_numbers: List[int]
    relevance_score: float  # 0-1 how relevant to query
    is_initial: bool  # Found in initial search vs through relationships

    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "value": str(self.value),
            "description": self.description,
            "confidence": self.confidence,
            "mention_count": self.mention_count,
            "page_numbers": self.page_numbers,
            "relevance_score": self.relevance_score,
            "is_initial": self.is_initial,
        }


@dataclass
class RetrievedRelationship:
    """Relationship with full context"""

    source_entity: str
    target_entity: str
    source_name: str
    target_name: str
    relationship_type: str
    description: str
    weight: float
    evidence_text: str
    spatial_cooccurrence: bool
    relevance_score: float

    def to_dict(self) -> Dict:
        return {
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "source_name": self.source_name,
            "target_name": self.target_name,
            "relationship_type": self.relationship_type,
            "description": self.description,
            "weight": self.weight,
            "evidence_text": self.evidence_text,
            "spatial_cooccurrence": self.spatial_cooccurrence,
            "relevance_score": self.relevance_score,
        }


@dataclass
class RetrievedRegion:
    """Visual region with retrieval context"""

    region_id: str
    page_num: int
    block_type: str
    text_content: str
    markdown_content: str
    relevance_score: float
    bbox: List[float]
    entities_in_region: List[str]  # Entity IDs found in this region
    retrieval_method: str = "entity"  # 'entity', 'text', 'visual'
    image_path: Optional[str] = None  # Path to extracted image

    def to_dict(self) -> Dict:
        return {
            "region_id": self.region_id,
            "page_num": self.page_num,
            "block_type": self.block_type,
            "text_content": self.text_content,
            "markdown_content": self.markdown_content,
            "relevance_score": self.relevance_score,
            "bbox": self.bbox,
            "entities_in_region": self.entities_in_region,
            "retrieval_method": self.retrieval_method,
            "image_path": self.image_path,
        }


class AdaptiveRetriever:
    """
    Adaptive Token Budgeting Retrieval System
    Retrieves optimal context based on query complexity
    """

    def __init__(
        self, dual_layer_graph: DualLayerGraph, query_classifier: Optional[QueryClassifier] = None, config: Optional[Dict] = None
    ):
        """
        Initialize retriever

        Args:
            dual_layer_graph: Dual-layer graph to retrieve from
            query_classifier: Query classifier instance
            config: Configuration dictionary for factory
        """
        self.graph = dual_layer_graph
        self.classifier = query_classifier or QueryClassifier()
        self.config = config or {}
        
        # Initialize Formatter via Factory
        # This allows swapping Toon/JSON/Markdown formatters
        self.formatter = component_factory.create_formatter(self.config)
        
        # Initialize TF-IDF index for hybrid search
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []
        self._build_text_index()

    def _build_text_index(self):
        """Build TF-IDF index from graph regions"""
        if not SKLEARN_AVAILABLE:
            return

        documents = []
        self.doc_ids = []

        for region_id, node in self.graph.visual_spatial.nodes.items():
            if node.region.text_content and len(node.region.text_content.strip()) > 10:
                documents.append(node.region.text_content)
                self.doc_ids.append(region_id)
        
        if documents:
            try:
                # Limit features to keep it fast and light
                self.vectorizer = TfidfVectorizer(
                    stop_words='english', 
                    max_features=5000,
                    ngram_range=(1, 2)
                )
                self.tfidf_matrix = self.vectorizer.fit_transform(documents)
                print(f"  ✅ Built TF-IDF index with {len(documents)} documents")
            except Exception as e:
                print(f"Warning: Failed to build TF-IDF index: {e}")
                self.vectorizer = None

    def retrieve(self, query: str, override_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve context for query with adaptive token budgeting

        Args:
            query: User query
            override_level: Override automatic classification

        Returns:
            Retrieved context with structured metadata including:
            - context: Formatted markdown for LLM
            - entities: Structured entity data with relevance scores
            - relationships: Relationship objects with evidence
            - regions: Visual regions with entities
            - metadata: Token counts, node counts, strategy info
        """
        # Classify query
        if override_level:
            level = self.classifier.levels[override_level]
        else:
            level = self.classifier.classify(query)

        print(f"Query Level: {level.level} ({level.name})")
        print(f"Token Budget: {level.max_tokens}")
        print(f"Strategy: {level.strategy}")

        # Apply retrieval strategy
        if level.strategy == "entity_lookup":
            result = self._entity_lookup(query, level)
        elif level.strategy == "hybrid":
            result = self._hybrid_retrieval(query, level)
        elif level.strategy == "hierarchical":
            result = self._hierarchical_retrieval(query, level)
        elif level.strategy == "visual_fusion":
            result = self._visual_fusion_retrieval(query, level)
        else:
            result = self._entity_lookup(query, level)

        # Add metadata
        result["query_level"] = level.level
        result["strategy"] = level.strategy
        result["max_tokens"] = level.max_tokens

        # Print retrieval summary
        self._print_retrieval_summary(result)

        return result

    def _print_retrieval_summary(self, result: Dict[str, Any]):
        """Print summary of retrieved information"""
        print(f"\n[Retrieval Summary]")
        print(f"  Entities: {len(result.get('entities', []))} retrieved")
        print(f"  Relationships: {len(result.get('relationships', []))} retrieved")
        print(f"  Regions: {len(result.get('regions', []))} retrieved")
        print(f"  Tokens: {result.get('token_count', 0)}/{result.get('max_tokens', 0)}")

        # Show top entities
        entities = result.get("entities", [])
        if entities:
            top_entities = sorted(
                entities, key=lambda e: e.get("relevance_score", 0), reverse=True
            )[:3]
            print(f"  Top entities: {', '.join([e['name'] for e in top_entities])}")

        # Show relationships
        rels = result.get("relationships", [])
        if rels:
            print(f"  Sample relationships: {len(rels)} total")

    def _search_entities_with_ranking(
        self, query: str, max_entities: int, include_relationships: bool = False
    ) -> Tuple[List[RetrievedEntity], List[RetrievedRelationship]]:
        """
        Search for entities with relevance ranking

        Args:
            query: Search query
            max_entities: Max entities to return
            include_relationships: Whether to include relationships

        Returns:
            Tuple of (entities, relationships)
        """
        # Search entities
        found_entities = self.graph.entity_relationship.search_entities(
            query, top_k=max_entities * 2  # Get more to rank
        )

        # Rank by relevance (combination of search score and mention count)
        retrieved_entities = []
        entity_ids = set()

        for entity in found_entities[:max_entities]:
            # Relevance score combines search score and mention frequency
            relevance = min(1.0, (entity.mention_count * 0.3 + entity.confidence * 0.7))

            retrieved = RetrievedEntity(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                value=entity.value,
                description=entity.description,
                confidence=entity.confidence,
                mention_count=entity.mention_count,
                page_numbers=entity.page_numbers,
                relevance_score=relevance,
                is_initial=True,
            )
            retrieved_entities.append(retrieved)
            entity_ids.add(entity.entity_id)

        # Find relationships involving these entities
        retrieved_relationships = []
        if include_relationships and entity_ids:
            for rel in self.graph.entity_relationship.relationships:
                # Include relationship if at least one entity is in our set
                if rel.source_entity in entity_ids or rel.target_entity in entity_ids:
                    source = self.graph.entity_relationship.get_entity(rel.source_entity)
                    target = self.graph.entity_relationship.get_entity(rel.target_entity)

                    if source and target:
                        # Prioritize relationships where both entities are in set
                        both_in_set = (rel.source_entity in entity_ids and rel.target_entity in entity_ids)
                        relevance = rel.weight * (1.0 if both_in_set else 0.7)
                        
                        retrieved_rel = RetrievedRelationship(
                            source_entity=rel.source_entity,
                            target_entity=rel.target_entity,
                            source_name=source.name,
                            target_name=target.name,
                            relationship_type=rel.relationship_type,
                            description=rel.description,
                            weight=rel.weight,
                            evidence_text=rel.evidence_text,
                            spatial_cooccurrence=rel.spatial_cooccurrence,
                            relevance_score=relevance,
                        )
                        retrieved_relationships.append(retrieved_rel)
            
            # Sort by relevance and limit
            retrieved_relationships.sort(key=lambda r: r.relevance_score, reverse=True)
            retrieved_relationships = retrieved_relationships[:20]  # Limit to top 20

        return retrieved_entities, retrieved_relationships

    def _search_regions_with_ranking(
        self, query: str, max_regions: int, entity_ids: Optional[set] = None
    ) -> List[RetrievedRegion]:
        """
        Search for visual regions with relevance ranking

        Args:
            query: Search query
            max_regions: Max regions to return
            entity_ids: Optional set of entity IDs to prioritize

        Returns:
            List of retrieved regions
        """
        query_lower = query.lower()
        query_words = set(word for word in query_lower.split() if len(word) > 3)

        scored_regions = []

        for region_id, node in self.graph.visual_spatial.nodes.items():
            text = node.region.text_content.lower()

            # Score based on keyword matches
            matches = sum(1 for word in query_words if word in text)
            if matches == 0:
                continue

            keyword_score = min(1.0, matches / len(query_words)) if query_words else 0

            # Boost score if region contains relevant entities
            entity_boost = 0
            entities_in_region = self.graph.get_entities_in_region(region_id)
            entities_in_region_ids = [e.entity_id for e in entities_in_region]

            if entity_ids:
                matching_entities = [e for e in entities_in_region if e.entity_id in entity_ids]
                if matching_entities:
                    entity_boost = min(0.5, len(matching_entities) * 0.1)

            # Combined relevance score
            relevance = min(1.0, keyword_score + entity_boost)

            scored_regions.append((relevance, region_id, node, entities_in_region_ids))

        # Sort by relevance and return top regions
        scored_regions.sort(reverse=True, key=lambda x: x[0])

        retrieved_regions = []
        for relevance, region_id, node, entities_ids in scored_regions[:max_regions]:
            region = node.region
            retrieved = RetrievedRegion(
                region_id=region_id,
                page_num=node.page_num,
                block_type=region.block_type,
                text_content=region.text_content,
                markdown_content=region.markdown_content,
                relevance_score=relevance,
                bbox=region.bbox.to_list(),
                entities_in_region=entities_ids,
                retrieval_method="entity",
                image_path=region.image_path if hasattr(region, "image_path") else None,
            )
            retrieved_regions.append(retrieved)

        return retrieved_regions
        
    def _search_text_tfidf(self, query: str, max_results: int = 5) -> List[RetrievedRegion]:
        """Search regions using TF-IDF sparse vector retrieval"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
            
        try:
            query_vec = self.vectorizer.transform([query])
            
            # Compute similarity
            similarity_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
            
            # Get top indices
            # Only consider scores > 0
            relevant_indices = np.where(similarity_scores > 0.05)[0]
            if len(relevant_indices) == 0:
                return []
                
            # Sort indices by score
            sorted_indices = relevant_indices[np.argsort(similarity_scores[relevant_indices])[::-1]]
            top_indices = sorted_indices[:max_results]
            
            retrieved_regions = []
            for idx in top_indices:
                region_id = self.doc_ids[idx]
                if region_id in self.graph.visual_spatial.nodes:
                    node = self.graph.visual_spatial.nodes[region_id]
                    region = node.region
                    entities_in_region = [
                        e.entity_id for e in self.graph.get_entities_in_region(region_id)
                    ]
                    
                    retrieved = RetrievedRegion(
                        region_id=region.region_id,
                        page_num=node.page_num,
                        block_type=region.block_type,
                        text_content=region.text_content,
                        markdown_content=region.markdown_content,
                        relevance_score=float(similarity_scores[idx]),
                        bbox=region.bbox.to_list(),
                        entities_in_region=entities_in_region,
                        retrieval_method="text_tfidf",
                        image_path=region.image_path if hasattr(region, "image_path") else None,
                    )
                    retrieved_regions.append(retrieved)
                    
            return retrieved_regions
            
        except Exception as e:
            print(f"Error in TF-IDF search: {e}")
            return []

    def _generate_context(self, result: Dict[str, Any], level: QueryLevel) -> str:
        """Generate formatted context string using configured formatter"""
        if level.level == 1:
            return self.formatter.format_simple_context(result)
        else:
            return self.formatter.format_retrieval_result(result)

    def _entity_lookup(self, query: str, level: QueryLevel) -> Dict[str, Any]:
        """
        Level 1: Simple entity lookup (Hybrid with fallbacks)

        Fast retrieval for factual queries
        """
        # 1. Search for relevant entities with ranking
        entities, relationships = self._search_entities_with_ranking(
            query, level.max_nodes, include_relationships=True
        )

        # 2. Get regions for these entities
        entity_ids = set(e.entity_id for e in entities) if entities else None
        
        if entities:
            # Use entity-driven region search
            regions = self._search_regions_with_ranking(query, min(3, level.max_nodes), entity_ids)
        else:
            regions = []

        # 3. Fallback: If minimal entities/regions found, use TF-IDF
        use_fallback = not entities or len(regions) == 0
        if use_fallback:
            print("  ℹ️ Entity search yielded low results, trying Hybrid text search...")
            text_regions = self._search_text_tfidf(query, max_results=5)
            
            # Deduplicate regions
            existing_ids = {r.region_id for r in regions}
            for tr in text_regions:
                if tr.region_id not in existing_ids:
                    regions.append(tr)
                    existing_ids.add(tr.region_id)
            
            # If we found regions via text, try to extract relevant entities from them to populate context
            if text_regions and not entities:
                for region in text_regions:
                    if region.entities_in_region:
                        for eid in region.entities_in_region:
                            # Add these entities to the result if we have room
                            if len(entities) < 5:
                                e_obj = self.graph.entity_relationship.get_entity(eid)
                                if e_obj:
                                    # Create partial retrieved entity
                                    retrieved = RetrievedEntity(
                                        entity_id=e_obj.entity_id,
                                        name=e_obj.name,
                                        entity_type=e_obj.entity_type,
                                        value=e_obj.value,
                                        description=e_obj.description,
                                        confidence=e_obj.confidence,
                                        mention_count=e_obj.mention_count,
                                        page_numbers=e_obj.page_numbers,
                                        relevance_score=0.5, # Lower confidence as inferred from region
                                        is_initial=False,
                                    )
                                    # Ensure unique
                                    if not any(e.entity_id == retrieved.entity_id for e in entities):
                                        entities.append(retrieved)

        # Build context
        context_parts = []
        total_tokens = 0

        # If STILL no entities/regions, return empty
        if not entities and not regions:
            return {
                "context": "",
                "entities": [],
                "relationships": [],
                "regions": [],
                "token_count": 0,
                "nodes_retrieved": 0,
                "error": "No relevant entities or text found for query"
            }

        if entities:
            # Use entity-based context with better formatting
            context_parts.append("## Key Entities\n\n")
            for entity in entities:
                entity_context = (
                    f"**{entity.name}** ({entity.entity_type})\n"
                    f"- Value: {entity.value}\n"
                    f"- Description: {entity.description}\n"
                    f"- Confidence: {entity.confidence:.1%}\n"
                    f"- Mentions: {entity.mention_count}\n"
                    f"- Pages: {', '.join(map(str, entity.page_numbers))}\n\n"
                )
                context_parts.append(entity_context)
                total_tokens += len(entity_context) // 4

                if total_tokens >= level.max_tokens:
                    break
        
        # Add relationships if available
        if relationships and total_tokens < level.max_tokens:
            context_parts.append("\n## Entity Relationships\n\n")
            for rel in relationships[:10]:  # Limit to top 10 relationships
                rel_text = (
                    f"- **{rel.source_name}** → _{rel.relationship_type}_ → **{rel.target_name}**\n"
                )
                if rel.description and rel.description != rel.relationship_type:
                    rel_text += f"  {rel.description}\n"
                context_parts.append(rel_text)
                total_tokens += len(rel_text) // 4
                
                if total_tokens >= level.max_tokens:
                    break
            context_parts.append("\n")
        
        # Add region content with visual-spatial information
        if regions and total_tokens < level.max_tokens:
            context_parts.append("\n## Document Content & Visual Layout\n\n")
            for region in regions:
                # Include spatial information
                bbox_str = f"[x:{region.bbox[0]:.0f}, y:{region.bbox[1]:.0f}, w:{region.bbox[2]-region.bbox[0]:.0f}, h:{region.bbox[3]-region.bbox[1]:.0f}]"
                
                source_marker = " (Text Match)" if region.retrieval_method == "text_tfidf" else ""
                
                region_text = (
                    f"**[{region.block_type}]** (Page {region.page_num}) {bbox_str}{source_marker}\n"
                    f"{region.markdown_content}\n"
                )
                
                # Check for image
                if region.image_path:
                    region_text += f"\n![Region Image]({region.image_path})\n"
                
                # Show entities in this region
                if region.entities_in_region:
                    entities_names = []
                    for eid in region.entities_in_region:
                        entity_obj = self.graph.entity_relationship.get_entity(eid)
                        if entity_obj:
                            entities_names.append(entity_obj.name)
                    if entities_names:
                        region_text += f"_Contains entities: {', '.join(entities_names)}_\n"
                
                region_text += "\n"
                context_parts.append(region_text)
                total_tokens += len(region_text) // 4

                if total_tokens >= level.max_tokens:
                    break

        return {
            "context": "".join(context_parts),
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships],
            "regions": [r.to_dict() for r in regions],
            "token_count": total_tokens,
            "nodes_retrieved": len(entities) + len(regions),
        }

    def _search_visual_regions(self, query: str, max_regions: int = 10) -> List:
        """Search visual regions by text similarity"""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_regions = []
        for region_id, node in self.graph.visual_spatial.nodes.items():
            text = node.region.text_content.lower()

            # Simple keyword matching score
            matches = sum(1 for word in query_words if word in text and len(word) > 3)
            if matches > 0:
                score = matches / len(query_words) if query_words else 0
                scored_regions.append((score, region_id, node))

        # Sort by score and return top regions
        scored_regions.sort(reverse=True, key=lambda x: x[0])
        return [(rid, node) for _, rid, node in scored_regions[:max_regions]]

    def _hybrid_retrieval(self, query: str, level: QueryLevel) -> Dict[str, Any]:
        """
        Level 2: Hybrid entity + relationship traversal
        Now enhanced with Hybrid Text Search
        """
        # 1. Get initial entities with ranking
        entities, _ = self._search_entities_with_ranking(
            query, min(5, level.max_nodes), include_relationships=False
        )
        
        # 2. Hybrid Augmentation: Search Text If needed
        text_regions = []
        if not entities or len(entities) < 3:
             print("  ℹ️ Transforming query for Hybrid Text Search...")
             text_regions = self._search_text_tfidf(query, max_results=5)
             
             # Extract entities from these regions to seed graph traversal
             for region in text_regions:
                for eid in region.entities_in_region:
                     if len(entities) < 5:
                        e_obj = self.graph.entity_relationship.get_entity(eid)
                        if e_obj and not any(e.entity_id == e_obj.entity_id for e in entities):
                            # Create inferred entity
                            retrieved = RetrievedEntity(
                                entity_id=e_obj.entity_id,
                                name=e_obj.name,
                                entity_type=e_obj.entity_type,
                                value=e_obj.value,
                                description=e_obj.description,
                                confidence=e_obj.confidence,
                                mention_count=e_obj.mention_count,
                                page_numbers=e_obj.page_numbers,
                                relevance_score=0.6,
                                is_initial=True, 
                            )
                            entities.append(retrieved)

        if not entities and not text_regions:
            return {
                "context": "",
                "entities": [],
                "relationships": [],
                "regions": [],
                "token_count": 0,
                "nodes_retrieved": 0,
                "error": "No relevant entities found for hybrid retrieval"
            }

        # Expand through relationships (2-hop)
        entity_ids = set(e.entity_id for e in entities)
        entity_id_to_hop = {e.entity_id: 0 for e in entities}

        for entity_id in list(entity_ids):
            related = self.graph.entity_relationship.get_related_entities(entity_id, hop_distance=2)
            for rel_id in related:
                if rel_id not in entity_ids:
                    entity_ids.add(rel_id)
                    entity_id_to_hop[rel_id] = 2

        # Limit to max nodes and maintain initial entities
        all_entity_ids = list(entity_ids)[: level.max_nodes]

        # Build retrieved entity objects with hop information
        retrieved_entities = []
        for eid in all_entity_ids:
            entity = self.graph.entity_relationship.get_entity(eid)
            if entity:
                is_initial = entity_id_to_hop.get(eid, 0) == 0
                relevance = min(1.0, (entity.mention_count * 0.3 + entity.confidence * 0.7))
                retrieved = RetrievedEntity(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    value=entity.value,
                    description=entity.description,
                    confidence=entity.confidence,
                    mention_count=entity.mention_count,
                    page_numbers=entity.page_numbers,
                    relevance_score=relevance,
                    is_initial=is_initial,
                )
                retrieved_entities.append(retrieved)

        # Get relationships with full evidence
        retrieved_relationships = []
        for rel in self.graph.entity_relationship.relationships:
            if rel.source_entity in all_entity_ids and rel.target_entity in all_entity_ids:
                source = self.graph.entity_relationship.get_entity(rel.source_entity)
                target = self.graph.entity_relationship.get_entity(rel.target_entity)

                if source and target:
                    retrieved_rel = RetrievedRelationship(
                        source_entity=rel.source_entity,
                        target_entity=rel.target_entity,
                        source_name=source.name,
                        target_name=target.name,
                        relationship_type=rel.relationship_type,
                        description=rel.description,
                        weight=rel.weight,
                        evidence_text=rel.evidence_text,
                        spatial_cooccurrence=rel.spatial_cooccurrence,
                        relevance_score=rel.weight,
                    )
                    retrieved_relationships.append(retrieved_rel)

        # Get visual regions for these entities
        regions = self._search_regions_with_ranking(
            query, min(5, level.max_nodes), set(all_entity_ids)
        )
        
        # Merge with Text-based regions found earlier
        existing_rids = {r.region_id for r in regions}
        for tr in text_regions:
            if tr.region_id not in existing_rids:
                regions.append(tr)
                existing_rids.add(tr.region_id)

        # Build context with full relationship information
        context_parts = []
        total_tokens = 0

        # Add entity information
        context_parts.append("## Key Entities\n\n")
        for entity in retrieved_entities:
            entity_marker = "🔵" if entity.is_initial else "⚪"
            entity_context = (
                f"{entity_marker} **{entity.name}** ({entity.entity_type})\n"
                f"- Description: {entity.description}\n"
                f"- Confidence: {entity.confidence:.1%}\n"
                f"- Mentions: {entity.mention_count}\n\n"
            )
            context_parts.append(entity_context)
            total_tokens += len(entity_context) // 4

        # Add relationships with evidence
        if retrieved_relationships:
            context_parts.append("## Entity Relationships\n\n")
            for rel in retrieved_relationships:
                rel_context = (
                    f"**{rel.source_name}** --[{rel.relationship_type}]--> **{rel.target_name}**\n"
                    f"- Description: {rel.description}\n"
                    f"- Evidence: {rel.evidence_text}\n"
                    f"- Co-occurrence: {'Yes' if rel.spatial_cooccurrence else 'No'}\n\n"
                )
                context_parts.append(rel_context)
                total_tokens += len(rel_context) // 4

        # Add source text from regions
        if regions:
            context_parts.append("## Source Documents\n\n")
            for region in regions:
                region_context = (
                    f"[{region.block_type}] Page {region.page_num} "
                    f"(relevance: {region.relevance_score:.1%})\n"
                    f"{region.text_content[:300]}\n\n"
                )
                context_parts.append(region_context)
                total_tokens += len(region_context) // 4

                if total_tokens >= level.max_tokens:
                    break

        return {
            "context": "".join(context_parts),
            "entities": [e.to_dict() for e in retrieved_entities],
            "relationships": [r.to_dict() for r in retrieved_relationships],
            "regions": [r.to_dict() for r in regions],
            "token_count": total_tokens,
            "nodes_retrieved": len(retrieved_entities) + len(regions),
        }

    def _hierarchical_retrieval(self, query: str, level: QueryLevel) -> Dict[str, Any]:
        """
        Level 3: Hierarchical cross-document retrieval

        For multi-document synthesis and comparison
        """
        # Get entities mentioned in query
        entities = self.graph.entity_relationship.search_entities(query, top_k=10)

        # Group entities by type for comparison
        entity_groups = {}
        for entity in entities:
            if entity.entity_type not in entity_groups:
                entity_groups[entity.entity_type] = []
            entity_groups[entity.entity_type].append(entity)

        # Build hierarchical context
        context_parts = []
        total_tokens = 0

        context_parts.append("# Multi-Document Analysis\n\n")

        # Add entities by type
        for entity_type, group_entities in entity_groups.items():
            context_parts.append(f"## {entity_type.upper()} Entities\n")
            for entity in group_entities[:10]:
                entity_text = f"- **{entity.name}**: {entity.description}\n"
                context_parts.append(entity_text)
                total_tokens += len(entity_text) // 4

                # Add cross-references
                related = self.graph.entity_relationship.get_related_entities(
                    entity.entity_id, hop_distance=1
                )
                for rel_id in related[:3]:
                    rel_entity = self.graph.entity_relationship.get_entity(rel_id)
                    if rel_entity:
                        ref_text = f"  - Related: {rel_entity.name}\n"
                        context_parts.append(ref_text)
                        total_tokens += len(ref_text) // 4

        # Add comprehensive source regions
        context_parts.append("\n## Source Documents\n")
        all_regions = set()
        for entity in entities:
            all_regions.update(self.graph.entity_to_regions.get(entity.entity_id, []))

        for rid in list(all_regions)[:8]:
            if rid in self.graph.visual_spatial.nodes:
                node = self.graph.visual_spatial.nodes[rid]
                doc_text = f"### Page {node.page_num} [{node.region.block_type}]\n"
                doc_text += f"{node.region.markdown_content}\n"
                context_parts.append(doc_text)
                total_tokens += len(doc_text) // 4

                if total_tokens >= level.max_tokens:
                    break

        return {
            "context": "".join(context_parts),
            "entities": [e.to_dict() for e in entities],
            "relationships": [],  # Hierarchical focuses on entity groups, not relationships
            "regions": list(all_regions),
            "token_count": total_tokens,
            "nodes_retrieved": len(entities) + len(all_regions),
            "entity_groups": {k: len(v) for k, v in entity_groups.items()},
        }

    def _visual_fusion_retrieval(self, query: str, level: QueryLevel) -> Dict[str, Any]:
        """
        Level 4: Visual-Semantic Fusion

        Combines visual regions with entity context
        """
        context_parts = []
        total_tokens = 0

        # Extract page/figure references from query
        import re

        page_refs = re.findall(r"page\s+(\d+)", query.lower())
        figure_refs = re.findall(r"figure\s+(\d+)", query.lower())
        table_refs = re.findall(r"table\s+(\d+)", query.lower())

        # Get specific pages if mentioned
        target_pages = [int(p) for p in page_refs]

        # Get visual regions
        visual_regions = []
        if target_pages:
            for page_num in target_pages:
                if page_num in self.graph.visual_spatial.page_nodes:
                    for node_id in self.graph.visual_spatial.page_nodes[page_num]:
                        visual_regions.append(self.graph.visual_spatial.nodes[node_id])
        else:
            # Search by content type
            for node in self.graph.visual_spatial.nodes.values():
                if node.region.block_type in ["figure", "table", "chart"]:
                    visual_regions.append(node)

        context_parts.append("# Visual Content Analysis\n\n")

        # Add visual regions with full detail
        for region in visual_regions[: level.max_nodes]:
            context_parts.append(
                f"## {region.region.block_type.upper()} (Page {region.page_num})\n"
            )
            context_parts.append(f"**Location**: {region.region.bbox.to_list()}\n")
            context_parts.append(f"**Content**:\n{region.region.markdown_content}\n")

            # Get caption if figure
            if region.region.block_type == "figure":
                caption_node = self.graph.get_caption_for_figure(region.node_id)
                if caption_node:
                    context_parts.append(f"**Caption**: {caption_node.region.text_content}\n")

            # Get entities in this region
            entities_in_region = self.graph.get_entities_in_region(region.node_id)
            if entities_in_region:
                context_parts.append("**Entities**:\n")
                for entity in entities_in_region[:5]:
                    context_parts.append(f"- {entity.name} ({entity.entity_type})\n")

            context_parts.append("\n")

            # Estimate tokens
            for part in context_parts[-8:]:
                total_tokens += len(part) // 4

            if total_tokens >= level.max_tokens:
                break

        # Add relevant entities from query
        context_parts.append("## Related Entities\n")
        entities = self.graph.entity_relationship.search_entities(query, top_k=10)
        for entity in entities:
            entity_text = f"- **{entity.name}**: {entity.description}\n"
            context_parts.append(entity_text)
            total_tokens += len(entity_text) // 4

        return {
            "context": "".join(context_parts),
            "entities": [e.to_dict() for e in entities],
            "relationships": [],  # Visual fusion doesn't traverse relationships
            "regions": [r.node_id for r in visual_regions],
            "token_count": total_tokens,
            "nodes_retrieved": len(visual_regions) + len(entities),
            "visual_focus": True,
        }

    def compress_context(self, context: str, max_tokens: int) -> str:
        """
        Apply path compression to reduce token usage

        Args:
            context: Original context
            max_tokens: Maximum allowed tokens

        Returns:
            Compressed context
        """
        # Estimate current tokens
        current_tokens = len(context) // 4

        if current_tokens <= max_tokens:
            return context

        # Apply compression strategies
        # 1. Remove redundant whitespace
        context = " ".join(context.split())

        # 2. Truncate long sections
        lines = context.split("\n")
        compressed_lines = []
        tokens_used = 0

        for line in lines:
            line_tokens = len(line) // 4
            if tokens_used + line_tokens <= max_tokens:
                compressed_lines.append(line)
                tokens_used += line_tokens
            else:
                # Truncate line
                remaining = max_tokens - tokens_used
                if remaining > 10:
                    compressed_lines.append(line[: remaining * 4] + "...")
                break

        return "\n".join(compressed_lines)

    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics"""
        return {
            "graph_stats": {
                "visual_nodes": len(self.graph.visual_spatial.nodes),
                "entity_nodes": len(self.graph.entity_relationship.entities),
                "relationships": len(self.graph.entity_relationship.relationships),
            },
            "query_levels": {
                level.level: {
                    "name": level.name,
                    "max_tokens": level.max_tokens,
                    "strategy": level.strategy,
                }
                for level in self.classifier.levels.values()
            },
        }
