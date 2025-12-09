"""
Query Classification for Adaptive Retrieval
Classifies queries into 4 levels based on complexity using SetFit
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from setfit import SetFitModel


logger = logging.getLogger(__name__)


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
    SetFit-based Query Classification
    Classifies queries into 4 complexity levels using few-shot learning
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2", device: str = "cpu", model_save_path: Optional[str] = None):
        """
        Initialize SetFit classifier
        
        Args:
            model_name: Base sentence transformer model
            device: Device to run model on (cpu, cuda, mps)
            model_save_path: Path to load/save the trained model
        """
        self.device = device
        self.model_name = model_name
        self.model_save_path = model_save_path or "./setfit_query_classifier_save"
        self.model: Optional[SetFitModel] = None
        
        self.levels = {
            0: QueryLevel(
                level=1,
                name="simple",
                max_tokens=2000,
                max_nodes=5,
                strategy="entity_lookup",
                description="Simple factual queries",
            ),
            1: QueryLevel(
                level=2,
                name="complex",
                max_tokens=6000,
                max_nodes=20,
                strategy="hybrid",
                description="Complex reasoning queries",
            ),
            2: QueryLevel(
                level=3,
                name="multi_doc",
                max_tokens=10000,
                max_nodes=50,
                strategy="hierarchical",
                description="Multi-document synthesis",
            ),
            3: QueryLevel(
                level=4,
                name="visual",
                max_tokens=12000,
                max_nodes=30,
                strategy="visual_fusion",
                description="Visual-semantic fusion queries",
            ),
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or train SetFit model with few-shot examples"""
        import os
        
        try:
            from datasets import Dataset
            from setfit import Trainer, TrainingArguments
            
            # Check if saved model exists
            if os.path.exists(self.model_save_path) and os.path.isdir(self.model_save_path):
                try:
                    self.model = SetFitModel.from_pretrained(self.model_save_path)
                    logger.info(f"Loaded existing SetFit classifier from {self.model_save_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load saved model, retraining: {e}")

            # Create dataset with training examples
            train_examples = self._get_training_examples()
            train_dataset = Dataset.from_dict({
                "text": [ex[0] for ex in train_examples],
                "label": [ex[1] for ex in train_examples]
            })
            
            # Initialize SetFit model
            self.model = SetFitModel.from_pretrained(
                self.model_name,
                labels=["simple", "complex", "multi_doc", "visual"]
            )
            logger.info(f"Loaded base SetFit model: {self.model_name}")
            
            # Train with few-shot examples
            args = TrainingArguments(
                num_epochs=1,
                batch_size=8,
                show_progress_bar=False,
                output_dir="./setfit_training_output" # Temporary output
            )
            
            trainer = Trainer(
                model=self.model,
                args=args,
                train_dataset=train_dataset
            )
            
            trainer.train()
            logger.info("Trained SetFit classifier with few-shot examples")
            
            # Save the model
            self.model.save_pretrained(self.model_save_path)
            logger.info(f"Saved trained classifier to {self.model_save_path}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize SetFit model: {e}")
            self.model = None
    
    def _get_training_examples(self):
        """Get few-shot training examples for each query level"""
        examples = [
            # Level 0: Simple factual queries
            ("What is the main topic?", 0),
            ("Who is the author?", 0),
            ("Define the term", 0),
            ("When was this published?", 0),
            ("List the key points", 0),
            ("What are the main entities?", 0),
            ("Which method is used?", 0),
            ("Name the components", 0),
            
            # Level 1: Complex reasoning queries  
            ("Why does this approach work better?", 1),
            ("How do these components interact?", 1),
            ("Explain the relationship between X and Y", 1),
            ("Analyze the impact of this decision", 1),
            ("What causes this effect?", 1),
            ("How does the architecture achieve performance?", 1),
            ("Explain the reasoning behind this design", 1),
            ("What is the correlation between these factors?", 1),
            
            # Level 2: Multi-document synthesis
            ("Compare approach A versus approach B", 2),
            ("What are the differences between these methods?", 2),
            ("Summarize all the experiments", 2),
            ("How do multiple papers address this problem?", 2),
            ("Compare the results across all studies", 2),
            ("What patterns emerge across the documents?", 2),
            ("Synthesize findings from all sections", 2),
            ("Contrast different architectures presented", 2),
            
            # Level 3: Visual-semantic fusion
            ("What does Figure 1 show?", 3),
            ("Describe the chart on page 5", 3),
            ("Explain the diagram", 3),
            ("What is shown in the table?", 3),
            ("Analyze the visualization", 3),
            ("What trends appear in the plot?", 3),
            ("Describe the image layout", 3),
            ("What is displayed in the screenshot?", 3),
        ]
        return examples

    def classify(self, query: str) -> QueryLevel:
        """
        Classify query into complexity level using SetFit

        Args:
            query: User query string

        Returns:
            QueryLevel with appropriate settings
        """
        if self.model is None:
            logger.warning("SetFit model not initialized, using simple fallback")
            return self.levels[0]  # Default to simple
        
        try:
            # Predict using SetFit
            prediction = self.model.predict([query])[0]
            
            # Map string labels to indices if needed
            if isinstance(prediction, str):
                label_map = {"simple": 0, "complex": 1, "multi_doc": 2, "visual": 3}
                prediction = label_map.get(prediction, 0)
            
            return self.levels[int(prediction)]
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return self.levels[0]  # Fallback to simple

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
        
        # Get prediction confidence if available
        confidence = None
        if self.model is not None:
            try:
                probs = self.model.predict_proba([query])[0]
                confidence = float(max(probs))
            except:
                pass

        return {
            "query": query,
            "level": level.level,
            "level_name": level.name,
            "max_tokens": level.max_tokens,
            "max_nodes": level.max_nodes,
            "strategy": level.strategy,
            "description": level.description,
            "confidence": confidence,
            "word_count": len(query.split()),
            "model": "SetFit" if self.model else "Fallback",
            "token_budget": f"{level.max_tokens} tokens (vs 30K fixed in LightRAG)",
        }
