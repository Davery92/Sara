import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import os
import logging
import sys

# Configure logger
logger = logging.getLogger("intent-classifier")

class IntentClassifier:
    def __init__(self, model_dir='./improved_model/'):
        """Initialize the intent classifier with the specified model"""
        self.model = None
        self.tokenizer = None
        self.label_dict = {}
        self.id_to_label = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        self.load_model(model_dir)
    
    def load_model(self, model_dir):
        """Load the model, tokenizer, and label dictionaries"""
        logger.info(f"Loading intent classification model from {model_dir}")
        
        # Check if the model directory exists
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory {model_dir} not found.")
            # Try the original model directory
            model_dir = './tool_classification_model/'
            if not os.path.exists(model_dir):
                logger.error("Neither improved nor original model directory found.")
                return False
            logger.info(f"Using original model from {model_dir} instead.")
        
        # Load the model, tokenizer, and label dictionaries
        try:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            
            # Load label dictionaries
            with open(os.path.join(model_dir, 'label_dict.json'), 'r') as f:
                self.label_dict = json.load(f)
            
            with open(os.path.join(model_dir, 'id_to_label.json'), 'r') as f:
                self.id_to_label = json.load(f)
                # Convert string keys to integers
                self.id_to_label = {int(k): v for k, v in self.id_to_label.items()}
            
            # Move model to device
            self.model.to(self.device)
            logger.info(f"Model loaded successfully with {len(self.label_dict)} intent classes on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_intent(self, query, top_k=3):
        """
        Predict the intent for a given query.
        
        Args:
            query (str): The user query to classify
            top_k (int): Number of top predictions to return
            
        Returns:
            dict: Dictionary containing top predictions with probabilities and features
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded")
            return {"error": "Model not loaded", "predictions": []}
        
        try:
            # Process the query
            encoding = self.tokenizer(
                query,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                
                # Get top k predictions
                top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.id_to_label)))
                
                # Convert to lists
                top_probs = top_probs.cpu().tolist()
                top_indices = top_indices.cpu().tolist()
            
            # Format results
            predictions = []
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                intent = self.id_to_label[idx]
                predictions.append({
                    "intent": intent,
                    "probability": prob,
                    "percentage": prob * 100
                })
            
            # Analyze features in the query
            features = []
            if "remember" in query.lower():
                features.append("contains 'remember'")
            if any(word in query.lower() for word in ["tomorrow", "today", "tonight", "later"]):
                features.append("refers to future time")
            if any(char.isdigit() for char in query):
                features.append("contains numbers")
            
            return {
                "predictions": predictions,
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Error predicting intent: {e}")
            return {"error": str(e), "predictions": []}
    
    def get_available_intents(self):
        """Return a list of all available intents"""
        return list(self.label_dict.keys())

# Create a singleton instance
_classifier = None

def get_intent_classifier(model_dir='./improved_model/'):
    """Get or create the intent classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier(model_dir)
    return _classifier
