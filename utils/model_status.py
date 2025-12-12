"""Model status and metadata utilities"""

from utils.model_manager import ModelManager
from utils.training_data import TrainingDataManager
from utils.config import MODELS_DIR
from pathlib import Path
import json

class ModelStatus:
    """Handles model status queries"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_manager = ModelManager()
        self.data_manager = TrainingDataManager(model_name)
    
    def describe(self):
        """Generate comprehensive model description"""
        metadata = self.model_manager.get_model_metadata(self.model_name)
        
        if not metadata:
            return {
                "error": f"Model '{self.model_name}' not found"
            }
        
        # Get all context files
        context_files = self.data_manager.get_all_context_files()
        context_file_summaries = [
            {
                "filename": cf["metadata"]["filename"],
                "tags": cf["metadata"].get("tags", []),
                "date": cf["metadata"].get("date", "")
            }
            for cf in context_files
        ]
        
        # Get behavioral rules
        behavioral_rules = self.data_manager.get_behavioral_rules()
        behavioral_summary = []
        for behavior in behavioral_rules.get("behaviors", []):
            behavioral_summary.append({
                "description": behavior.get("description", ""),
                "weight": behavior.get("weight", 1),
                "category": behavior.get("category", "general")
            })
        
        # Get training history (from single metadata file, no versioning)
        model_dir = MODELS_DIR / self.model_name
        training_count = metadata.get("training_count", 0)
        last_training_date = metadata.get("last_training_date", "Never")
        dataset_size = 0
        
        # Get latest training metadata if available
        metadata_path = model_dir / "fine_tune_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                ft_metadata = json.load(f)
            dataset_size = ft_metadata.get("dataset_size", 0)
            if not last_training_date or last_training_date == "Never":
                last_training_date = ft_metadata.get("date", "Never")
        
        # Get learned pairs count
        learned_pairs = self.data_manager.get_learned_pairs()
        
        return {
            "model_name": self.model_name,
            "base_model": metadata.get("base_model", "unknown"),
            "training_count": training_count,
            "context_files": {
                "count": len(context_files),
                "items": context_file_summaries
            },
            "behavioral_rules": {
                "count": len(behavioral_summary),
                "summary": behavioral_summary
            },
            "training_history": {
                "training_count": training_count,
                "last_training_date": last_training_date,
                "latest_dataset_size": dataset_size,
                "learned_pairs_count": len(learned_pairs)
            }
        }



