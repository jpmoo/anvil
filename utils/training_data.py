"""Training data management"""

import json
from pathlib import Path
from datetime import datetime
from utils.config import (
    get_model_data_dir, get_model_logs_dir, 
    get_model_context_dir, get_model_training_dir,
    get_model_behavioral_path
)

class TrainingDataManager:
    """Manages training data collection and storage"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.data_dir = get_model_data_dir(model_name)
        self.logs_dir = get_model_logs_dir(model_name)
        self.context_dir = get_model_context_dir(model_name)
        self.training_dir = get_model_training_dir(model_name)
    
    def save_learned_pair(self, question: str, answer: str):
        """Save a question/answer pair to learned_pairs.json"""
        learned_pairs_path = self.data_dir / "learned_pairs.json"
        
        # Load existing pairs
        if learned_pairs_path.exists():
            with open(learned_pairs_path, 'r') as f:
                data = json.load(f)
        else:
            data = {"pairs": []}
        
        # Add new pair
        pair = {
            "question": question,
            "answer": answer,
            "date": datetime.now().isoformat()
        }
        data["pairs"].append(pair)
        
        # Save
        with open(learned_pairs_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return pair
    
    def log_conversation_turn(self, user_message: str, assistant_message: str):
        """Log a conversation turn to daily log file"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = self.logs_dir / f"{today}.json"
        
        # Load existing log
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {"date": today, "turns": []}
        
        # Add new turn
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": assistant_message,
            "model": self.model_name
        }
        log_data["turns"].append(turn)
        
        # Save
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def get_all_context_files(self):
        """Get all context files from context directory (all files are used in training)"""
        context_files = []
        
        for metadata_file in self.context_dir.glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if metadata.get("model") == self.model_name:
                    text_file = self.context_dir / metadata.get("text_file", "")
                    if text_file.exists():
                        with open(text_file, 'r', encoding='utf-8') as tf:
                            context_files.append({
                                "text": tf.read(),
                                "metadata": metadata
                            })
        
        return context_files
    
    def get_behavioral_rules(self):
        """Get behavioral rules for the model"""
        behavioral_path = get_model_behavioral_path(self.model_name)
        
        if behavioral_path.exists():
            with open(behavioral_path, 'r') as f:
                return json.load(f)
        else:
            return {"behaviors": []}
    
    def get_all_training_data(self):
        """Get all training data for fine-tuning"""
        return {
            "context_files": self.get_all_context_files(),
            "behavioral_rules": self.get_behavioral_rules(),
            "learned_pairs": self.get_learned_pairs()
        }
    
    def get_learned_pairs(self):
        """Get all learned pairs for this model"""
        learned_pairs_path = self.data_dir / "learned_pairs.json"
        
        if learned_pairs_path.exists():
            with open(learned_pairs_path, 'r') as f:
                data = json.load(f)
                return data.get("pairs", [])
        return []


