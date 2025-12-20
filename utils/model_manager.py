"""Model management utilities"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from utils.config import MODELS_DIR, TRAINING_DIR, ensure_model_directories, get_model_behavioral_path, get_model_training_dir
from tinydb import TinyDB, Query

class ModelManager:
    """Manages model profiles and metadata"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        # Store db in models directory instead of training
        self.db_path = MODELS_DIR / "models_db.json"
        if self.db_path.exists():
            self.db = TinyDB(str(self.db_path))
        else:
            self.db = TinyDB(str(self.db_path))
    
    def create_model_profile(self, model_name: str, base_model: str):
        """Create a new model profile"""
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training directories for this model
        ensure_model_directories(model_name)
        
        # Save model metadata (no versioning)
        metadata = {
            "name": model_name,
            "base_model": base_model,
            "training_count": 0,
            "created_date": datetime.now().isoformat(),
            "archived": False
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Initialize behavioral.json in model's training directory
        try:
            behavioral_path = get_model_behavioral_path(model_name)
            # Ensure parent directory exists
            behavioral_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if there's an old behavioral.json file to migrate (only if old directory exists)
            if TRAINING_DIR.exists():
                old_behavioral_path = TRAINING_DIR / f"{model_name}_behavioral.json"
                if old_behavioral_path.exists() and not behavioral_path.exists():
                    # Migrate old file to new location
                    try:
                        import shutil
                        shutil.copy2(old_behavioral_path, behavioral_path)
                    except Exception as e:
                        print(f"Warning: Could not migrate old behavioral.json: {e}")
            
            if not behavioral_path.exists():
                with open(behavioral_path, 'w') as f:
                    json.dump({"behaviors": []}, f, indent=2)
        except Exception as e:
            # If behavioral.json creation fails, log but don't fail model creation
            print(f"Warning: Could not create behavioral.json for {model_name}: {e}")
        
        # Initialize behavior_packs.json in model's profile directory
        try:
            behavior_packs_path = model_dir / "behavior_packs.json"
            if not behavior_packs_path.exists():
                # Create with blank stems (empty exemplars)
                blank_behavior_packs = {
                    "behavior_version": "1.0",
                    "default_mode": "coaching",
                    "exemplars": {}
                }
                with open(behavior_packs_path, 'w') as f:
                    json.dump(blank_behavior_packs, f, indent=2)
                print(f"Created blank behavior_packs.json for {model_name}")
        except Exception as e:
            # If behavior_packs.json creation fails, log but don't fail model creation
            print(f"Warning: Could not create behavior_packs.json for {model_name}: {e}")
        
        return model_name
    
    def get_available_models(self, include_archived: bool = False):
        """Get list of available model profiles"""
        if not self.models_dir.exists():
            return []
        
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    metadata = self.get_model_metadata(model_dir.name)
                    if metadata:
                        is_archived = metadata.get("archived", False)
                        if include_archived or not is_archived:
                            models.append(model_dir.name)
        
        return models
    
    def get_archived_models(self):
        """Get list of archived model profiles"""
        if not self.models_dir.exists():
            return []
        
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    metadata = self.get_model_metadata(model_dir.name)
                    if metadata and metadata.get("archived", False):
                        models.append(model_dir.name)
        
        return models
    
    def get_model_metadata(self, model_name: str):
        """Get metadata for a specific model"""
        model_dir = self.models_dir / model_name
        metadata_path = model_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_fine_tuned_model_path(self, model_name: str) -> Path:
        """
        Get the path to the fine-tuned model (always latest, no versioning)
        Supports both LoRA adapters and full models
        
        Args:
            model_name: Name of the model profile
        
        Returns:
            Path to the fine-tuned model directory (contains either adapter or full model), or None if not found
        """
        model_dir = self.models_dir / model_name
        model_path = model_dir / "model"
        
        # Check if we have a tokenizer (indicates model exists)
        # For LoRA adapters, tokenizer is in model/ directory
        # For full models, everything is in model/ directory
        if model_path.exists() and (model_path / "tokenizer_config.json").exists():
            return model_path
        
        return None
    
    def has_fine_tuned_model(self, model_name: str) -> bool:
        """Check if a fine-tuned model exists for this profile"""
        return self.get_fine_tuned_model_path(model_name) is not None
    
    def get_next_version_number(self, model_name: str) -> int:
        """
        Get the next version number for a model
        
        Args:
            model_name: Name of the model profile
        
        Returns:
            Next version number (1, 2, 3, etc.)
        """
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return 1
        
        # Find all existing version folders
        existing_versions = []
        for item in model_dir.iterdir():
            if item.is_dir() and item.name.startswith("V") and item.name[1:].isdigit():
                try:
                    version_num = int(item.name[1:])
                    existing_versions.append(version_num)
                except:
                    pass
        
        if not existing_versions:
            return 1
        
        return max(existing_versions) + 1
    
    def get_previous_version_dir(self, model_name: str) -> Optional[Path]:
        """
        Get the previous version directory (for V2+, returns V1, V2, etc.)
        
        Args:
            model_name: Name of the model profile
        
        Returns:
            Path to previous version directory, or None if no previous version
        """
        next_version = self.get_next_version_number(model_name)
        if next_version <= 1:
            return None
        
        previous_version = next_version - 1
        model_dir = self.models_dir / model_name
        previous_version_dir = model_dir / f"V{previous_version}"
        
        if previous_version_dir.exists():
            return previous_version_dir
        
        return None
    
    def create_version_folder(self, model_name: str) -> Path:
        """
        Create a new version folder (V1, V2, etc.)
        
        Args:
            model_name: Name of the model profile
        
        Returns:
            Path to the new version folder
        """
        version_num = self.get_next_version_number(model_name)
        model_dir = self.models_dir / model_name
        version_dir = model_dir / f"V{version_num}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training subdirectory
        training_dir = version_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        return version_dir
    
    def list_available_versions(self, model_name: str) -> list:
        """
        List all available versions for a model (V1, V2, etc.)
        
        Args:
            model_name: Name of the model profile
        
        Returns:
            List of version numbers (integers) sorted ascending
        """
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return []
        
        versions = []
        for item in model_dir.iterdir():
            if item.is_dir() and item.name.startswith("V") and item.name[1:].isdigit():
                try:
                    version_num = int(item.name[1:])
                    # Check if this version has weights
                    weights_dir = item / "weights"
                    if weights_dir.exists():
                        # Check for adapter directory or adapter files
                        adapter_dir = weights_dir / "adapter"
                        has_adapter = (adapter_dir.exists() and any(adapter_dir.iterdir())) or \
                                     any(weights_dir.glob("adapter_model.bin"))
                        if has_adapter:
                            versions.append(version_num)
                except:
                    pass
        
        return sorted(versions)
    
    def get_most_recent_version(self, model_name: str) -> Optional[int]:
        """
        Get the most recent version number for a model
        
        Args:
            model_name: Name of the model profile
        
        Returns:
            Most recent version number, or None if no versions exist
        """
        versions = self.list_available_versions(model_name)
        return max(versions) if versions else None
    
    def get_version_weights_path(self, model_name: str, version: int) -> Optional[Path]:
        """
        Get the path to weights for a specific version
        
        Args:
            model_name: Name of the model profile
            version: Version number (1, 2, etc.)
        
        Returns:
            Path to adapter directory if found, or None
        """
        model_dir = self.models_dir / model_name
        version_dir = model_dir / f"V{version}"
        
        if not version_dir.exists():
            return None
        
        weights_dir = version_dir / "weights"
        if not weights_dir.exists():
            return None
        
        # Check for adapter directory first
        adapter_dir = weights_dir / "adapter"
        if adapter_dir.exists() and any(adapter_dir.iterdir()):
            return adapter_dir
        
        # Check for adapter files directly in weights dir
        if any(weights_dir.glob("adapter_model.bin")):
            return weights_dir
        
        return None
    
    def get_version_metadata(self, model_name: str, version: int) -> Optional[dict]:
        """
        Get metadata for a specific version (from version_metadata.json)
        
        Args:
            model_name: Name of the model profile
            version: Version number
        
        Returns:
            Dictionary with version metadata, or None
        """
        model_dir = self.models_dir / model_name
        version_dir = model_dir / f"V{version}"
        
        if not version_dir.exists():
            return None
        
        # Check for version_metadata.json in the version directory (Axolotl format)
        version_metadata_path = version_dir / "version_metadata.json"
        if version_metadata_path.exists():
            try:
                with open(version_metadata_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Also check vast_jobs.json for training info (fallback)
        training_dir = get_model_training_dir(model_name)
        jobs_file = training_dir / "vast_jobs.json"
        if jobs_file.exists():
            try:
                with open(jobs_file, 'r') as f:
                    jobs = json.load(f)
                    # Find job for this version
                    for job in jobs:
                        if job.get("version") == f"V{version}" or job.get("version") == version:
                            return job
            except:
                pass
        
        return None
    
    def archive_model(self, model_name: str):
        """Archive a model profile"""
        metadata = self.get_model_metadata(model_name)
        if metadata:
            metadata["archived"] = True
            metadata["archived_date"] = datetime.now().isoformat()
            
            model_dir = self.models_dir / model_name
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        return False
    
    def unarchive_model(self, model_name: str):
        """Unarchive a model profile"""
        metadata = self.get_model_metadata(model_name)
        if metadata:
            metadata["archived"] = False
            if "archived_date" in metadata:
                del metadata["archived_date"]
            
            model_dir = self.models_dir / model_name
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        return False
    
    def delete_model(self, model_name: str):
        """Delete a model profile and all its files (only if archived)"""
        import shutil
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            return {"success": False, "error": "Model profile not found"}
        
        # Check if model is archived
        metadata = self.get_model_metadata(model_name)
        if not metadata:
            return {"success": False, "error": "Could not read model metadata"}
        
        if not metadata.get("archived", False):
            return {"success": False, "error": "Model must be archived before deletion. Please archive the model first."}
        
        try:
            # Delete the entire model directory (includes all training data, versions, etc.)
            # This deletes everything: models/{model_name}/ and all subdirectories
            shutil.rmtree(model_dir)
            
            message = f"Model profile '{model_name}' and all associated training data deleted successfully"
            return {"success": True, "message": message}
        except Exception as e:
            return {"success": False, "error": str(e)}

