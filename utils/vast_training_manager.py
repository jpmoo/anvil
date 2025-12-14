"""Manage training jobs on Vast.ai with Axolotl"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import shutil
from utils.vast_ai_client import VastAIClient
from utils.axolotl_prep import AxolotlDataPrep
from utils.model_manager import ModelManager
from utils.config import get_model_training_dir, get_model_queue_dir, MODELS_DIR


class VastAILogger:
    """Logger for Vast.ai operations with detailed output"""
    
    def __init__(self):
        self.logs: List[Dict] = []
    
    def log(self, level: str, message: str, details: Optional[Dict] = None):
        """Log a message with level and optional details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "details": details or {}
        }
        self.logs.append(log_entry)
        # Also print for console debugging
        print(f"[{level}] {message}")
        if details:
            print(f"  Details: {json.dumps(details, indent=2, default=str)[:500]}")
    
    def get_logs(self) -> List[Dict]:
        """Get all logs"""
        return self.logs
    
    def get_recent_logs(self, count: int = 20) -> List[Dict]:
        """Get recent logs"""
        return self.logs[-count:] if len(self.logs) > count else self.logs
    
    def clear(self):
        """Clear all logs"""
        self.logs = []


class VastTrainingManager:
    """Manage LoRA training jobs on Vast.ai using Axolotl"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize training manager
        
        Args:
            model_name: Name of the model profile
            api_key: Vast.ai API key (optional, will use env var if not provided)
        """
        self.model_name = model_name
        self.vast_client = VastAIClient(api_key)
        self.axolotl_prep = AxolotlDataPrep(model_name)
        self.model_manager = ModelManager()
        self.logger = VastAILogger()  # Add logger
        # Use explicit Path import to avoid scoping issues
        from pathlib import Path as PathClass
        self.training_dir = get_model_training_dir(model_name)
        # Ensure jobs_file is a proper Path object
        if isinstance(self.training_dir, PathClass):
            self.jobs_file = self.training_dir / "vast_jobs.json"
        else:
            # Fallback if training_dir is not a Path (shouldn't happen, but be safe)
            self.jobs_file = PathClass(self.training_dir) / "vast_jobs.json"
    
    def prepare_training_package(self, epochs: int = 10, learning_rate: float = 2e-4, hf_model_override: Optional[str] = None, yaml_config_path: Optional[str] = None, file_group: Optional[List[Dict]] = None) -> Dict:
        """
        Prepare all training data and configuration files (without moving from queue)
        
        Returns:
            Dictionary with paths and statistics
        """
        # Use a temporary directory for preparing the dataset
        # Files stay in queue until training is confirmed successful
        from tempfile import mkdtemp
        temp_dir = Path(mkdtemp())
        
        # Prepare dataset from queue files (without moving them)
        # If file_group is provided, only process those specific files
        dataset_path = temp_dir / "training_data.jsonl"
        dataset_stats = self.axolotl_prep.prepare_training_dataset(dataset_path, file_group=file_group)
        
        # Get model metadata
        metadata = self.model_manager.get_model_metadata(self.model_name)
        base_model = metadata.get("base_model", "llama2") if metadata else "llama2"
        
        # Map to Hugging Face model name (or use override)
        if hf_model_override:
            hf_model = hf_model_override
        else:
            hf_model = self.axolotl_prep.get_hf_model_name(base_model)
        
        if not hf_model:
            # Show available mappings to help user
            from utils.axolotl_prep import MODEL_MAPPING
            available_models = ", ".join(sorted(MODEL_MAPPING.keys())[:10])  # Show first 10
            raise ValueError(
                f"Could not map Ollama model '{base_model}' to a Hugging Face model identifier.\n\n"
                f"Available mappings include: {available_models}...\n\n"
                f"Please either:\n"
                f"1. Use a supported model name (see list above)\n"
                f"2. Or manually specify the Hugging Face model identifier in the code"
            )
        
        # Check if we have a previous version to merge weights from
        previous_version_dir = self.model_manager.get_previous_version_dir(self.model_name)
        previous_adapter_path = None
        
        if previous_version_dir:
            # Check if previous version has weights
            previous_weights_dir = previous_version_dir / "weights"
            if previous_weights_dir.exists():
                # Look for adapter directory or adapter files
                adapter_dir = previous_weights_dir / "adapter"
                if adapter_dir.exists() and any(adapter_dir.iterdir()):
                    previous_adapter_path = str(adapter_dir)
                elif any(previous_weights_dir.glob("adapter_model.bin")):
                    # Adapter files directly in weights dir
                    previous_adapter_path = str(previous_weights_dir)
        
        # Handle YAML config
        output_dir = "/workspace/output/training"
        config_path = temp_dir / "axolotl_config.yaml"
        
        if yaml_config_path and Path(yaml_config_path).exists():
            # Use provided YAML config file
            import shutil
            import yaml
            shutil.copy2(yaml_config_path, config_path)
            # Read the YAML to get config dict for package_info
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Fix adapter issue: Always remove adapter: "lora" as it causes Axolotl to treat it as a path
            # Axolotl will automatically infer LoRA mode from lora_* parameters (lora_r, lora_alpha, etc.)
            if config.get("adapter") == "lora" and not Path(str(config.get("adapter", ""))).exists():
                del config["adapter"]
                self.logger.log("INFO", "Removed 'adapter: lora' from YAML config (LoRA will be inferred from lora_* parameters)")
            
            # Auto-adjust config for small datasets to prevent empty batch errors
            total_examples = dataset_stats.get("total_examples", 0)
            if total_examples > 0:
                # Auto-calculate epochs based on dataset size (same logic as default config)
                batch_size = config.get("micro_batch_size", 4)
                gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4)
                effective_batch_size = batch_size * gradient_accumulation_steps
                
                # Calculate steps per epoch
                sample_packing = config.get("sample_packing", True)
                if sample_packing:
                    sample_packing_efficiency = 0.7  # Conservative estimate
                    effective_dataset_size = max(1, int(total_examples * sample_packing_efficiency))
                else:
                    effective_dataset_size = total_examples
                
                steps_per_epoch = max(1, effective_dataset_size // effective_batch_size)
                
                # Target steps based on dataset size
                if total_examples < 200:
                    target_steps = 150
                elif total_examples < 1000:
                    target_steps = 300
                else:
                    target_steps = 400
                
                # Calculate epochs needed to reach target steps
                calculated_epochs = max(1, int(target_steps / steps_per_epoch))
                calculated_epochs = min(max(calculated_epochs, 1), 50)  # Cap at 50
                
                # Update epochs in config to use calculated value
                current_epochs = config.get("num_epochs", epochs)
                final_epochs = max(calculated_epochs, current_epochs)  # Use higher of calculated or existing
                config["num_epochs"] = final_epochs
                final_steps = steps_per_epoch * final_epochs
                self.logger.log("INFO", f"Auto-calculated epochs for YAML config: {final_epochs} (targeting ~{target_steps} steps for {total_examples} examples, ~{steps_per_epoch} steps/epoch, will result in ~{final_steps} total steps)")
                
                # Calculate minimum eval examples needed (at least 2 for batch creation)
                min_eval_examples = 2
                val_set_size = config.get("val_set_size", 0.1)
                
                # If validation set would be too small, adjust it
                if total_examples * val_set_size < min_eval_examples:
                    if total_examples < 50:
                        # Very small dataset: disable validation entirely
                        config["val_set_size"] = 0.0
                        self.logger.log("INFO", f"Dataset has only {total_examples} examples. Disabling validation set to prevent empty batch errors.")
                    elif total_examples < 200:
                        # Small dataset: reduce validation set or disable sample packing
                        # Disable sample packing for small datasets as it can cause empty batch issues
                        # CRITICAL: Always disable sample_packing regardless of current value
                        config["sample_packing"] = False
                        if config.get("sample_packing") != False:  # Log only if it was previously enabled
                            self.logger.log("INFO", f"Dataset has {total_examples} examples. Disabled sample_packing for stability with small datasets.")
                        # Ensure val_set_size results in at least min_eval_examples
                        min_val_size = min_eval_examples / total_examples
                        if val_set_size < min_val_size:
                            config["val_set_size"] = min_val_size
                            self.logger.log("INFO", f"Adjusted val_set_size to {min_val_size:.3f} to ensure at least {min_eval_examples} eval examples.")
                    else:
                        # Medium dataset: just ensure val_set_size is reasonable
                        min_val_size = min_eval_examples / total_examples
                        if val_set_size < min_val_size:
                            config["val_set_size"] = min_val_size
                            self.logger.log("INFO", f"Adjusted val_set_size to {min_val_size:.3f} to ensure at least {min_eval_examples} eval examples.")
                    
                    # For all datasets, ALWAYS disable sample_packing to prevent multipack sampler errors
                    # sample_packing can cause IndexError with certain batch configurations
                    # CRITICAL: Always set to False explicitly, even if not in config (Axolotl may default to True)
                    config["sample_packing"] = False
                    if config.get("sample_packing") != False:  # Log only if it was previously enabled
                        self.logger.log("INFO", f"Disabled sample_packing to prevent multipack sampler errors. Enable manually if needed for large datasets.")
            
            # YAML configs don't include datasets or paths - we need to set them
            # Update paths and model settings in config to match remote paths and correct model
            # Update paths and model settings in config to match remote paths and correct model
            # Always set datasets (YAML won't include this)
            config["datasets"] = [{"path": "/workspace/data/training_data.jsonl", "type": "alpaca"}]
            # Always set output directory (YAML won't include paths)
            config["output_dir"] = output_dir
            # Always set base_model and base_model_config to use the correct HF model
            # This ensures the YAML uses the correct model even if it was set for a different one
            config["base_model"] = hf_model
            config["base_model_config"] = hf_model
            # Ensure dataset_preparation_path is set (Axolotl needs this)
            if "dataset_preparation_path" not in config:
                config["dataset_preparation_path"] = "/workspace/axolotl/prepared_data"
            
            # Maximize sample retention: set train_on_inputs to True if not explicitly set
            # This prevents dropping samples where input has no trainable tokens
            if "train_on_inputs" not in config:
                config["train_on_inputs"] = True
                self.logger.log("INFO", "Set train_on_inputs=True to maximize sample retention")
            elif config.get("train_on_inputs") == False:
                # Override if it's explicitly False - user wants to maximize retention
                config["train_on_inputs"] = True
                self.logger.log("INFO", "Overrode train_on_inputs=False to True to maximize sample retention")
            
            # CRITICAL: Always ensure sample_packing is explicitly False before writing config
            # This prevents multipack sampler IndexError issues
            config["sample_packing"] = False
            
            # Write updated config back
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            # Calculate appropriate number of epochs based on dataset size to target reasonable training steps
            # Target: 200-500 steps for good training (adjust based on dataset size)
            total_examples = dataset_stats.get("total_examples", 0)
            batch_size = 4  # Default micro_batch_size
            gradient_accumulation_steps = 4  # Default
            effective_batch_size = batch_size * gradient_accumulation_steps
            
            if total_examples > 0:
                # Calculate steps per epoch
                # With sample packing, Axolotl packs multiple samples per sequence, reducing steps
                # Estimate: sample packing efficiency is typically 0.6-0.9, so we'll use 0.7 as average
                # This means ~70% of samples are packed, effectively reducing dataset size
                sample_packing_efficiency = 0.7  # Conservative estimate
                effective_dataset_size = max(1, int(total_examples * sample_packing_efficiency))
                steps_per_epoch = max(1, effective_dataset_size // effective_batch_size)
                
                # Target steps based on dataset size:
                # - Small datasets (< 200): target 100-200 steps
                # - Medium datasets (200-1000): target 200-400 steps  
                # - Large datasets (> 1000): target 300-500 steps
                if total_examples < 200:
                    target_steps = 150
                elif total_examples < 1000:
                    target_steps = 300
                else:
                    target_steps = 400
                
                # Calculate epochs needed to reach target steps
                calculated_epochs = max(1, int(target_steps / steps_per_epoch))
                # Cap at reasonable maximum (50 epochs) and minimum (1 epoch)
                calculated_epochs = min(max(calculated_epochs, 1), 50)
                
                # Always use auto-calculated epochs for optimal training
                # User can still override via epochs parameter if needed, but we'll use calculated by default
                # Only use user-provided epochs if it's higher than calculated (to ensure enough steps)
                if epochs >= calculated_epochs:
                    # User provided higher value - use it
                    final_epochs = epochs
                    final_steps = steps_per_epoch * final_epochs
                    self.logger.log("INFO", f"Using {final_epochs} epochs (~{final_steps} steps) - user override")
                else:
                    # Use calculated epochs (better for training)
                    final_epochs = calculated_epochs
                    final_steps = steps_per_epoch * final_epochs
                    if epochs != 10:  # User provided a value but it was lower
                        self.logger.log("INFO", f"Auto-calculated {final_epochs} epochs (~{final_steps} steps) instead of user-specified {epochs} epochs (~{steps_per_epoch * epochs} steps) to ensure adequate training")
                    else:
                        self.logger.log("INFO", f"Auto-calculated epochs: {final_epochs} (targeting ~{target_steps} steps for {total_examples} examples, ~{steps_per_epoch} steps/epoch, will result in ~{final_steps} total steps)")
                
                epochs = final_epochs
            
            # Create stock/default Axolotl config (no YAML file provided)
            # This creates a standard Axolotl configuration with default settings
            config = self.axolotl_prep.create_axolotl_config(
                base_model=hf_model,
                dataset_path="/workspace/data/training_data.jsonl",  # Path on Vast.ai instance
                output_dir=output_dir,
                output_path=config_path,
                num_epochs=epochs,
                learning_rate=learning_rate,
                train_on_inputs=True,  # Maximize sample retention - set to False if you want to focus only on responses
                previous_adapter_path=previous_adapter_path  # Path to previous adapter for incremental training
            )
            
            # Auto-adjust config for small datasets to prevent empty batch errors
            total_examples = dataset_stats.get("total_examples", 0)
            if total_examples > 0:
                # Calculate minimum eval examples needed (at least 2 for batch creation)
                min_eval_examples = 2
                val_set_size = config.get("val_set_size", 0.1)
                
                # If validation set would be too small, adjust it
                if total_examples * val_set_size < min_eval_examples:
                    if total_examples < 50:
                        # Very small dataset: disable validation entirely
                        config["val_set_size"] = 0.0
                        self.logger.log("INFO", f"Dataset has only {total_examples} examples. Disabling validation set to prevent empty batch errors.")
                    elif total_examples < 200:
                        # Small dataset: reduce validation set or disable sample packing
                        # Disable sample packing for small datasets as it can cause empty batch issues
                        # CRITICAL: Always disable sample_packing regardless of current value
                        config["sample_packing"] = False
                        if config.get("sample_packing") != False:  # Log only if it was previously enabled
                            self.logger.log("INFO", f"Dataset has {total_examples} examples. Disabled sample_packing for stability with small datasets.")
                        # Ensure val_set_size results in at least min_eval_examples
                        min_val_size = min_eval_examples / total_examples
                        if val_set_size < min_val_size:
                            config["val_set_size"] = min_val_size
                            self.logger.log("INFO", f"Adjusted val_set_size to {min_val_size:.3f} to ensure at least {min_eval_examples} eval examples.")
                    else:
                        # Medium dataset: just ensure val_set_size is reasonable
                        min_val_size = min_eval_examples / total_examples
                        if val_set_size < min_val_size:
                            config["val_set_size"] = min_val_size
                            self.logger.log("INFO", f"Adjusted val_set_size to {min_val_size:.3f} to ensure at least {min_eval_examples} eval examples.")
            
            # CRITICAL: Always ensure sample_packing is explicitly False to prevent multipack sampler errors
            # This must be done before writing the config file
            config["sample_packing"] = False
            
            # Write updated config back to file
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return {
            "temp_dir": str(temp_dir),
            "dataset_path": str(dataset_path),
            "config_path": str(config_path),
            "hf_model": hf_model,
            "base_model": base_model,
            "dataset_stats": dataset_stats,
            "config": config,
            "previous_adapter_path": previous_adapter_path  # Local path to previous adapter
        }
    
    # Note: create_training_script method removed - we no longer use onstart scripts
    # All training is started manually via SSH on existing instances
    
    def launch_training_job(self,
                           gpu_name: Optional[str] = None,
                           min_gpu_ram: int = 16,
                           max_price: Optional[float] = None,
                           disk_space: int = 100,
                           epochs: int = 10,
                           learning_rate: float = 2e-4,
                           hf_model_override: Optional[str] = None,
                           num_gpus: Optional[int] = None,
                           yaml_config_path: Optional[str] = None,
                           file_group: Optional[List[Dict]] = None,
                           job_queue: Optional[List[Dict]] = None,
                           hf_token: Optional[str] = None,
                           existing_instance_id: Optional[str] = None,
                           ssh_port_override: Optional[int] = None) -> Dict:
        """
        Launch a training job on Vast.ai
        
        Args:
            gpu_name: Preferred GPU name (e.g., "RTX 3090", "A100")
            min_gpu_ram: Minimum GPU RAM in GB
            max_price: Maximum price per hour in USD
            disk_space: Disk space needed in GB
            job_queue: List of job configurations to process sequentially (overrides yaml_config_path/file_group)
        
        Returns:
            Job information dictionary
        """
        # If job_queue is provided, use it; otherwise use single job parameters
        if job_queue and len(job_queue) > 0:
            # Process first job in queue
            current_job = job_queue[0]
            yaml_config_path = current_job.get("yaml_path")
            file_group = current_job.get("file_group")
            current_job_index = 0
        else:
            # Single job mode (backward compatible)
            current_job_index = None
        
        # Prepare training package (files stay in queue until training succeeds)
        # Note: prepare_training_package will auto-calculate optimal epochs based on dataset size
        package_info = self.prepare_training_package(
            epochs=epochs, 
            learning_rate=learning_rate, 
            hf_model_override=hf_model_override,
            yaml_config_path=yaml_config_path,
            file_group=file_group
        )
        
        # Get the actual epochs used (may have been auto-calculated)
        # Check both the config dict and the YAML file if it exists
        actual_epochs = epochs
        if package_info.get("config"):
            actual_epochs = package_info["config"].get("num_epochs", epochs)
        elif package_info.get("config_path"):
            # If config is a YAML file, read it
            import yaml
            config_path = Path(package_info["config_path"])
            if config_path.exists():
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}
                    actual_epochs = yaml_config.get("num_epochs", epochs)
        
        if actual_epochs != epochs:
            epochs = actual_epochs  # Use the calculated value
            self.logger.log("INFO", f"Using auto-calculated {epochs} epochs for training")
        
        # We only support existing instances - instance_id must be provided
        if not existing_instance_id:
            raise Exception("Creating new instances is no longer supported. Please use an existing instance.")
        
        instance_id = existing_instance_id
        # Get instance info to extract offer details
        try:
            instance_status = self.vast_client.get_instance_status(instance_id)
            # Extract offer_id from instance if available (for job tracking)
            offer_id = instance_status.get("machine_id") or instance_status.get("offer_id") or "existing"
            # Create a mock selected_offer dict from instance status for job_info
            selected_offer = {
                "gpu_ram": instance_status.get("gpu_ram", 0),
                "dph_total": instance_status.get("dph_total", 0),
                "disk_space": instance_status.get("disk_space", disk_space),
                "geolocation": instance_status.get("geolocation"),
                "location": instance_status.get("location"),
                "country": instance_status.get("country"),
                "gpu_name": instance_status.get("gpu_name", "Unknown")
            }
        except:
            offer_id = "existing"
            # Create minimal mock selected_offer if we can't get instance status
            selected_offer = {
                "gpu_ram": 0,
                "dph_total": 0,
                "disk_space": disk_space,
                "gpu_name": "Unknown"
            }
        
        # Wait for instance to be ready, then upload files via SCP
        # Note: This requires SSH keys to be configured
        # Store package_info in job so we can upload files later when instance is ready
        # The UI will need to poll status and trigger file upload when instance becomes "running"
        
        # Extract actual values from selected offer
        # GPU RAM is in MB in the offer, convert to GB
        actual_gpu_ram_mb = selected_offer.get("gpu_ram", 0)
        actual_gpu_ram_gb = actual_gpu_ram_mb / 1024 if actual_gpu_ram_mb > 0 else None
        
        # Get actual price
        actual_price = selected_offer.get("dph_total", 0)
        
        # Get actual disk space (from offer or requested)
        actual_disk_space = selected_offer.get("disk_space", disk_space)
        
        # Get location/country
        location = selected_offer.get("geolocation") or selected_offer.get("location") or selected_offer.get("country") or "Unknown"
        
        # Save job information (no version directory yet - will be created on success)
        # Store YAML config info in package_info for display
        if yaml_config_path:
            yaml_filename = Path(yaml_config_path).name
            package_info["yaml_config"] = yaml_filename
        else:
            package_info["yaml_config"] = None
        
        job_info = {
            "model_name": self.model_name,
            "instance_id": instance_id,
            "offer_id": offer_id,
            "gpu_info": selected_offer.get("gpu_name", "Unknown"),
            "gpu_name": gpu_name,  # Store requested GPU name
            "min_gpu_ram": min_gpu_ram,  # Store minimum GPU RAM requirement (for filtering)
            "actual_gpu_ram": actual_gpu_ram_gb,  # Store actual GPU RAM in GB
            "max_price": max_price,  # Store maximum price (for filtering)
            "disk_space": disk_space,  # Store requested disk space
            "actual_disk_space": actual_disk_space,  # Store actual disk space
            "num_gpus": num_gpus,  # Store number of GPUs
            "price_per_hour": actual_price,  # Store actual price per hour
            "location": location,  # Store server location
            "status": "launching",
            "created_at": datetime.now().isoformat(),
            "package_info": package_info,
            "version_dir": None,  # Will be set when training succeeds
            "version": None,  # Will be set when training succeeds
            "job_queue": job_queue if job_queue else None,  # Store job queue for sequential processing
            "current_job_index": current_job_index if job_queue else None,  # Track which job in queue is active
            "epochs": epochs,  # Store for queue processing
            "learning_rate": learning_rate,  # Store for queue processing
            "hf_model_override": hf_model_override,  # Store for queue processing
            "ssh_port_override": ssh_port_override,  # Store SSH port override if provided
        }
        
        # If SSH port override is provided, save it to the job
        if ssh_port_override:
            job_info["ssh_port"] = ssh_port_override
        
        self._save_job(job_info)
        
        return job_info
    
    def get_job_status(self, instance_id: Optional[str] = None) -> Dict:
        """
        Get status of training job
        
        Args:
            instance_id: Instance ID (if None, gets latest job for this model)
        
        Returns:
            Job status information
        """
        jobs = self._load_jobs()
        
        if instance_id:
            job = next((j for j in jobs if j.get("instance_id") == instance_id), None)
        else:
            # Get latest job for this model
            model_jobs = [j for j in jobs if j.get("model_name") == self.model_name]
            if not model_jobs:
                return {"error": "No jobs found for this model"}
            job = sorted(model_jobs, key=lambda x: x.get("created_at", ""), reverse=True)[0]
        
        if not job:
            return {"error": "Job not found"}
        
        # Skip status check if job is already finalized
        if job.get("finalized", False):
            self.logger.log("INFO", f"Job {job.get('instance_id')} is already finalized, skipping status check")
            return job
        
        # Preserve "launching" and "validated" statuses - don't overwrite them based on instance status
        # These statuses are set by the UI workflow and should only change when explicitly updated
        current_status = job.get("status", "unknown")
        if current_status in ["launching", "validated"]:
            # Still get instance info for SSH details, but don't update status
            preserve_status = True
        else:
            preserve_status = False
        
        # Get current instance status from Vast.ai
        # Safety check: ensure instance_id exists in job
        job_instance_id = job.get("instance_id")
        if not job_instance_id:
            self.logger.log("WARNING", f"Job missing instance_id, cannot check status")
            job["error"] = "Job missing instance_id"
            return job
        
        try:
            self.logger.log("INFO", f"Checking status for instance {job_instance_id}")
            api_response = self.vast_client.get_instance_status(job_instance_id)
            job["vast_status"] = api_response
            self.logger.log("INFO", f"Received API response for instance {job_instance_id}", {
                "response_keys": list(api_response.keys())[:10]
            })
            
            # Extract instance data - API response may have 'instances' key with nested data
            if "instances" in api_response and isinstance(api_response["instances"], dict):
                instance_status = api_response["instances"]
            elif isinstance(api_response, dict) and "actual_status" in api_response:
                # Already at instance level
                instance_status = api_response
            else:
                # Try to find instance data in response
                instance_status = api_response
            
            # Store old status before updating
            old_status = job.get("status", "unknown")
            
            # Determine status with improved detection, but preserve "launching" and "validated" statuses
            determined_status = self._determine_status(instance_status)
            if not preserve_status:
                job["status"] = determined_status
            # else: keep the existing "launching" or "validated" status
            
            # Update SSH info if available (for connection)
            # Use get_instance_ssh_info for consistent SSH info extraction (prefers direct IP over gateway)
            ssh_info = self.get_instance_ssh_info(job_instance_id)
            ssh_host = ssh_info.get("host")
            api_ssh_port = ssh_info.get("port", 22)
            
            # Preserve SSH port override if it exists (user-specified port takes precedence)
            ssh_port_override = job.get("ssh_port_override")
            if ssh_port_override:
                ssh_port = ssh_port_override
            else:
                ssh_port = api_ssh_port
            
            if ssh_host:
                job["ssh_host"] = ssh_host
            # Always update port (either override or API port)
            job["ssh_port"] = ssh_port
            
            # Update actual instance values from API response if available
            # GPU RAM (convert from MB to GB if present)
            if "gpu_ram" in instance_status:
                gpu_ram_mb = instance_status.get("gpu_ram", 0)
                if gpu_ram_mb > 0:
                    job["actual_gpu_ram"] = gpu_ram_mb / 1024  # Convert MB to GB
            
            # Actual price
            if "dph_total" in instance_status:
                actual_price = instance_status.get("dph_total", 0)
                if actual_price > 0:
                    job["price_per_hour"] = actual_price
            
            # Actual disk space
            if "disk_space" in instance_status:
                actual_disk = instance_status.get("disk_space")
                if actual_disk and actual_disk > 0:
                    job["actual_disk_space"] = actual_disk
            
            # Location/country
            if "geolocation" in instance_status:
                job["location"] = instance_status.get("geolocation")
            elif "location" in instance_status:
                job["location"] = instance_status.get("location")
            elif "country" in instance_status:
                job["location"] = instance_status.get("country")
            
            # Clear any previous errors if status check succeeded
            if "error" in job:
                job["error"] = None
            if "status_note" in job and "API error" in job.get("status_note", ""):
                job["status_note"] = None
            
            # If status is "running", try to upload files if not already uploaded
            # But only if actual_status is actually "running" (not None or "loading")
            vast_actual_status = instance_status.get("actual_status")
            actual_status_str = str(vast_actual_status).lower() if vast_actual_status else None
            
            # Only attempt upload if instance is actually running (not loading/starting/None)
            # Be conservative - only upload when we have explicit confirmation the instance is ready
            if (determined_status == "running" and 
                actual_status_str is not None and
                actual_status_str in ["running", "active", "ready"] and
                actual_status_str not in ["loading", "starting", "launching", "initializing", "booting"] and
                not job.get("files_uploaded", False)):
                # Check if we have package_info to upload
                package_info = job.get("package_info")
                if package_info:
                    try:
                        # Use SSH info from job record if available (avoids API call)
                        ssh_host = job.get("ssh_host")
                        ssh_port = job.get("ssh_port")
                        
                        # Attempt to upload files
                        print(f"[DEBUG] Attempting to upload training files to instance {job_instance_id}")
                        upload_success = self.upload_training_files(
                            job_instance_id, 
                            package_info,
                            ssh_host=ssh_host,
                            ssh_port=ssh_port
                        )
                        if upload_success:
                            job["files_uploaded"] = True
                            job["status_note"] = "Files uploaded successfully, training should start soon"
                            job["upload_error"] = None  # Clear any previous errors
                            print(f"[DEBUG] Files uploaded successfully")
                            # Save immediately after successful upload
                            self._save_job(job)
                        else:
                            job["status_note"] = "File upload attempted but may have failed. Check SSH keys."
                            self._save_job(job)
                    except Exception as upload_error:
                        error_msg = str(upload_error)
                        # Check if it's a rate limit issue
                        if "RATE_LIMIT" in error_msg or "429" in error_msg:
                            job["status_note"] = "Rate limited by Vast.ai API. Please wait a moment before refreshing."
                            job["upload_error"] = error_msg
                        # Check if it's an SSH key issue
                        elif "Permission denied" in error_msg or "publickey" in error_msg.lower() or "SSH keys" in error_msg:
                            job["status_note"] = "SSH keys required for file upload. Please configure SSH keys or upload files manually."
                            job["upload_error"] = error_msg
                        # Check if it's a connection refused issue (instance not ready)
                        elif "Connection refused" in error_msg or "connection refused" in error_msg.lower():
                            job["status_note"] = "Instance is not ready yet. SSH connection refused. Please wait a moment and refresh status."
                            job["upload_error"] = error_msg
                        else:
                            job["status_note"] = f"File upload error: {error_msg[:100]}"
                            job["upload_error"] = error_msg
                        print(f"[DEBUG] File upload failed: {error_msg}")
                        # Save even on error so we don't retry repeatedly
                        self._save_job(job)
            
            # If status is "running", try to check training status (but don't fail if SSH unavailable)
            # But first verify that Vast.ai actually says the instance is running (not loading/launching)
            vast_actual_status = None
            if "actual_status" in instance_status:
                vast_actual_status = str(instance_status.get("actual_status", "")).lower()
            
            # Only check training status if instance is actually running (not loading/launching)
            if determined_status == "running" and vast_actual_status not in ["loading", "starting", "launching", "initializing", "booting"]:
                try:
                    training_status = self.check_training_status(job_instance_id)
                    if not training_status.get("error"):
                        training_status_val = training_status.get("status", "unknown")
                        
                        # If training is completed, update job status
                        if training_status_val == "completed":
                            job["status"] = "completed"
                            job["status_note"] = "Training completed (detected automatically)"
                            job["training_status"] = training_status
                            self._save_job(job)
                            return job  # Return early with updated status
                        elif training_status_val == "training":
                            # Training is actively running - store this info
                            job["training_status"] = training_status
                        else:
                            # Store training status info even if unclear
                            job["training_status"] = training_status
                    else:
                        # SSH check failed - store error but don't fail
                        job["training_status"] = training_status
                except Exception as e:
                    # Don't fail the whole status check if training status check fails
                    # This is expected if SSH keys aren't set up
                    print(f"[DEBUG] Could not check training status (SSH may not be configured): {e}")
            elif determined_status == "running" and vast_actual_status in ["loading", "starting", "launching", "initializing", "booting"]:
                # Instance is still loading/starting - clear any old training status
                if "training_status" in job:
                    job["training_status"] = None
            
            # Always save status (even if unchanged) to ensure we have latest data
            # But especially if it changed
            if old_status != determined_status:
                # Status changed - save immediately
                self._save_job(job)
            else:
                # Status didn't change but we have fresh data - save to update vast_status
                self._save_job(job)
        except Exception as e:
            # Check if instance was destroyed or doesn't exist
            error_msg = str(e)
            # Check for explicit INSTANCE_NOT_FOUND marker or 404
            if "INSTANCE_NOT_FOUND" in error_msg or "404" in error_msg or "not found" in error_msg.lower():
                # Instance was destroyed or doesn't exist
                job["status"] = "destroyed"
                job["status_note"] = "Instance no longer exists (may have been destroyed)"
                job["error"] = None  # Clear any previous errors
                # Save the updated status immediately
                self._save_job(job)
            elif "RATE_LIMIT" in error_msg or "429" in error_msg:
                # Rate limit error - but try to use cached vast_status if available
                if job.get("vast_status"):
                    # We have cached status data, try to determine status from it
                    try:
                        # Extract instance data from cached response
                        cached_response = job["vast_status"]
                        if "instances" in cached_response and isinstance(cached_response["instances"], dict):
                            cached_instance = cached_response["instances"]
                        else:
                            cached_instance = cached_response
                        
                        cached_status = self._determine_status(cached_instance)
                        old_status = job.get("status", "unknown")
                        if cached_status != old_status:
                            job["status"] = cached_status
                            job["status_note"] = "Status updated from cached data (rate limited)"
                            self._save_job(job)
                        else:
                            job["status_note"] = "Rate limited by Vast.ai API. Using cached status."
                    except Exception as e:
                        print(f"[DEBUG] Error using cached status: {e}")
                        job["status_note"] = "Rate limited by Vast.ai API. Please wait a moment before refreshing again."
                else:
                    job["status_note"] = "Rate limited by Vast.ai API. Please wait a moment before refreshing again."
                job["error"] = None  # Don't treat rate limit as an error
            else:
                # API call failed but instance might still be running
                # Check if we have a recent job (within last hour) - might still be starting
                from datetime import datetime, timedelta
                created_at = job.get("created_at")
                if created_at:
                    try:
                        created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        if datetime.now(created_time.tzinfo) - created_time < timedelta(hours=1):
                            job["status"] = "launching"  # Still might be starting
                            job["status_note"] = f"Could not check status (API error: {error_msg[:100]})"
                        else:
                            # Been more than an hour, likely an issue
                            job["status"] = "error"
                            job["error"] = error_msg
                    except:
                        job["status"] = "unknown"
                        job["error"] = error_msg
                else:
                    job["status"] = "unknown"
                    job["error"] = error_msg
        
        return job
    
    def _determine_status(self, instance_status: Dict) -> str:
        """Determine job status from Vast.ai instance status"""
        # Debug: log what we're checking
        import json
        print(f"[DEBUG] _determine_status called with keys: {list(instance_status.keys())}")
        print(f"[DEBUG] Full instance_status: {json.dumps(instance_status, indent=2, default=str)[:500]}")
        
        # Check actual_status first (most reliable field from Vast.ai)
        # Try multiple possible field names
        actual_status = (instance_status.get("actual_status") or 
                       instance_status.get("actualStatus") or
                       instance_status.get("status") or
                       None)
        
        print(f"[DEBUG] actual_status value: {repr(actual_status)}")
        
        # Convert to string and normalize
        if actual_status is not None:
            actual_status_str = str(actual_status).strip()
            if actual_status_str:
                actual_status_lower = actual_status_str.lower()
                
                # If actual_status is "running", trust it immediately - this is the most reliable indicator
                if actual_status_lower == "running":
                    return "running"
                # Map other actual_status values
                if actual_status_lower in ["active", "ready"]:
                    return "running"
                elif actual_status_lower in ["stopped", "stopping"]:
                    return "stopped"
                elif actual_status_lower in ["failed", "error", "crashed"]:
                    return "failed"
                elif actual_status_lower in ["starting", "launching", "initializing", "booting", "loading"]:
                    # "loading" means instance is still starting up - don't treat as running yet
                    # Even if SSH is available, if status is "loading", it's still launching
                    if actual_status_lower == "loading":
                        return "launching"
                    # For other starting states, check SSH availability as secondary indicator
                    ssh_available = (instance_status.get("ssh_host") or 
                                   instance_status.get("connectable"))
                    if ssh_available:
                        return "running"
                    return "launching"
                elif actual_status_lower in ["completed", "finished", "done"]:
                    return "completed"
                elif actual_status_lower in ["destroyed", "deleted", "terminated"]:
                    return "destroyed"
        
        # Fallback to other status fields if actual_status not available or empty
        status = (instance_status.get("status") or 
                 instance_status.get("state") or 
                 instance_status.get("status_msg") or
                 "unknown")
        
        status_lower = str(status).lower().strip()
        
        # Also check if SSH is available (indicates instance is ready)
        ssh_available = (instance_status.get("ssh_host") or 
                         instance_status.get("connectable"))
        
        # If actual_status was None, be more conservative - don't assume running just because SSH is available
        # The instance might still be starting up
        if actual_status is None:
            # If we have a clear status from other fields, use that
            if status_lower in ["running", "active", "ready"]:
                return "running"
            elif status_lower in ["stopped", "stopping"]:
                return "stopped"
            elif status_lower in ["failed", "error", "crashed"]:
                return "failed"
            elif status_lower in ["starting", "launching", "initializing", "booting", "loading"]:
                return "launching"
            elif status_lower in ["completed", "finished", "done"]:
                return "completed"
            elif status_lower in ["destroyed", "deleted", "terminated"]:
                return "destroyed"
            else:
                # actual_status is None and status is unknown - be conservative
                # If SSH is available, might be ready, but don't assume
                if ssh_available:
                    return "launching"  # More conservative - treat as launching if status unclear
                return "launching"
        
        # Map Vast.ai statuses to our internal statuses (when actual_status was not None)
        if status_lower in ["running", "active", "ready"]:
            return "running"
        elif status_lower in ["stopped", "stopping"]:
            return "stopped"
        elif status_lower in ["failed", "error", "crashed"]:
            return "failed"
        elif status_lower in ["starting", "launching", "initializing", "booting", "loading"]:
            # "loading" means instance is still starting up - don't treat as running yet
            # Even if SSH is available, if status is "loading", it's still launching
            if status_lower == "loading":
                return "launching"
            # For other starting states, if SSH is available, it's probably ready
            if ssh_available:
                return "running"
            return "launching"
        elif status_lower in ["completed", "finished", "done"]:
            return "completed"
        elif status_lower in ["destroyed", "deleted", "terminated"]:
            return "destroyed"
        else:
            # If SSH is available, assume it's running even if status is unknown
            if ssh_available:
                return "running"
            # If status is unknown but instance exists, might be starting
            return "launching"
    
    def _save_job(self, job_info: Dict):
        """Save job information to file"""
        jobs = self._load_jobs()
        # Check if this job already exists (by instance_id)
        instance_id = job_info.get("instance_id")
        if instance_id:
            # Update existing job if found
            for i, job in enumerate(jobs):
                if job.get("instance_id") == instance_id:
                    jobs[i] = job_info  # Update existing job
                    break
            else:
                # Job not found, add as new
                jobs.append(job_info)
        else:
            # No instance_id, just append
            jobs.append(job_info)
        
        self.training_dir.mkdir(parents=True, exist_ok=True)
        with open(self.jobs_file, 'w') as f:
            json.dump(jobs, f, indent=2)
    
    def _load_jobs(self) -> list:
        """Load job information from file"""
        try:
            if self.jobs_file:
                # self.jobs_file is already a Path object from __init__
                # Use str() to avoid any Path scoping issues
                jobs_path_str = str(self.jobs_file)
                import os
                if os.path.exists(jobs_path_str):
                    with open(jobs_path_str, 'r') as f:
                        return json.load(f)
        except Exception as e:
            import traceback
            print(f"[DEBUG] Error loading jobs file: {e}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return []
    
    def list_jobs(self) -> list:
        """List all jobs for this model"""
        jobs = self._load_jobs()
        return [j for j in jobs if j.get("model_name") == self.model_name]
    
    def upload_training_files(self, instance_id: str, package_info: Dict, ssh_host: Optional[str] = None, ssh_port: Optional[int] = None) -> bool:
        """
        Upload training files to Vast.ai instance via SSH/SCP
        
        Args:
            instance_id: Vast.ai instance ID
            package_info: Package information with file paths
            ssh_host: SSH host (optional, will fetch from API if not provided)
            ssh_port: SSH port (optional, defaults to 22 if not provided)
        
        Returns:
            True if successful
        """
        try:
            # Use provided SSH info if available, otherwise fetch from API
            if not ssh_host:
                try:
                    # Get instance info to get SSH connection details
                    instance_status = self.vast_client.get_instance_status(instance_id)
                    
                    # Extract instance data if nested (same pattern as in get_job_status)
                    if "instances" in instance_status and isinstance(instance_status["instances"], dict):
                        instance_data = instance_status["instances"]
                    else:
                        instance_data = instance_status
                    
                    # Extract SSH connection info from instance status
                    # Vast.ai API structure may vary - try multiple possible fields
                    ssh_host = (instance_data.get("ssh_host") or 
                               instance_data.get("public_ipaddr") or
                               instance_data.get("ipaddr") or
                               instance_data.get("ssh_ip"))
                    ssh_port = (instance_data.get("ssh_port") or 
                               instance_data.get("port") or 
                               22)
                except Exception as api_error:
                    error_msg = str(api_error)
                    # If we hit a rate limit, re-raise with a clear message
                    if "RATE_LIMIT" in error_msg or "429" in error_msg:
                        raise Exception(f"RATE_LIMIT: Too many requests (429). Please wait a moment before refreshing.")
                    # For other errors, try to continue (maybe SSH info is in job record)
                    pass
            
            # If still no SSH host, try to get from list_instances (but this may also hit rate limits)
            if not ssh_host:
                try:
                    instances = self.vast_client.list_instances()
                    for inst in instances:
                        if str(inst.get("id")) == str(instance_id):
                            ssh_host = inst.get("ssh_host") or inst.get("public_ipaddr")
                            ssh_port = inst.get("ssh_port") or 22
                            break
                except Exception as list_error:
                    error_msg = str(list_error)
                    if "RATE_LIMIT" in error_msg or "429" in error_msg:
                        raise Exception(f"RATE_LIMIT: Too many requests (429). Please wait a moment before refreshing.")
                    pass  # list_instances may fail, that's okay
            
            if not ssh_host:
                raise Exception("Could not get SSH connection details from instance. Instance may not be ready yet or SSH may not be available.")
            
            # Use default port if not provided
            if not ssh_port:
                ssh_port = 22
            
            # Verify files exist before attempting upload
            dataset_path = Path(package_info["dataset_path"])
            config_path = Path(package_info["config_path"])
            
            if not dataset_path.exists():
                raise Exception(f"Training data file not found: {dataset_path}. The temporary directory may have been cleaned up.")
            if not config_path.exists():
                raise Exception(f"Config file not found: {config_path}. The temporary directory may have been cleaned up.")
            
            # Test SSH connection first with a simple command before attempting file upload
            # This helps catch "Connection refused" errors early
            import time
            test_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                "echo 'SSH connection test'"
            ]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
            if test_result.returncode != 0:
                error_output = test_result.stderr or test_result.stdout
                if "Connection refused" in error_output or "Connection timed out" in error_output:
                    raise Exception(f"SSH connection refused or timed out. Instance may not be ready yet. Please wait a moment and refresh status. Error: {error_output[:200]}")
                elif "Permission denied" in error_output or "publickey" in error_output.lower():
                    raise Exception(f"SSH authentication failed. SSH keys are required. Error: {error_output[:200]}")
                else:
                    raise Exception(f"SSH connection test failed. Instance may not be ready yet. Error: {error_output[:200]}")
            
            # Ensure the target directory exists on the remote instance
            mkdir_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                "mkdir -p /workspace/data && mkdir -p /workspace/output/training && echo 'Directories ready'"
            ]
            mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=15)
            if mkdir_result.returncode != 0:
                error_output = mkdir_result.stderr or mkdir_result.stdout
                # Filter out the Vast.ai welcome message
                if "Welcome to vast.ai" in error_output:
                    # Check if the actual error is something else
                    if "Permission denied" in error_output or "publickey" in error_output.lower():
                        raise Exception(f"SSH authentication failed. SSH keys are required. Please configure SSH keys in Vast.ai account settings.")
                    # If it's just the welcome message, the command might have succeeded
                    # Check if directories exist
                    check_cmd = [
                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                        f"root@{ssh_host}",
                        "test -d /workspace/data && echo 'exists' || echo 'missing'"
                    ]
                    check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=15)
                    if "exists" not in check_result.stdout:
                        raise Exception(f"Could not create or verify /workspace/data directory. Instance may not be fully ready. Please wait a moment and refresh status.")
                else:
                    raise Exception(f"Could not create directories on instance: {error_output[:200]}")
            
            # Upload training data JSONL (from temp directory) with retry logic
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    if dataset_path.exists():
                        cmd = [
                            "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                            str(dataset_path),
                            f"root@{ssh_host}:/workspace/data/training_data.jsonl"
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode != 0:
                            error_output = result.stderr or result.stdout
                            # Filter out Vast.ai welcome message from error output
                            if "Welcome to vast.ai" in error_output:
                                # Extract the actual error (usually after the welcome message)
                                lines = error_output.split('\n')
                                actual_errors = [line for line in lines if line and 'Welcome to vast.ai' not in line and 'Have fun!' not in line]
                                error_output = '\n'.join(actual_errors) if actual_errors else error_output
                            
                            # Check for specific errors
                            if "No such file or directory" in error_output:
                                # Directory doesn't exist - try creating it and retry
                                if attempt < max_retries - 1:
                                    print(f"[DEBUG] Directory missing, creating it and retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                                    mkdir_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "mkdir -p /workspace/data"
                                    ]
                                    subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=15)
                                    time.sleep(retry_delay)
                                    continue
                                raise Exception(f"Directory /workspace/data does not exist on instance. Instance may not be fully ready. Please wait a moment and refresh status.")
                            elif "Connection refused" in error_output and attempt < max_retries - 1:
                                print(f"[DEBUG] Connection refused, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                                time.sleep(retry_delay)
                                continue
                            elif "Permission denied" in error_output or "publickey" in error_output.lower():
                                raise Exception(f"SSH authentication failed. SSH keys are required. Please configure SSH keys in Vast.ai account settings.")
                            raise Exception(f"Failed to upload dataset: {error_output[:500]}")
                        break  # Success
                except subprocess.TimeoutExpired:
                    if attempt < max_retries - 1:
                        print(f"[DEBUG] Upload timed out, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    raise Exception("Upload timed out after multiple retries. Instance may not be ready.")
            
            # Upload config file (from temp directory) with retry logic
            for attempt in range(max_retries):
                try:
                    if config_path.exists():
                        cmd = [
                            "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                            str(config_path),
                            f"root@{ssh_host}:/workspace/data/axolotl_config.yaml"
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode != 0:
                            error_output = result.stderr or result.stdout
                            # Filter out Vast.ai welcome message from error output
                            if "Welcome to vast.ai" in error_output:
                                # Extract the actual error (usually after the welcome message)
                                lines = error_output.split('\n')
                                actual_errors = [line for line in lines if line and 'Welcome to vast.ai' not in line and 'Have fun!' not in line]
                                error_output = '\n'.join(actual_errors) if actual_errors else error_output
                            
                            # Check for specific errors
                            if "No such file or directory" in error_output:
                                # Directory doesn't exist - try creating it and retry
                                if attempt < max_retries - 1:
                                    print(f"[DEBUG] Directory missing, creating it and retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                                    mkdir_cmd = [
                                        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                                        f"root@{ssh_host}",
                                        "mkdir -p /workspace/data"
                                    ]
                                    subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=15)
                                    time.sleep(retry_delay)
                                    continue
                                raise Exception(f"Directory /workspace/data does not exist on instance. Instance may not be fully ready. Please wait a moment and refresh status.")
                            elif "Connection refused" in error_output and attempt < max_retries - 1:
                                print(f"[DEBUG] Connection refused, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                                time.sleep(retry_delay)
                                continue
                            elif "Permission denied" in error_output or "publickey" in error_output.lower():
                                raise Exception(f"SSH authentication failed. SSH keys are required. Please configure SSH keys in Vast.ai account settings.")
                            raise Exception(f"Failed to upload config: {error_output[:500]}")
                        break  # Success
                except subprocess.TimeoutExpired:
                    if attempt < max_retries - 1:
                        print(f"[DEBUG] Upload timed out, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    raise Exception("Upload timed out after multiple retries. Instance may not be ready.")
            
            # Upload previous adapter weights if we have them (for V2+ training)
            previous_adapter_path = package_info.get("previous_adapter_path")
            if previous_adapter_path:
                adapter_path = Path(previous_adapter_path)
                if adapter_path.exists():
                    # Upload entire adapter directory
                    remote_adapter_path = "/workspace/data/previous_adapter"
                    cmd = [
                        "scp", "-r", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no",
                        str(adapter_path),
                        f"root@{ssh_host}:{remote_adapter_path}"
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode != 0:
                        raise Exception(f"Failed to upload previous adapter: {result.stderr}")
                    # Update config to point to uploaded adapter
                    # The config will be updated to use /workspace/data/previous_adapter
                    package_info["remote_adapter_path"] = remote_adapter_path
            
            return True
        except subprocess.TimeoutExpired:
            raise Exception("File upload timed out. Instance may not be ready yet.")
        except FileNotFoundError:
            raise Exception("SCP not found. Please install OpenSSH client to upload files automatically.")
        except Exception as e:
            raise Exception(f"Error uploading files: {str(e)}")
    
    def finalize_training_success(self, instance_id: str, output_dir: str = "/workspace/output") -> Dict:
        """
        Finalize a successful training job: create version directory, move files, download weights
        
        Args:
            instance_id: Vast.ai instance ID
            output_dir: Remote output directory on instance
        
        Returns:
            Dictionary with version information
        """
        self.logger.log("INFO", f"Starting finalization for instance {instance_id}", {"output_dir": output_dir})
        
        # Get job info - try multiple ways to find it
        jobs = self._load_jobs()
        self.logger.log("INFO", f"Loaded {len(jobs)} job(s) from file")
        
        # Try to find job by instance_id (as string or int)
        job = None
        for j in jobs:
            job_instance_id = j.get("instance_id")
            if job_instance_id:
                # Compare as strings to handle int/string mismatches
                if str(job_instance_id) == str(instance_id):
                    job = j
                    break
        
        # If still not found, try to find by model_name and most recent
        if not job:
            model_jobs = [j for j in jobs if j.get("model_name") == self.model_name]
            if model_jobs:
                # Get most recent job for this model
                job = sorted(model_jobs, key=lambda x: x.get("created_at", ""), reverse=True)[0]
                # Update instance_id in case it was different
                job["instance_id"] = str(instance_id)
        
        if not job:
            # Create a minimal job record if none exists
            job = {
                "instance_id": str(instance_id),
                "model_name": self.model_name,
                "status": "completed",
                "created_at": datetime.now().isoformat()
            }
            jobs.append(job)
            self._save_job(job)
        
        # Create version folder now that training is successful
        # Check if version_dir already exists in job (to avoid creating duplicates)
        # Use explicit Path import to avoid scoping issues
        from pathlib import Path as PathClass
        if job.get("version_dir"):
            existing_version_dir = PathClass(job["version_dir"])
            if existing_version_dir.exists():
                version_dir = existing_version_dir
                training_dir = version_dir / "training"
                # Ensure training dir exists
                training_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Version dir was specified but doesn't exist - create it
                version_dir = self.model_manager.create_version_folder(self.model_name)
                training_dir = version_dir / "training"
        else:
            # No version dir in job - create new one
            version_dir = self.model_manager.create_version_folder(self.model_name)
            training_dir = version_dir / "training"
        
        # Move files from queue to version training folder
        queue_dir = get_model_queue_dir(self.model_name)
        moved_files = []
        
        if queue_dir.exists():
            # Move all queued files (JSON, JSONL, TXT) and their metadata
            for file_path in queue_dir.iterdir():
                if file_path.is_file():
                    # Skip metadata files - we'll handle them separately
                    if file_path.name.endswith("_metadata.json"):
                        continue
                    try:
                        dest_path = training_dir / file_path.name
                        shutil.move(str(file_path), str(dest_path))
                        moved_files.append(file_path.name)
                        print(f"✅ Moved file to version folder: {file_path.name}")
                        
                        # Also move corresponding metadata file if it exists
                        # file_path is already a Path object, so use .stem directly
                        metadata_file = queue_dir / f"{file_path.stem}_metadata.json"
                        if metadata_file.exists():
                            metadata_dest = training_dir / metadata_file.name
                            shutil.move(str(metadata_file), str(metadata_dest))
                            print(f"✅ Moved metadata: {metadata_file.name}")
                    except Exception as e:
                        print(f"❌ Error moving file {file_path.name}: {e}")
        else:
            print(f"⚠️ Queue directory does not exist: {queue_dir}")
        
        if not moved_files:
            print(f"⚠️ Warning: No files were moved from queue. Queue directory: {queue_dir}")
            print(f"   Queue exists: {queue_dir.exists() if queue_dir else False}")
            if queue_dir and queue_dir.exists():
                files_in_queue = list(queue_dir.iterdir())
                print(f"   Files in queue: {[f.name for f in files_in_queue if f.is_file()]}")
        
        # Download weights
        weights_dir = version_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get instance info
            instance_status = self.vast_client.get_instance_status(instance_id)
            
            # Extract instance data if nested
            if "instances" in instance_status and isinstance(instance_status["instances"], dict):
                instance_data = instance_status["instances"]
            else:
                instance_data = instance_status
            
            # Try multiple possible fields for SSH info
            ssh_host = (instance_data.get("ssh_host") or 
                       instance_data.get("public_ipaddr") or
                       instance_data.get("ipaddr") or
                       instance_data.get("ssh_ip"))
            ssh_port = (instance_data.get("ssh_port") or 
                       instance_data.get("port") or 
                       22)
            
            # If still no SSH host, try to get from job record
            if not ssh_host:
                ssh_host = job.get("ssh_host")
                ssh_port = job.get("ssh_port", 22)
            
            if not ssh_host:
                raise Exception("Could not get SSH connection details from instance. SSH keys may be required to download weights. You can manually download weights from Vast.ai and place them in the weights directory.")
            
            # First, check what files actually exist on the remote instance
            self.logger.log("INFO", f"Checking for training output files on instance {instance_id}", {
                "ssh_host": ssh_host,
                "ssh_port": ssh_port,
                "output_dir": output_dir
            })
            
            check_files_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                f"find {output_dir} -type f -name '*.bin' -o -name '*.safetensors' -o -name 'adapter_config.json' 2>/dev/null | head -20"
            ]
            check_result = subprocess.run(check_files_cmd, capture_output=True, text=True, timeout=30)
            available_files = []
            if check_result.returncode == 0 and check_result.stdout.strip():
                available_files = [f.strip() for f in check_result.stdout.strip().split('\n') if f.strip()]
                self.logger.log("INFO", f"Found {len(available_files)} weight files on instance", {"files": available_files[:10]})
            else:
                self.logger.log("WARNING", "Could not list files on instance, will try standard paths", {
                    "error": check_result.stderr[:200] if check_result.stderr else "No output"
                })
            
            # Try to download adapter weights (LoRA) first
            adapter_path = f"{output_dir}/adapter"
            weights_downloaded = False
            downloaded_files = []
            
            # Try adapter directory first
            self.logger.log("INFO", f"Attempting to download adapter from {adapter_path}")
            cmd = [
                "scp", "-r", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                f"root@{ssh_host}:{adapter_path}",
                str(weights_dir)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Check if files were actually downloaded
                adapter_dir = weights_dir / "adapter"
                if adapter_dir.exists() and any(adapter_dir.iterdir()):
                    downloaded_files = [f.name for f in adapter_dir.iterdir() if f.is_file()]
                    weights_downloaded = True
                    self.logger.log("SUCCESS", f"Downloaded adapter weights: {len(downloaded_files)} files", {"files": downloaded_files[:10]})
                else:
                    self.logger.log("WARNING", "SCP succeeded but no files found in adapter directory")
            
            # If adapter download failed, try full model directory
            if not weights_downloaded:
                self.logger.log("INFO", "Adapter download failed, trying full model directory")
                model_path = f"{output_dir}"
                cmd = [
                    "scp", "-r", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                    f"root@{ssh_host}:{model_path}/*",
                    str(weights_dir)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    # Check if files were actually downloaded
                    downloaded_files = [f.name for f in weights_dir.iterdir() if f.is_file()]
                    if downloaded_files:
                        weights_downloaded = True
                        self.logger.log("SUCCESS", f"Downloaded model weights: {len(downloaded_files)} files", {"files": downloaded_files[:10]})
            
            # If still failed, try downloading specific files one by one
            if not weights_downloaded:
                self.logger.log("INFO", "Trying to download specific weight files")
                for file_name in ["adapter_model.bin", "adapter_model.safetensors", "adapter_config.json", "training_args.bin"]:
                    file_path = f"{output_dir}/adapter/{file_name}"
                    # Also try without adapter subdirectory
                    alt_paths = [file_path, f"{output_dir}/{file_name}"]
                    for try_path in alt_paths:
                        cmd = [
                            "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                            f"root@{ssh_host}:{try_path}",
                            str(weights_dir / file_name)
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode == 0 and (weights_dir / file_name).exists():
                            downloaded_files.append(file_name)
                            self.logger.log("SUCCESS", f"Downloaded {file_name}")
                            weights_downloaded = True
                            break
                
                # Check if we got at least adapter_config.json (minimum required)
                if not any("adapter_config.json" in f for f in downloaded_files):
                    raise Exception(f"No weight files were downloaded. Available files on instance: {available_files[:10] if available_files else 'unknown'}. Please check if training actually completed.")
            
            if not weights_downloaded:
                raise Exception(f"Failed to download weights. SCP commands completed but no files found. Available files on instance: {available_files[:10] if available_files else 'unknown'}")
            
            self.logger.log("SUCCESS", f"Weight download completed: {len(downloaded_files)} files", {"files": downloaded_files})
        except subprocess.TimeoutExpired:
            raise Exception("Download timed out. Weights may still be uploading.")
        except FileNotFoundError:
            raise Exception("SCP not found. Please install OpenSSH client to download weights automatically.")
        except Exception as e:
            raise Exception(f"Error downloading weights: {str(e)}")
        
        # Do NOT stop the instance - user must shut it down manually
        # This prevents accidental data loss and gives users control
        print(f"ℹ️ Instance {instance_id} will remain running. User must shut it down manually in Vast.ai to stop charges.")
        instance_stopped = False  # Track that we did NOT stop it
        stop_error = None
        
        # Get model metadata for version info
        metadata = self.model_manager.get_model_metadata(self.model_name)
        base_model = metadata.get("base_model", "llama2") if metadata else "llama2"
        package_info = job.get("package_info", {})
        hf_model = package_info.get("hf_model", "")
        
        # Save version metadata
        version_metadata = {
            "version": version_dir.name,
            "created_date": datetime.now().isoformat(),
            "base_model": base_model,
            "hf_model": hf_model,
            "dataset_stats": package_info.get("dataset_stats", {}),
            "moved_files": moved_files,
            "status": "completed",
            "instance_id": instance_id,
            "weights_downloaded": True,
            "weights_path": str(weights_dir),
            "instance_stopped": False,  # We don't stop instances anymore
            "stop_error": None,
            "note": "Instance is still running. You must shut it down manually in Vast.ai to stop charges."
        }
        metadata_path = version_dir / "version_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        # Update job info with version and mark as finalized
        job["version"] = version_dir.name
        job["version_dir"] = str(version_dir)
        job["status"] = "completed"
        job["finalized"] = True  # Mark as finalized to prevent going back to launching
        job["finalized_at"] = datetime.now().isoformat()
        job["instance_stopped"] = False  # We don't stop instances anymore
        job["weights_downloaded"] = True
        job["weights_path"] = str(weights_dir)
        self._save_job(job)
        self.logger.log("SUCCESS", f"Training job finalized successfully", {
            "version": version_dir.name,
            "weights_downloaded": True,
            "instance_stopped": False,
            "note": "Instance still running - user must shut down manually"
        })
        
        # Clean up temp directory
        temp_dir_str = package_info.get("temp_dir", "")
        if temp_dir_str:
            temp_dir = Path(temp_dir_str)
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass  # Ignore cleanup errors
        
        return {
            "version": version_dir.name,
            "version_dir": str(version_dir),
            "weights_dir": str(weights_dir),
            "moved_files": moved_files
        }
    
    def undo_finalize(self, instance_id: str) -> Dict:
        """
        Undo a finalize operation: move files back to queue, delete version folder, reset job
        
        Args:
            instance_id: Vast.ai instance ID
        
        Returns:
            Dictionary with undo results
        """
        # Get job info
        jobs = self._load_jobs()
        job = None
        for j in jobs:
            job_instance_id = j.get("instance_id")
            if job_instance_id and str(job_instance_id) == str(instance_id):
                job = j
                break
        
        if not job:
            raise Exception(f"Job not found for instance {instance_id}")
        
        version_dir_str = job.get("version_dir")
        if not version_dir_str:
            raise Exception("No version directory found in job record")
        
        # Use explicit Path import to avoid scoping issues
        from pathlib import Path as PathClass
        version_dir = PathClass(version_dir_str)
        if not version_dir.exists():
            raise Exception(f"Version directory does not exist: {version_dir}")
        
        # Get list of moved files from version metadata
        moved_files = []
        version_metadata_path = version_dir / "version_metadata.json"
        if version_metadata_path.exists():
            with open(version_metadata_path, 'r') as f:
                version_metadata = json.load(f)
                moved_files = version_metadata.get("moved_files", [])
        
        # If no metadata, try to get files from training directory
        training_dir = version_dir / "training"
        if not moved_files and training_dir.exists():
            for file_path in training_dir.iterdir():
                if file_path.is_file() and not file_path.name.endswith("_metadata.json"):
                    moved_files.append(file_path.name)
        
        # Move files back to queue
        queue_dir = get_model_queue_dir(self.model_name)
        queue_dir.mkdir(parents=True, exist_ok=True)
        
        restored_files = []
        if training_dir.exists():
            # Move ALL files from training directory back to queue
            # This includes all files that were moved from queue (JSON, JSONL, TXT) and their metadata
            for file_path in training_dir.iterdir():
                if file_path.is_file():
                    filename = file_path.name
                    # Skip version metadata file (not a queue file)
                    if filename == "version_metadata.json":
                        continue
                    
                    dest_path = queue_dir / filename
                    try:
                        # Check if destination already exists
                        if dest_path.exists():
                            # Don't overwrite - keep the one in queue
                            print(f"Skipping {filename} - already exists in queue")
                            continue
                        
                        # Move the file
                        shutil.move(str(file_path), str(dest_path))
                        restored_files.append(filename)
                        print(f"✅ Restored file: {filename}")
                    except Exception as e:
                        print(f"❌ Error moving file {filename} back to queue: {e}")
            
            # Also check for any remaining files (in case of errors)
            remaining_files = list(training_dir.iterdir())
            if remaining_files:
                print(f"⚠️ Warning: {len(remaining_files)} file(s) still in training directory after restore")
                for f in remaining_files:
                    if f.is_file() and f.name != "version_metadata.json":
                        print(f"  - {f.name}")
        
        # Delete version folder
        try:
            shutil.rmtree(version_dir)
        except Exception as e:
            raise Exception(f"Error deleting version folder: {e}")
        
        # Reset job status
        job["status"] = "completed"  # Keep as completed, but not finalized
        job["version"] = None
        job["version_dir"] = None
        job["weights_downloaded"] = False
        job["weights_path"] = None
        self._save_job(job)
        
        return {
            "success": True,
            "restored_files": restored_files,
            "deleted_version": str(version_dir),
            "message": f"Undo complete: {len(restored_files)} file(s) restored to queue, version folder deleted"
        }
    
    def validate_instance(self, instance_id: str, ssh_host_override: Optional[str] = None, ssh_port_override: Optional[int] = None) -> Dict:
        """
        Validate that an existing instance meets all requirements for training
        
        Args:
            instance_id: Vast.ai instance ID to validate
        
        Returns:
            Dictionary with validation results:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "details": Dict with detailed information
            }
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # 1. Check instance exists and is running
            try:
                api_response = self.vast_client.get_instance_status(instance_id)
                
                # Extract instance data - API response may have 'instances' key with nested data
                if "instances" in api_response and isinstance(api_response["instances"], dict):
                    instance_status = api_response["instances"]
                elif isinstance(api_response, dict) and "actual_status" in api_response:
                    # Already at instance level
                    instance_status = api_response
                else:
                    # Try to find instance data in response
                    instance_status = api_response
                
                # Get actual_status - check multiple possible fields
                actual_status = (instance_status.get("actual_status") or 
                               instance_status.get("status") or 
                               "unknown")
                details["status"] = actual_status
                
                if actual_status.lower() != "running":
                    errors.append(f"Instance is not running. Current status: {actual_status}")
                    return {
                        "valid": False,
                        "errors": errors,
                        "warnings": warnings,
                        "details": details
                    }
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "not found" in error_msg.lower():
                    errors.append(f"Instance {instance_id} not found. It may have been destroyed.")
                else:
                    errors.append(f"Could not get instance status: {error_msg}")
                return {
                    "valid": False,
                    "errors": errors,
                    "warnings": warnings,
                    "details": details
                }
            
            # 2. Check SSH connectivity
            try:
                # Use manual override if provided, otherwise get from API
                if ssh_host_override and ssh_port_override:
                    ssh_host = ssh_host_override
                    ssh_port = ssh_port_override
                    details["ssh_host"] = ssh_host
                    details["ssh_port"] = ssh_port
                    details["ssh_override"] = True
                else:
                    ssh_info = self.get_instance_ssh_info(instance_id)
                    ssh_host = ssh_info.get("host")
                    ssh_port = ssh_info.get("port", 22)
                    details["ssh_host"] = ssh_host
                    details["ssh_port"] = ssh_port
                    details["ssh_override"] = False
                    # Show what we found vs what might be available
                    raw_data = ssh_info.get("raw_data", {})
                    if raw_data:
                        details["ssh_raw_data"] = raw_data
                
                if not ssh_host:
                    errors.append("SSH connection details not available. Instance may still be initializing.")
                    return {
                        "valid": False,
                        "errors": errors,
                        "warnings": warnings,
                        "details": details
                    }
                
                # Use the user's actual known_hosts file so we can leverage keys they've already accepted
                # Also try to add the key if it's not there
                user_known_hosts = os.path.expanduser("~/.ssh/known_hosts")
                known_hosts_file = user_known_hosts if os.path.exists(user_known_hosts) else "/dev/null"
                
                # Try to add the host key if it's not already in known_hosts
                try:
                    # Check if key is already in known_hosts
                    key_in_file = False
                    if os.path.exists(user_known_hosts):
                        with open(user_known_hosts, 'r') as f:
                            if ssh_host in f.read() or f"[{ssh_host}]:{ssh_port}" in f.read():
                                key_in_file = True
                    
                    # If not in file, try to add it
                    if not key_in_file:
                        ssh_keyscan_cmd = [
                            "ssh-keyscan", "-p", str(ssh_port), "-H", ssh_host
                        ]
                        keyscan_result = subprocess.run(ssh_keyscan_cmd, capture_output=True, text=True, timeout=5)
                        if keyscan_result.returncode == 0 and keyscan_result.stdout:
                            # Append to user's known_hosts
                            with open(user_known_hosts, 'a') as f:
                                f.write(keyscan_result.stdout)
                            known_hosts_file = user_known_hosts
                except Exception as e:
                    # If we can't modify known_hosts, that's okay - use what we have
                    pass
                
                # Test SSH connection
                # Use accept-new to auto-accept new keys, but prefer existing known_hosts
                # Don't use LogLevel=ERROR as it suppresses useful error messages
                test_ssh_cmd = [
                    "ssh", "-p", str(ssh_port), 
                    "-o", "StrictHostKeyChecking=accept-new",  # Auto-accept new host keys
                    "-o", f"UserKnownHostsFile={known_hosts_file}",
                    "-o", "ConnectTimeout=10",
                    "-o", "PasswordAuthentication=no",  # Prefer key-based auth
                    f"root@{ssh_host}",
                    "echo 'SSH_OK'"
                ]
                
                ssh_test = subprocess.run(test_ssh_cmd, capture_output=True, text=True, timeout=15)
                
                if ssh_test.returncode != 0 or "SSH_OK" not in ssh_test.stdout:
                    # SSH connection failed - provide helpful error message
                    stderr_msg = ssh_test.stderr.strip() if ssh_test.stderr else ""
                    stdout_msg = ssh_test.stdout.strip() if ssh_test.stdout else ""
                    combined_output = (stderr_msg + " " + stdout_msg).lower()
                    
                    # Always include the actual error output first
                    error_detail = f"SSH connection failed (return code: {ssh_test.returncode})"
                    
                    # Always show both stderr and stdout for debugging
                    error_detail += "\n\nSSH Error Output (stderr):"
                    if stderr_msg:
                        error_detail += f"\n{stderr_msg}"
                    else:
                        error_detail += "\n(empty - no stderr output)"
                    
                    error_detail += "\n\nSSH Standard Output (stdout):"
                    if stdout_msg:
                        error_detail += f"\n{stdout_msg}"
                    else:
                        error_detail += "\n(empty - no stdout output)"
                    
                    # Check for specific error patterns and provide guidance
                    if "connection closed" in combined_output:
                        error_detail += ("\n\nDiagnosis: Connection is being established but immediately closed by the server.")
                        # If instance is running, this is likely a Vast.ai SSH gateway issue
                        # Make it a warning instead of error since user can connect manually
                        if actual_status.lower() == "running":
                            error_detail += (f"\n\nNote: This is common with Vast.ai instances. The SSH gateway may close "
                                           "non-interactive connections. If you can connect manually "
                                           f"(ssh -p {ssh_port} root@{ssh_host}), the instance is fine.")
                            error_detail += ("\nYou can proceed to Phase 2 if you've verified SSH works manually.")
                            warnings.append(f"SSH automated test failed (connection closed by server). If you can connect manually (ssh -p {ssh_port} root@{ssh_host}), you can proceed to Phase 2.")
                            # Don't add as error - just warning
                        else:
                            error_detail += ("\nSolution: Try connecting manually first: ssh -p {} root@{}".format(ssh_port, ssh_host))
                            errors.append(f"SSH connection test failed. {error_detail}")
                            errors.append("Note: SSH connectivity is required for Phase 2 (file upload).")
                    elif "connection refused" in combined_output:
                        error_detail += ("\n\nDiagnosis: Connection refused - SSH service may not be running.")
                        error_detail += ("\nSolution: Check the instance status in Vast.ai dashboard.")
                    elif "permission denied" in combined_output or "authentication failed" in combined_output:
                        error_detail += ("\n\nDiagnosis: Authentication failed.")
                        error_detail += ("\nSolution: Ensure your SSH keys are configured in your Vast.ai account settings.")
                    elif "host key verification failed" in combined_output or ("host key" in combined_output and "verification" in combined_output):
                        error_detail += (f"\n\nDiagnosis: Host key verification issue.")
                        error_detail += (f"\nSolution: Try: ssh -p {ssh_port} root@{ssh_host} and accept the host key.")
                    elif "could not resolve hostname" in combined_output:
                        error_detail += "\n\nDiagnosis: Could not resolve hostname."
                        error_detail += "\nSolution: Check network connectivity."
                    elif ssh_test.returncode == 255:
                        # Generic SSH error - provide common solutions
                        error_detail += ("\n\nCommon causes for SSH error 255:")
                        error_detail += ("\n  1. Host key not accepted - connect manually first: ssh -p {} root@{}".format(ssh_port, ssh_host))
                        error_detail += ("\n  2. SSH keys not configured - check Vast.ai account settings")
                        error_detail += ("\n  3. Instance SSH service not ready - wait a few minutes")
                        error_detail += ("\n  4. Network/firewall issues")
                        errors.append(f"SSH connection test failed. {error_detail}")
                        errors.append("Note: SSH connectivity is required for Phase 2 (file upload).")
                    else:
                        # Other errors - keep as errors
                        errors.append(f"SSH connection test failed. {error_detail}")
                        errors.append("Note: SSH connectivity is required for Phase 2 (file upload).")
                    
                    # Continue with other checks to provide full validation report
                else:
                    details["ssh_connection_ok"] = True
            except subprocess.TimeoutExpired:
                errors.append("SSH connection timed out. Instance may be overloaded or network issues.")
                return {
                    "valid": False,
                    "errors": errors,
                    "warnings": warnings,
                    "details": details
                }
            except FileNotFoundError:
                errors.append("SSH client not found. Please install OpenSSH client.")
                return {
                    "valid": False,
                    "errors": errors,
                    "warnings": warnings,
                    "details": details
                }
            except Exception as e:
                errors.append(f"SSH connection error: {str(e)}")
                return {
                    "valid": False,
                    "errors": errors,
                    "warnings": warnings,
                    "details": details
                }
            
            # 3. Check disk space
            try:
                check_disk_cmd = [
                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                    f"root@{ssh_host}",
                    "df -h /workspace 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || echo '0'"
                ]
                disk_result = subprocess.run(check_disk_cmd, capture_output=True, text=True, timeout=10)
                if disk_result.returncode == 0:
                    try:
                        available_gb = float(disk_result.stdout.strip().replace('G', '').replace('M', ''))
                        # If in MB, convert to GB
                        if 'M' in disk_result.stdout:
                            available_gb = available_gb / 1024
                        details["disk_available_gb"] = available_gb
                        if available_gb < 50:
                            warnings.append(f"Low disk space: {available_gb:.1f} GB available. Recommend at least 50 GB.")
                    except:
                        warnings.append("Could not parse disk space information.")
                else:
                    warnings.append("Could not check disk space.")
            except Exception as e:
                warnings.append(f"Disk space check failed: {str(e)[:100]}")
            
            # 4. Check GPU availability
            try:
                check_gpu_cmd = [
                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                    f"root@{ssh_host}",
                    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'NO_GPU'"
                ]
                gpu_result = subprocess.run(check_gpu_cmd, capture_output=True, text=True, timeout=10)
                if "NO_GPU" in gpu_result.stdout or gpu_result.returncode != 0:
                    errors.append("GPU not detected. Instance may not have GPU access or nvidia-smi is not available.")
                else:
                    gpu_info = gpu_result.stdout.strip().split('\n')
                    details["gpu_info"] = gpu_info
                    if not gpu_info or not gpu_info[0] or gpu_info[0] == "NO_GPU":
                        errors.append("No GPU detected on instance.")
            except Exception as e:
                warnings.append(f"GPU check failed: {str(e)[:100]}")
            
            # 5. Check required directories
            try:
                check_dirs_cmd = [
                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                    f"root@{ssh_host}",
                    "test -d /workspace && test -d /workspace/data && test -d /workspace/output && echo 'DIRS_OK' || echo 'DIRS_MISSING'"
                ]
                dirs_result = subprocess.run(check_dirs_cmd, capture_output=True, text=True, timeout=10)
                if "DIRS_MISSING" in dirs_result.stdout:
                    warnings.append("Required directories (/workspace, /workspace/data, /workspace/output) may be missing. They will be created if needed.")
                else:
                    details["directories_ok"] = True
            except Exception as e:
                warnings.append(f"Directory check failed: {str(e)[:100]}")
            
            # 6. Check Python and basic tools
            try:
                check_python_cmd = [
                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                    f"root@{ssh_host}",
                    "python3 --version 2>&1 || python --version 2>&1 || echo 'NO_PYTHON'"
                ]
                python_result = subprocess.run(check_python_cmd, capture_output=True, text=True, timeout=10)
                if "NO_PYTHON" in python_result.stdout or python_result.returncode != 0:
                    warnings.append("Python may not be available. Training setup will install required tools.")
                else:
                    details["python_version"] = python_result.stdout.strip()
            except Exception as e:
                warnings.append(f"Python check failed: {str(e)[:100]}")
            
            # Note: We no longer use onstart scripts - all training is started manually via SSH
            # Removed onstart script checks since they're not relevant for existing instances
            
            # If we got here with no errors, instance is valid
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "details": details
            }
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "details": details
            }
    
    def get_instance_ssh_info(self, instance_id: str) -> Dict:
        """
        Get SSH connection information for an instance
        
        Args:
            instance_id: Vast.ai instance ID
        
        Returns:
            Dictionary with SSH connection details
        """
        instance_status = self.vast_client.get_instance_status(instance_id)
        
        # Extract instance data if nested
        if "instances" in instance_status and isinstance(instance_status["instances"], dict):
            instance_data = instance_status["instances"]
        else:
            instance_data = instance_status
        
        # Try multiple fields to find the correct SSH connection info
        # Vast.ai may provide different connection methods (gateway vs direct)
        # Priority: Direct IP first, then gateway
        public_ip = instance_data.get("public_ipaddr") or instance_data.get("ipaddr")
        ssh_ip = instance_data.get("ssh_ip")
        ssh_host_gateway = instance_data.get("ssh_host")
        
        # Determine if we're using direct IP or gateway
        using_direct_ip = bool(public_ip or ssh_ip)
        using_gateway = bool(ssh_host_gateway and not using_direct_ip)
        
        # Select host - prefer direct IP
        ssh_host = (public_ip or
                   ssh_ip or
                   ssh_host_gateway or
                   instance_data.get("host"))
        
        # For port selection:
        # - Direct IP connections typically use port 22 (standard SSH)
        # - Gateway/proxy connections use ssh_port (which may be a non-standard port like 16890)
        ssh_port = None
        
        if using_direct_ip:
            # For direct SSH connections, use port 22 (standard SSH port)
            # Vast.ai direct connections always use port 22
            ssh_port = 22
        elif using_gateway:
            # For gateway/proxy connections, use the ssh_port field
            ssh_port = (instance_data.get("ssh_port") or
                       instance_data.get("port") or
                       instance_data.get("conn_port") or
                       instance_data.get("connection_port") or
                       22)
        else:
            # Fallback - default to 22
            ssh_port = 22
        
        # Ensure port is an integer
        try:
            ssh_port = int(ssh_port)
        except (ValueError, TypeError):
            ssh_port = 22
        
        # Check for Jupyter connection info (if available)
        jupyter_url = instance_data.get("jupyter_url") or instance_data.get("jupyter")
        jupyter_port = instance_data.get("jupyter_port")
        
        return {
            "host": ssh_host,
            "port": ssh_port,
            "status": instance_data.get("actual_status", "unknown"),
            # Jupyter info if available
            "jupyter_url": jupyter_url,
            "jupyter_port": jupyter_port,
            # Also return raw data for debugging
            "raw_data": {
                "connection_type": "direct" if using_direct_ip else ("gateway" if using_gateway else "unknown"),
                "ssh_host": instance_data.get("ssh_host"),
                "public_ipaddr": instance_data.get("public_ipaddr"),
                "ipaddr": instance_data.get("ipaddr"),
                "ssh_ip": instance_data.get("ssh_ip"),
                "ssh_port": instance_data.get("ssh_port"),
                "port": instance_data.get("port"),
                "conn_port": instance_data.get("conn_port"),
                "connection_port": instance_data.get("connection_port"),
                "jupyter_url": jupyter_url,
                "jupyter_port": jupyter_port,
                "all_keys": list(instance_data.keys())  # For debugging - see all available fields
            }
        }
    
    def check_training_status(self, instance_id: str) -> Dict:
        """
        Check training status by SSHing into instance and checking logs
        
        Args:
            instance_id: Vast.ai instance ID
        
        Returns:
            Dictionary with training status information
        """
        try:
            # Get SSH connection info
            ssh_info = self.get_instance_ssh_info(instance_id)
            ssh_host = ssh_info.get("host")
            ssh_port = ssh_info.get("port", 22)
            
            if not ssh_host:
                return {
                    "error": "Could not get SSH connection details",
                    "status": "unknown"
                }
            
            # Check if training is still running
            # Check for training process
            check_process_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                "ps aux | grep -E '(axolotl|accelerate|train)' | grep -v grep || echo 'no_training'"
            ]
            
            process_result = subprocess.run(
                check_process_cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            training_running = "no_training" not in process_result.stdout
            
            # Check for output directory and latest checkpoint
            check_output_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                "ls -la /workspace/output/training 2>/dev/null | tail -5 || echo 'no_output'"
            ]
            
            output_result = subprocess.run(
                check_output_cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            has_output = "no_output" not in output_result.stdout
            
            # Get last few lines of training logs if available
            # Check multiple possible log locations and also check stdout/stderr
            log_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                """bash -c '
                    # Try to find and tail training logs
                    if [ -f /workspace/output/training/training.log ]; then
                        tail -100 /workspace/output/training/training.log
                    elif [ -f /workspace/axolotl/training.log ]; then
                        tail -100 /workspace/axolotl/training.log
                    elif [ -f /tmp/training.log ]; then
                        tail -100 /tmp/training.log
                    else
                        # Try to find any log files
                        found_log=$(find /workspace -name "*.log" -type f 2>/dev/null | head -1)
                        if [ -n "$found_log" ]; then
                            tail -100 "$found_log"
                        else
                            echo "no_logs"
                        fi
                    fi
                '"""
            ]
            
            log_result = subprocess.run(
                log_cmd,
                capture_output=True,
                text=True,
                timeout=15,
                shell=True
            )
            
            logs = None
            if log_result.returncode == 0 and log_result.stdout.strip() and "no_logs" not in log_result.stdout:
                logs = log_result.stdout.strip()
            
            # If no log file found, try to get output from any running processes
            if not logs or len(logs) < 10:
                # Check for output in /workspace/axolotl directory (where training runs)
                check_axolotl_output_cmd = [
                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                    f"root@{ssh_host}",
                    """bash -c '
                        # Check for any output files or logs in axolotl directory
                        if [ -d /workspace/axolotl ]; then
                            # Look for any recent output
                            find /workspace/axolotl -type f -name "*.log" -o -name "*.out" -o -name "*.err" 2>/dev/null | head -3 | while read f; do
                                echo "=== $f ==="
                                tail -30 "$f" 2>/dev/null
                            done
                        fi
                        # Also check if there's any output in the output directory
                        if [ -d /workspace/output/training ]; then
                            ls -la /workspace/output/training/ 2>/dev/null | head -20
                        fi
                    '"""
                ]
                axolotl_output_result = subprocess.run(check_axolotl_output_cmd, capture_output=True, text=True, timeout=15, shell=True)
                if axolotl_output_result.returncode == 0 and axolotl_output_result.stdout.strip():
                    logs = (logs + "\n\n=== Additional Output ===\n" + axolotl_output_result.stdout.strip()) if logs else axolotl_output_result.stdout.strip()
            
            # Note: Removed onstart log checks since we no longer use onstart scripts
            
            # Also try to get output from stdout/stderr of running process
            if not logs or len(logs.strip()) < 10:
                # Try to get output from training process
                output_cmd = [
                    "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                    f"root@{ssh_host}",
                    "journalctl -u vastai --no-pager -n 50 2>/dev/null || dmesg | tail -30 2>/dev/null || echo 'no_system_logs'"
                ]
                output_result = subprocess.run(
                    output_cmd,
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                if "no_system_logs" not in output_result.stdout and output_result.stdout.strip():
                    logs = output_result.stdout
            
            # More thorough completion check - look for actual completion indicators
            # Check for adapter files or completion messages in logs
            completion_indicators = False
            completion_details = []
            if logs:
                # Look for completion indicators in logs
                log_lower = logs.lower()
                if "training completed" in log_lower:
                    completion_indicators = True
                    completion_details.append("Found 'training completed' in logs")
                if "training finished" in log_lower:
                    completion_indicators = True
                    completion_details.append("Found 'training finished' in logs")
                if "saved checkpoint" in log_lower or "saving final checkpoint" in log_lower:
                    completion_indicators = True
                    completion_details.append("Found checkpoint save messages in logs")
                if "epoch" in log_lower and "loss" in log_lower:
                    completion_indicators = True
                    completion_details.append("Found training progress (epoch/loss) in logs")
                if "ready for download" in log_lower:
                    completion_indicators = True
                    completion_details.append("Found 'ready for download' message in logs")
                if "training complete" in log_lower:
                    completion_indicators = True
                    completion_details.append("Found 'training complete' in logs")
            
            # Check for actual adapter files in multiple locations
            check_adapter_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                """bash -c '
                    # Check multiple possible locations for adapter files
                    if [ -f /workspace/output/training/adapter/adapter_config.json ]; then
                        echo "adapter_exists:/workspace/output/training/adapter"
                    elif [ -f /workspace/output/adapter/adapter_config.json ]; then
                        echo "adapter_exists:/workspace/output/adapter"
                    elif [ -f /workspace/output/training/adapter_config.json ]; then
                        echo "adapter_exists:/workspace/output/training"
                    elif [ -f /workspace/output/adapter_config.json ]; then
                        echo "adapter_exists:/workspace/output"
                    elif find /workspace/output -name "adapter_config.json" -type f 2>/dev/null | head -1 | grep -q .; then
                        echo "adapter_exists:$(find /workspace/output -name adapter_config.json -type f 2>/dev/null | head -1)"
                    else
                        echo "no_adapter"
                    fi
                '"""
            ]
            adapter_result = subprocess.run(check_adapter_cmd, capture_output=True, text=True, timeout=15, shell=True)
            adapter_exists = "adapter_exists" in adapter_result.stdout
            adapter_location = None
            if adapter_exists:
                # Extract location from output
                for line in adapter_result.stdout.split('\n'):
                    if 'adapter_exists:' in line:
                        adapter_location = line.split('adapter_exists:')[1].strip()
                        break
            
            # Also check for any weight files (bin, safetensors, etc.)
            check_weights_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                """bash -c '
                    # Look for any weight files
                    find /workspace/output -type f \\( -name "*.bin" -o -name "*.safetensors" -o -name "adapter_model*" -o -name "pytorch_model*" \\) 2>/dev/null | head -5
                '"""
            ]
            weights_result = subprocess.run(check_weights_cmd, capture_output=True, text=True, timeout=15, shell=True)
            weight_files = []
            if weights_result.returncode == 0 and weights_result.stdout.strip():
                weight_files = [f.strip() for f in weights_result.stdout.strip().split('\n') if f.strip()]
            
            # Note: We no longer use onstart scripts - all training is started manually via SSH
            # Set these to default values since onstart checks are not applicable
            onstart_running = False
            onstart_process_info = None
            onstart_status = "not_applicable"
            onstart_logs = None
            
            # Check if training files were uploaded (check if they exist on instance)
            check_files_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                "test -f /workspace/data/training_data.jsonl && test -f /workspace/data/axolotl_config.yaml && echo 'files_exist' || echo 'no_files'"
            ]
            files_result = subprocess.run(check_files_cmd, capture_output=True, text=True, timeout=15)
            training_files_exist = "files_exist" in files_result.stdout
            
            # Determine status - be more thorough
            # Check multiple indicators of completion
            
            # If we have weight files, that's a strong indicator of completion
            has_weight_files = len(weight_files) > 0
            
            # Check if training actually started and then failed
            training_failed = False
            failure_reason = None
            training_started = False
            
            # Check if training ever started by looking for evidence
            check_training_started_cmd = [
                "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                f"root@{ssh_host}",
                """bash -c '
                    # Check for evidence that training started
                    # Look for checkpoint directories, training state files, etc.
                    if [ -d /workspace/output/training ]; then
                        # Check for any subdirectories (checkpoints, etc.)
                        find /workspace/output/training -type d -mindepth 1 2>/dev/null | head -5
                        # Check for any files
                        find /workspace/output/training -type f 2>/dev/null | head -10
                    fi
                    # Also check axolotl directory
                    if [ -d /workspace/axolotl ]; then
                        find /workspace/axolotl -name "*checkpoint*" -o -name "*training*" -type f 2>/dev/null | head -5
                    fi
                '"""
            ]
            training_started_result = subprocess.run(check_training_started_cmd, capture_output=True, text=True, timeout=15, shell=True)
            if training_started_result.returncode == 0 and training_started_result.stdout.strip():
                # If we found any files or directories in output, training likely started
                output_lines = [l.strip() for l in training_started_result.stdout.strip().split('\n') if l.strip()]
                training_started = len(output_lines) > 0
            
            if logs:
                log_lower = logs.lower()
                # Look for error indicators
                if any(err in log_lower for err in ["error", "failed", "exception", "traceback", "fatal", "crash"]):
                    training_failed = True
                    # Try to extract the error - get more context
                    error_lines = []
                    log_lines = logs.split('\n')
                    for i, line in enumerate(log_lines):
                        line_lower = line.lower()
                        if any(err in line_lower for err in ["error", "failed", "exception", "traceback", "fatal"]):
                            # Get some context around the error
                            start = max(0, i-2)
                            end = min(len(log_lines), i+3)
                            error_context = '\n'.join(log_lines[start:end])
                            error_lines.append(error_context)
                    if error_lines:
                        # Get the last error with context
                        failure_reason = error_lines[-1][:500]  # Last error with context, truncated
                
                # Also check if training actually started by looking for training-related messages
                # Include preprocessing activities (tokenizing, dropping sequences, etc.) as indicators of active training
                training_indicators = [
                    "starting training", "beginning training", "epoch", "step", "loss", 
                    "accelerate launch", "tokenizing", "preprocessing", "pre-process",
                    "dropping long sequences", "drop samples", "saving the dataset",
                    "loading dataset", "preparing dataset", "sample packing"
                ]
                if any(indicator in log_lower for indicator in training_indicators):
                    training_started = True
            
            # Check training status
            # But also check if preprocessing is happening (which is part of training)
            is_preprocessing = False
            if logs:
                log_lower = logs.lower()
                preprocessing_indicators = ["tokenizing", "preprocessing", "pre-process", "dropping long sequences", 
                                          "drop samples", "saving the dataset", "sample packing", "loading dataset"]
                is_preprocessing = any(indicator in log_lower for indicator in preprocessing_indicators)
            
            if training_running or is_preprocessing:
                # Training is actively running (either actual training or preprocessing)
                status = "training"
                if is_preprocessing and not training_running:
                    self.logger.log("INFO", f"Training preprocessing in progress on instance {instance_id}")
            # Only mark as completed if:
            # 1. Training process is NOT running
            # 2. AND (we have adapter files OR weight files OR completion indicators in logs)
            # 3. AND training didn't fail
            elif not training_running and (adapter_exists or has_weight_files or completion_indicators) and not training_failed:
                status = "completed"
                self.logger.log("SUCCESS", f"Training detected as completed on instance {instance_id}", {
                    "adapter_exists": adapter_exists,
                    "adapter_location": adapter_location,
                    "weight_files_count": len(weight_files),
                    "completion_indicators": completion_indicators,
                    "completion_details": completion_details,
                    "has_output": has_output
                })
            # This check is now handled earlier in the preprocessing detection above
            # Keeping for backward compatibility but it should already be set to "training" if preprocessing is detected
            elif training_failed:
                status = "failed"
                self.logger.log("ERROR", f"Training appears to have failed on instance {instance_id}", {
                    "failure_reason": failure_reason,
                    "has_output": has_output,
                    "logs_available": logs is not None and len(logs) > 0
                })
            elif has_output and not training_running:
                # Has output but no clear completion - provide detailed diagnostics
                # Check if training ever started by looking for any training-related files or logs
                if not training_started:
                    # Training never started
                    status = "failed"
                    failure_reason = "Training never started. Training process did not begin. Check if files were uploaded correctly and training logs."
                    self.logger.log("ERROR", f"Training never started on instance {instance_id}", {
                        "training_files_exist": training_files_exist,
                        "has_output": has_output
                    })
                else:
                    # Training started but unclear if it completed
                    status = "unknown"
                    self.logger.log("WARNING", f"Training process stopped but completion unclear on instance {instance_id}", {
                        "has_output": has_output,
                        "adapter_exists": adapter_exists,
                        "adapter_location": adapter_location,
                        "weight_files": weight_files[:5],  # Show first 5
                        "weight_files_count": len(weight_files),
                        "completion_indicators": completion_indicators,
                        "completion_details": completion_details,
                        "training_files_exist": training_files_exist,
                        "training_started": training_started,
                        "logs_available": logs is not None and len(logs) > 0,
                        "logs_length": len(logs) if logs else 0,
                        "training_failed": training_failed,
                        "failure_reason": failure_reason
                    })
            else:
                status = "unknown"
                self.logger.log("INFO", f"Training status unclear on instance {instance_id}", {
                    "training_running": training_running,
                    "has_output": has_output,
                    "adapter_exists": adapter_exists,
                    "weight_files_count": len(weight_files),
                    "training_files_exist": training_files_exist,
                    "onstart_running": onstart_running,
                    "onstart_status": onstart_status
                })
            
            return {
                "status": status,
                "training_running": training_running,
                "has_output": has_output,
                "adapter_exists": adapter_exists,
                "adapter_location": adapter_location,
                "weight_files": weight_files[:10],  # Include first 10 weight files
                "weight_files_count": len(weight_files),
                "completion_indicators": completion_indicators,
                "completion_details": completion_details,
                "training_files_exist": training_files_exist,
                "training_failed": training_failed,
                "training_started": training_started,
                "failure_reason": failure_reason,
                "logs": logs,
                "logs_length": len(logs) if logs else 0,
                "ssh_host": ssh_host,
                "ssh_port": ssh_port
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": "SSH connection timed out. This may mean SSH keys are not configured.",
                "status": "unknown",
                "ssh_required": True
            }
        except FileNotFoundError:
            return {
                "error": "SSH client not found. Please install OpenSSH client.",
                "status": "unknown",
                "ssh_required": True
            }
        except subprocess.CalledProcessError as e:
            # SSH authentication failed - likely missing SSH keys
            if "Permission denied" in str(e.stderr) or "publickey" in str(e.stderr).lower():
                return {
                    "error": "SSH authentication failed. SSH keys are required. See instructions below.",
                    "status": "unknown",
                    "ssh_required": True,
                    "ssh_host": ssh_host,
                    "ssh_port": ssh_port
                }
            return {
                "error": f"SSH command failed: {str(e.stderr)}",
                "status": "unknown",
                "ssh_required": True
            }
        except Exception as e:
            error_msg = str(e)
            if "Permission denied" in error_msg or "publickey" in error_msg.lower():
                return {
                    "error": "SSH authentication failed. SSH keys are required. See instructions below.",
                    "status": "unknown",
                    "ssh_required": True,
                    "ssh_host": ssh_host,
                    "ssh_port": ssh_port
                }
            return {
                "error": f"Error checking training status: {error_msg}",
                "status": "unknown",
                "ssh_required": True
            }

