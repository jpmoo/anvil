"""Prepare training data for Axolotl format"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from utils.training_data import TrainingDataManager
from utils.config import get_model_queue_dir

# Model mapping from Ollama names to Hugging Face identifiers
MODEL_MAPPING = {
    # LLaMA models
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama2:7b": "meta-llama/Llama-2-7b-hf",
    "llama2:13b": "meta-llama/Llama-2-13b-hf",
    "llama2:70b": "meta-llama/Llama-2-70b-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3:8b": "meta-llama/Meta-Llama-3-8B",
    "llama3:70b": "meta-llama/Meta-Llama-3-70B",
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    
    # Mistral models
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral:7b": "mistralai/Mistral-7B-v0.1",
    "mistral:latest": "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral:8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    
    # Phi models
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi3:small": "microsoft/Phi-3-small-8k-instruct",
    "phi3:medium": "microsoft/Phi-3-medium-4k-instruct",
    "phi": "microsoft/phi-2",
    "phi-2": "microsoft/phi-2",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    
    # CodeLlama
    "codellama": "codellama/CodeLlama-7b-hf",
    "codellama:7b": "codellama/CodeLlama-7b-hf",
    "codellama:13b": "codellama/CodeLlama-13b-hf",
    "codellama:34b": "codellama/CodeLlama-34b-hf",
    
    # Gemma
    "gemma": "google/gemma-7b",
    "gemma:7b": "google/gemma-7b",
    "gemma:2b": "google/gemma-2b",
    # Gemma 2 models
    "gemma2": "google/gemma-2-2b-it",
    "gemma2:2b": "google/gemma-2-2b-it",
    "gemma2:9b": "google/gemma-2-9b-it",
    "gemma2:27b": "google/gemma-2-27b-it",
    # Gemma 3 models (4B parameter model)
    "gemma3": "google/gemma-3-4b-it",
    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma-3": "google/gemma-3-4b-it",
    "gemma-3:4b": "google/gemma-3-4b-it",
    
    # Qwen
    "qwen": "Qwen/Qwen-7B-Chat",
    "qwen:7b": "Qwen/Qwen-7B-Chat",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "qwen2:7b": "Qwen/Qwen2-7B-Instruct",
}


class AxolotlDataPrep:
    """Prepare training data in Axolotl-compatible format"""
    
    def __init__(self, model_name: str):
        """
        Initialize Axolotl data preparer
        
        Args:
            model_name: Name of the model profile
        """
        self.model_name = model_name
        self.data_manager = TrainingDataManager(model_name)
    
    def prepare_training_dataset(self, output_path: Path, file_group: Optional[List] = None) -> Dict:
        """
        Prepare all training data in Axolotl JSONL format
        
        Args:
            output_path: Path to save the JSONL file
            file_group: Optional list of file metadata dicts to process (if None, processes all files)
        
        Returns:
            Dictionary with dataset statistics
        """
        examples = []
        
        # Process queued files (JSON, JSONL, TXT)
        queue_dir = get_model_queue_dir(self.model_name)
        queued_files = []
        
        # If file_group is provided, only process those files
        if file_group:
            allowed_filenames = {f.get("filename") for f in file_group if f.get("filename")}
        else:
            allowed_filenames = None  # Process all files
        
        if queue_dir.exists():
            # Process JSON files
            for json_file in queue_dir.glob("*.json"):
                # Skip metadata files
                if json_file.name.endswith("_metadata.json"):
                    continue
                # Skip YAML files
                if json_file.suffix.lower() in ['.yaml', '.yml']:
                    continue
                # If file_group specified, only process allowed files
                if allowed_filenames and json_file.name not in allowed_filenames:
                    continue
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        # If it's already in the correct format (instruction/input/output), use it directly
                        if isinstance(json_data, dict) and "instruction" in json_data:
                            examples.append(json_data)
                        elif isinstance(json_data, list):
                            # If it's a list of examples, add each one
                            for item in json_data:
                                if isinstance(item, dict) and "instruction" in item:
                                    examples.append(item)
                        else:
                            # Otherwise, format as instruction-following example
                            examples.append({
                                "instruction": "Learn and remember the following information.",
                                "input": "",
                                "output": json.dumps(json_data, ensure_ascii=False)[:5000]
                            })
                        queued_files.append(json_file.name)
                except Exception as e:
                    print(f"Error processing queued file {json_file.name}: {e}")
            
            # Process JSONL files
            for jsonl_file in queue_dir.glob("*.jsonl"):
                # If file_group specified, only process allowed files
                if allowed_filenames and jsonl_file.name not in allowed_filenames:
                    continue
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                json_data = json.loads(line)
                                # If it's already in the correct format, use it directly
                                if isinstance(json_data, dict) and "instruction" in json_data:
                                    examples.append(json_data)
                                else:
                                    # Format as instruction-following example
                                    examples.append({
                                        "instruction": "Learn and remember the following information.",
                                        "input": "",
                                        "output": json.dumps(json_data, ensure_ascii=False)[:5000]
                                    })
                            except json.JSONDecodeError as e:
                                print(f"Error processing JSONL line {line_num} in {jsonl_file.name}: {e}")
                    queued_files.append(jsonl_file.name)
                except Exception as e:
                    print(f"Error processing queued file {jsonl_file.name}: {e}")
            
            # Process TXT files
            for txt_file in queue_dir.glob("*.txt"):
                # If file_group specified, only process allowed files
                if allowed_filenames and txt_file.name not in allowed_filenames:
                    continue
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                        if text_content.strip():
                            # Format as instruction-following example
                            examples.append({
                                "instruction": "Learn and remember the following information.",
                                "input": "",
                                "output": text_content[:5000]  # Limit length
                            })
                    queued_files.append(txt_file.name)
                except Exception as e:
                    print(f"Error processing queued file {txt_file.name}: {e}")
        
        # Process Q&A pairs from learning sessions
        learned_pairs = self.data_manager.get_learned_pairs()
        for pair in learned_pairs:
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            if question and answer:
                examples.append({
                    "instruction": question,
                    "input": "",
                    "output": answer
                })
        
        # Write to JSONL file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        return {
            "total_examples": len(examples),
            "queued_files": len(queued_files),
            "learned_pairs": len(learned_pairs),
            "output_file": str(output_path)
        }
    
    def get_hf_model_name(self, ollama_model_name: str) -> Optional[str]:
        """
        Map Ollama model name to Hugging Face model identifier, or return HuggingFace ID if already provided
        
        Args:
            ollama_model_name: Ollama model name (e.g., "llama2", "phi3:mini") or HuggingFace ID (e.g., "google/gemma-3-4b-it")
        
        Returns:
            Hugging Face model identifier or None if not found
        """
        if not ollama_model_name:
            return None
        
        # Check if input is already a HuggingFace model ID (contains "/")
        # This handles new profiles that use HuggingFace IDs directly
        if "/" in ollama_model_name:
            # Already a HuggingFace ID, return as-is
            return ollama_model_name.strip()
        
        # Otherwise, try to map from Ollama name (backward compatibility)
        normalized = ollama_model_name.strip().lower()
        
        # Try exact match first (with or without tag)
        if normalized in MODEL_MAPPING:
            return MODEL_MAPPING[normalized]
        
        # Remove tag if present (e.g., "phi3:mini" -> "phi3")
        base_name = normalized.split(':')[0] if ':' in normalized else normalized
        
        # Try base name
        if base_name in MODEL_MAPPING:
            return MODEL_MAPPING[base_name]
        
        # Try with common variations
        variations = [
            base_name,
            base_name.replace("-", ""),
            base_name.replace("_", ""),
            base_name.replace("-", "_"),
        ]
        
        for var in variations:
            if var in MODEL_MAPPING:
                return MODEL_MAPPING[var]
        
        return None
    
    def create_axolotl_config(self, 
                              base_model: str,
                              dataset_path: str,
                              output_dir: str,
                              lora_r: int = 8,
                              lora_alpha: int = 16,
                              lora_dropout: float = 0.05,
                              learning_rate: float = 2e-4,
                              num_epochs: int = 10,
                              batch_size: int = 4,
                              gradient_accumulation_steps: int = 4,
                              max_steps: Optional[int] = None,
                              train_on_inputs: bool = True,  # Set to True to maximize sample retention
                              output_path: Optional[Path] = None,
                              previous_adapter_path: Optional[str] = None) -> Dict:
        """
        Create Axolotl configuration YAML
        
        Args:
            base_model: Hugging Face model identifier
            dataset_path: Path to training dataset JSONL file
            output_dir: Directory to save trained model
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            output_path: Path to save config file (optional)
        
        Returns:
            Configuration dictionary
        """
        # Determine model type based on the base model
        # Default to Llama types, but override for specific model families
        # Note: tokenizer_type is removed - Axolotl will auto-detect it
        model_type = "LlamaForCausalLM"
        use_model_type = True  # Flag to control whether to include model_type in config
        
        base_model_lower = base_model.lower()
        if "gemma" in base_model_lower:
            # Gemma 3 uses a different architecture - let Axolotl auto-detect
            if "gemma-3" in base_model_lower or "gemma3" in base_model_lower:
                # Don't set model_type for Gemma 3 - let transformers auto-detect
                use_model_type = False
            else:
                # Gemma 1/2 use GemmaForCausalLM
                model_type = "GemmaForCausalLM"
        elif "mistral" in base_model_lower or "mixtral" in base_model_lower:
            model_type = "MistralForCausalLM"
        elif "phi" in base_model_lower:
            model_type = "PhiForCausalLM"
        elif "qwen" in base_model_lower:
            model_type = "Qwen2ForCausalLM"
        
        # Axolotl validation requires at least TWO of: micro_batch_size, gradient_accumulation_steps, batch_size
        # We set both micro_batch_size and gradient_accumulation_steps to satisfy this requirement
        
        config = {
            "base_model": base_model,
            "base_model_config": base_model,
            "trust_remote_code": True,
            "load_in_8bit": False,
            "load_in_4bit": True,  # Enable 4-bit quantization to reduce memory usage
            # Note: Setting adapter to a path loads an existing adapter, but for new LoRA training,
            # we should NOT set adapter (or set it to empty) - Axolotl will create adapters from lora_* parameters
            # However, some Axolotl versions need explicit adapter setting to save adapters separately
            "strict": False,
            
            "datasets": [
                {
                    "path": dataset_path,
                    "type": "alpaca"
                }
            ],
            
            "dataset_preparation_path": "/workspace/axolotl/prepared_data",
            
            "val_set_size": 0.1,
            "output_dir": output_dir,
            
            "sequence_len": 2048,
            "sample_packing": False,  # Disabled by default to prevent multipack sampler errors
            # sample_packing can cause IndexError with small datasets or certain batch configurations
            # Enable only if you have a large dataset (>1000 samples) and understand the risks
            "pad_to_sequence_len": True,
            
            # lora_model_dir: Only set when loading a previous adapter (set below if previous_adapter_path exists)
            # For new training, don't set lora_model_dir to output_dir as it causes Axolotl to try loading from there
            "lora_out_dir": f"{output_dir}/adapter",  # Explicitly set adapter output directory
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            
            "wandb_project": "",
            "wandb_entity": "",
            "wandb_watch": "",
            "wandb_name": "",
            "wandb_log_model": "",
            
            # Axolotl validation requires at least TWO of: micro_batch_size, gradient_accumulation_steps, batch_size
            # We set both micro_batch_size and gradient_accumulation_steps to satisfy this requirement
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_epochs": num_epochs,
            "optimizer": "adamw_torch",
            "lr_scheduler": "cosine",
            "learning_rate": learning_rate,
            "warmup_steps": 100,
            "train_on_inputs": train_on_inputs,  # True = maximize retention (learns from prompts), False = focus on responses only
            "group_by_length": True,
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 3,
            "save_strategy": "steps",  # Save at specific steps
            "save_safetensors": True,  # Ensures LoRA weights are saved in .safetensors format
            # LoRA adapter saving settings - prevent merging and ensure adapters are saved separately
            # These settings work for both new training and incremental training (loading previous adapters)
            "merge_lora": False,  # Critical: prevents merging LoRA into base model during save
            # Note: merge_lora: false does NOT prevent loading previous adapters for incremental training
            # It only prevents merging adapters into the base model when saving checkpoints
            "save_merged_lora": False,  # Don't save merged copies
            "lora_apply_dir": None,  # Prevent auto-merging on load (doesn't prevent loading adapters)
            "ddp_timeout": 180000000,
            "dataloader_num_workers": 4
        }
        
        # Only add model_type if we want to explicitly set it
        # For Gemma 3, we let Axolotl/transformers auto-detect the correct model type
        if use_model_type:
            config["model_type"] = model_type
        
        # Load previous adapter if provided (for incremental training V2+)
        # When previous_adapter_path is set, Axolotl will:
        # 1. Load the adapter from that path
        # 2. Continue training from those weights (incremental/cumulative training)
        # 3. Save new adapters separately (due to merge_lora: false)
        # The merge_lora: false setting does NOT interfere with loading - it only prevents merging during save
        if previous_adapter_path:
            config["adapter"] = previous_adapter_path
            # Also set lora_model_dir to the previous adapter path for explicit incremental training
            # This ensures Axolotl knows to load and continue from this adapter
            config["lora_model_dir"] = previous_adapter_path
        else:
            # For new LoRA training: Set adapter: "lora" to explicitly enable LoRA mode
            # Axolotl requires adapter: "lora" to enable LoRA, not just lora_* parameters
            # We do NOT set lora_model_dir to output_dir (that causes path loading errors)
            # The lora_out_dir parameter ensures adapters are saved to output_dir/adapter/
            config["adapter"] = "lora"
        
        if output_path:
            import yaml
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return config
    
    def get_training_data_summary(self) -> Dict:
        """
        Get summary of available training data
        
        Returns:
            Dictionary with data statistics
        """
        # Count queued files (JSON, JSONL, TXT)
        queue_dir = get_model_queue_dir(self.model_name)
        queued_files = []
        total_queue_size = 0
        if queue_dir.exists():
            # Count JSON files
            for json_file in queue_dir.glob("*.json"):
                if json_file.name.endswith("_metadata.json"):
                    continue
                queued_files.append(json_file.name)
                total_queue_size += json_file.stat().st_size
            # Count JSONL files
            for jsonl_file in queue_dir.glob("*.jsonl"):
                queued_files.append(jsonl_file.name)
                total_queue_size += jsonl_file.stat().st_size
            # Count TXT files
            for txt_file in queue_dir.glob("*.txt"):
                queued_files.append(txt_file.name)
                total_queue_size += txt_file.stat().st_size
        
        # Count learned pairs
        learned_pairs = self.data_manager.get_learned_pairs()
        total_qa_chars = sum(
            len(p.get("question", "")) + len(p.get("answer", "")) 
            for p in learned_pairs
        )
        
        return {
            "queued_json_files_count": len(queued_files),
            "queued_json_files_total_size": total_queue_size,
            "learned_pairs_count": len(learned_pairs),
            "learned_pairs_total_chars": total_qa_chars,
            "total_training_examples": len(queued_files) + len(learned_pairs),
            "has_data": len(queued_files) > 0 or len(learned_pairs) > 0
        }

