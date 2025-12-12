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
        Map Ollama model name to Hugging Face model identifier
        
        Args:
            ollama_model_name: Ollama model name (e.g., "llama2", "phi3:mini")
        
        Returns:
            Hugging Face model identifier or None if not found
        """
        if not ollama_model_name:
            return None
        
        # Normalize the input
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
                              num_epochs: int = 3,
                              batch_size: int = 4,
                              gradient_accumulation_steps: int = 4,
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
        config = {
            "base_model": base_model,
            "base_model_config": base_model,
            "model_type": "LlamaForCausalLM",
            "tokenizer_type": "LlamaTokenizer",
            "load_in_8bit": False,
            "load_in_4bit": False,
            "strict": False,
            
            "datasets": [
                {
                    "path": dataset_path,
                    "type": "alpaca"
                }
            ],
            
            "dataset_preparation_path": "./prepared_data",
            
            "val_set_size": 0.1,
            "output_dir": output_dir,
            
            "sequence_len": 2048,
            "sample_packing": True,
            "pad_to_sequence_len": True,
            
            "lora_model_dir": output_dir,
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
            
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "micro_batch_size": batch_size,
            "num_epochs": num_epochs,
            "optimizer": "adamw_torch",
            "lr_scheduler": "cosine",
            "learning_rate": learning_rate,
            "warmup_steps": 100,
            "train_on_inputs": False,
            "group_by_length": True,
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 3,
            "ddp_timeout": 180000000,
            "dataloader_num_workers": 4
        }
        
        # Load previous adapter if provided (for incremental training V2+)
        # Axolotl will load this adapter and continue training from it
        if previous_adapter_path:
            config["adapter"] = previous_adapter_path
        
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

