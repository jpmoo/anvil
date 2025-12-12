"""Fine-tuning utilities using Hugging Face transformers"""

from pathlib import Path
from utils.config import MODELS_DIR
from utils.training_data import TrainingDataManager
from utils.model_manager import ModelManager
import json
from datetime import datetime
import os

# Try to import transformers and related libraries
try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class FineTuner:
    """Handles fine-tuning of language models"""
    
    # Mapping from Ollama model names to Hugging Face model identifiers
    MODEL_MAPPING = {
        "phi3": "microsoft/phi-3-mini-4k-instruct",
        "phi3:mini": "microsoft/phi-3-mini-4k-instruct",
        "phi-3": "microsoft/phi-3-mini-4k-instruct",
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
        "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "mistral:latest": "mistralai/Mistral-7B-Instruct-v0.2",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }
    
    def __init__(self, model_name: str, base_model: str = "llama2"):
        self.model_name = model_name
        self.base_model = base_model
        self.data_manager = TrainingDataManager(model_name)
        self.models_dir = MODELS_DIR
        self.model_manager = ModelManager()
    
    def get_hf_model_name(self, ollama_model_name: str) -> str:
        """Map Ollama model name to Hugging Face model identifier"""
        # Remove tag if present (e.g., "phi3:mini" -> "phi3")
        base_name = ollama_model_name.split(':')[0] if ':' in ollama_model_name else ollama_model_name
        
        # Check mapping
        if base_name.lower() in self.MODEL_MAPPING:
            return self.MODEL_MAPPING[base_name.lower()]
        
        # Check with full name
        if ollama_model_name.lower() in self.MODEL_MAPPING:
            return self.MODEL_MAPPING[ollama_model_name.lower()]
        
        # If not found, try to construct a reasonable default
        # For unknown models, return None to indicate we can't proceed
        return None
    
    def prepare_dataset(self, tokenizer=None):
        """Prepare training dataset from all sources in instruction format"""
        training_data = self.data_manager.get_all_training_data()
        
        examples = []
        
        # Add all context files as instruction-following examples
        context_files = training_data["context_files"]
        for i, context_item in enumerate(context_files):
            # Diminishing reinforcement: repeat newer files more than older ones
            repetitions = max(1, 5 - i)
            text = context_item["text"]
            # Format as instruction: "Learn the following information: {text}"
            for _ in range(repetitions):
                examples.append({
                    "instruction": "Learn and remember the following information.",
                    "input": "",
                    "output": text
                })
        
        # Add behavioral rules as instructions
        behavioral_rules = training_data["behavioral_rules"]
        for behavior in behavioral_rules.get("behaviors", []):
            weight = behavior.get("weight", 1)
            description = behavior.get("description", "")
            for _ in range(weight):
                examples.append({
                    "instruction": "Follow this behavioral rule in all responses.",
                    "input": "",
                    "output": description
                })
        
        # Add learned pairs as Q&A format (instruction tuning format)
        learned_pairs = training_data["learned_pairs"]
        for pair in learned_pairs:
            examples.append({
                "instruction": "Answer the following question.",
                "input": pair["question"],
                "output": pair["answer"]
            })
        
        return examples
    
    def format_instruction(self, example, tokenizer=None):
        """Format a single example for instruction tuning"""
        # Try to use chat template if available
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            try:
                # Use chat template format
                messages = [
                    {"role": "system", "content": example["instruction"]},
                    {"role": "user", "content": example["input"] if example["input"] else "Continue."},
                    {"role": "assistant", "content": example["output"]}
                ]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return formatted
            except (AttributeError, KeyError, TypeError):
                # Fallback if chat template not available or fails
                pass
        
        # Fallback format (Alpaca-style)
        if example["input"]:
            return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    
    def fine_tune(
        self,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        epochs: int = 3,
        output_dir: str = None,
        use_lora: bool = True,
        hf_token: str = None,
        progress_callback=None
    ):
        """Fine-tune the model using Hugging Face transformers"""
        if not TRANSFORMERS_AVAILABLE:
            return {
                "success": False,
                "error": "Transformers library not available. Please install: pip install transformers datasets"
            }
        
        try:
            # Get Hugging Face model name
            hf_model_name = self.get_hf_model_name(self.base_model)
            if not hf_model_name:
                return {
                    "success": False,
                    "error": f"Unknown base model '{self.base_model}'. Supported models: {', '.join(self.MODEL_MAPPING.keys())}"
                }
            
            # Prepare dataset
            if progress_callback:
                progress_callback("📊 Preparing training dataset...", 0.05)
            
            examples = self.prepare_dataset()
            
            if progress_callback:
                progress_callback(f"✅ Dataset prepared: {len(examples)} training examples", 0.10)
            
            if not examples:
                return {"success": False, "error": "No training data available"}
            
            if progress_callback:
                progress_callback(f"📥 Loading model: {hf_model_name}...", 0.15)
            
            print(f"Loading model: {hf_model_name}")
            print(f"Training examples: {len(examples)}")
            
            # Check for Hugging Face token (prioritize passed token, then env, then cache)
            if not hf_token or not hf_token.strip():
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not hf_token or not hf_token.strip():
                # Try to get token from huggingface_hub cache
                try:
                    from huggingface_hub import HfFolder
                    hf_token = HfFolder.get_token()
                except:
                    pass
            
            # Clean token (remove whitespace)
            if hf_token:
                hf_token = hf_token.strip()
            
            # Debug output
            if hf_token:
                print(f"Using Hugging Face token: {hf_token[:8]}...")
            else:
                print("Warning: No Hugging Face token found")
            
            # Load tokenizer and model with authentication handling
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name,
                    token=hf_token
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with memory-efficient settings
                import torch
                # For MPS, models can be too large - use CPU instead
                # Determine device and dtype for memory efficiency
                use_mps = False  # Disable MPS for now due to memory constraints
                
                if torch.backends.mps.is_available() and use_mps:
                    # MPS (Apple Silicon) - use float32 for better compatibility and less memory
                    device_map = "cpu"  # Start on CPU, let trainer handle device placement
                    torch_dtype = torch.float32
                    print("Using MPS (Apple Silicon) - loading model in float32 on CPU initially")
                elif torch.cuda.is_available():
                    # CUDA - use auto dtype
                    device_map = "auto"
                    torch_dtype = "auto"
                    print("Using CUDA")
                else:
                    # CPU (also use for MPS to avoid memory issues)
                    device_map = None
                    torch_dtype = torch.float32
                    if torch.backends.mps.is_available():
                        print("⚠️ MPS available but using CPU to avoid memory issues")
                    else:
                        print("Using CPU")
                
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    token=hf_token,
                    low_cpu_mem_usage=True  # More memory efficient loading
                )
                
                if progress_callback:
                    progress_callback(f"✅ Model loaded: {hf_model_name}", 0.25)
            except OSError as e:
                error_msg = str(e)
                if "gated repo" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower() or "access" in error_msg.lower():
                    # Check if token was provided
                    token_status = "Token was provided" if hf_token else "No token provided"
                    return {
                        "success": False,
                        "error": f"Authentication failed for model '{hf_model_name}'. {token_status}.",
                        "solution": f"""This is a gated model that requires:
1. **A valid Hugging Face token** (you can enter it above)
2. **Accepting the model's terms** on Hugging Face

**Steps to fix:**
1. Visit the model page: https://huggingface.co/{hf_model_name.replace('/', '/')}
2. Click "Agree and access repository" to accept the terms
3. Get your token: https://huggingface.co/settings/tokens
4. Paste the token in the field above
5. Try fine-tuning again""",
                        "model_name": hf_model_name,
                        "is_auth_error": True,
                        "token_provided": bool(hf_token)
                    }
                else:
                    raise
            except Exception as e:
                error_msg = str(e)
                if "gated" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower():
                    token_status = "Token was provided" if hf_token else "No token provided"
                    return {
                        "success": False,
                        "error": f"Authentication failed for model '{hf_model_name}'. {token_status}.",
                        "solution": f"""This is a gated model that requires:
1. **A valid Hugging Face token** (you can enter it above)
2. **Accepting the model's terms** on Hugging Face

**Steps to fix:**
1. Visit the model page: https://huggingface.co/{hf_model_name.replace('/', '/')}
2. Click "Agree and access repository" to accept the terms
3. Get your token: https://huggingface.co/settings/tokens
4. Paste the token in the field above
5. Try fine-tuning again""",
                        "model_name": hf_model_name,
                        "is_auth_error": True,
                        "token_provided": bool(hf_token)
                    }
                else:
                    raise
            
            # Apply LoRA if available and requested
            if use_lora and PEFT_AVAILABLE:
                if progress_callback:
                    progress_callback("🔧 Applying LoRA adapter for efficient fine-tuning...", 0.30)
                print("Applying LoRA for efficient fine-tuning...")
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] if "phi" in hf_model_name.lower() 
                    else ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            
            # Format examples for training
            if progress_callback:
                progress_callback("📝 Formatting training examples...", 0.35)
            
            print("Formatting training data...")
            formatted_texts = []
            total_examples = len(examples)
            for i, example in enumerate(examples):
                formatted = self.format_instruction(example, tokenizer)
                formatted_texts.append(formatted)
                if progress_callback and (i + 1) % max(1, total_examples // 10) == 0:
                    progress_callback(f"📝 Formatting examples: {i + 1}/{total_examples}", 0.35 + (i / total_examples) * 0.05)
            
            # Create dataset
            if progress_callback:
                progress_callback("🔨 Creating dataset...", 0.40)
            
            dataset = Dataset.from_dict({"text": formatted_texts})
            
            # Tokenize - use batched=True for efficiency
            # Reduce max_length to save memory (1024 is usually sufficient for fine-tuning)
            def tokenize_function(examples):
                # When batched=True, examples["text"] is a list of strings
                # Tokenize all texts at once
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=1024,  # Reduced from 2048 to save memory
                    padding=False  # Don't pad here - data collator will handle it
                )
                # Don't set labels here - DataCollatorForLanguageModeling will handle it
                # Setting labels manually can cause nesting issues
                return tokenized
            
            # Tokenize dataset - use batched=True for efficiency
            # The data collator will handle padding sequences to the same length
            if progress_callback:
                progress_callback("🔤 Tokenizing dataset (this may take a moment)...", 0.45)
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,  # Process in batches for efficiency
                remove_columns=["text"],
                desc="Tokenizing dataset"
            )
            
            if progress_callback:
                progress_callback(f"✅ Tokenization complete: {len(tokenized_dataset)} examples", 0.50)
            
            # Verify dataset structure
            print(f"Dataset size: {len(tokenized_dataset)}")
            if len(tokenized_dataset) > 0:
                sample = tokenized_dataset[0]
                print(f"Sample keys: {sample.keys()}")
                if "input_ids" in sample:
                    print(f"Sample input_ids type: {type(sample['input_ids'])}, length: {len(sample['input_ids'])}")
                    if len(sample['input_ids']) > 0:
                        print(f"First input_id type: {type(sample['input_ids'][0])}, value: {sample['input_ids'][0]}")
                # Note: labels will be set by the data collator, so we don't check for them here
            
            # Set up output directory - always use the same location (no versioning)
            model_dir = self.models_dir / self.model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Always save to the same location
            output_path = str(model_dir / "checkpoint")
            
            # Training arguments with memory optimizations
            import torch
            # Force CPU for training to avoid MPS memory issues
            # MPS has limited memory and models can be too large
            device_type = "cpu"  # Always use CPU for training to avoid memory issues
            if torch.cuda.is_available():
                device_type = "cuda"
                print("Using CUDA for training")
            else:
                print("Using CPU for training (MPS disabled to avoid memory issues)")
            
            # Adjust batch size and gradient accumulation for memory constraints
            # Use smaller batches for CPU training
            effective_batch_size = min(batch_size, 2)  # Max 2 for CPU training
            gradient_accumulation = max(1, batch_size // effective_batch_size)  # Accumulate to maintain effective batch size
            
            if effective_batch_size < batch_size:
                print(f"⚠️ Reduced batch size to {effective_batch_size} with gradient accumulation {gradient_accumulation} for memory efficiency")
            
            training_args = TrainingArguments(
                output_dir=output_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=effective_batch_size,
                learning_rate=learning_rate,
                logging_dir=str(model_dir / "logs"),
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=1,  # Only keep latest
                warmup_steps=min(10, len(tokenized_dataset) // max(effective_batch_size, 1)),
                fp16=False,  # Disable fp16 for CPU training
                bf16=False,  # bf16 not widely supported yet
                gradient_accumulation_steps=gradient_accumulation,
                report_to="none",  # Disable wandb/tensorboard for simplicity
                remove_unused_columns=False,
                dataloader_pin_memory=False,  # Disable pinning for CPU
                max_grad_norm=1.0,  # Gradient clipping to help with stability
                no_cuda=(device_type == "cpu"),  # Explicitly disable CUDA if using CPU
            )
            
            # Data collator with padding enabled
            # This will automatically pad sequences to the same length in each batch
            # Make sure padding token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM, not masked LM
                pad_to_multiple_of=8  # Pad to multiple of 8 for efficiency (optional, helps with GPU)
            )
            
            # Ensure model is on CPU if we're using CPU training
            import torch
            if device_type == "cpu":
                # Explicitly move model to CPU and disable MPS
                model = model.to("cpu")
                # Disable MPS backend to prevent accidental usage
                if torch.backends.mps.is_available():
                    import os
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
                    print("⚠️ MPS disabled - using CPU for training to avoid memory issues")
            
            # Train
            if progress_callback:
                progress_callback(f"🚀 Starting training: {epochs} epochs, batch size {effective_batch_size}...", 0.55)
            
            print("Starting training...")
            
            # Add callback if available
            from transformers import TrainerCallback
            class ProgressCallback(TrainerCallback):
                def __init__(self, progress_callback, total_steps, epochs):
                    self.progress_callback = progress_callback
                    self.total_steps = total_steps
                    self.epochs = epochs
                    self.start_progress = 0.55
                    self.training_progress_range = 0.35  # 55% to 90% for training
                
                def on_epoch_begin(self, args, state, control, model=None, **kwargs):
                    if self.progress_callback:
                        progress = self.start_progress + (state.epoch / self.epochs) * self.training_progress_range
                        self.progress_callback(f"📈 Epoch {int(state.epoch) + 1}/{self.epochs} starting...", progress)
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if self.progress_callback and logs:
                        loss = logs.get('loss', 'N/A')
                        step = state.global_step
                        epoch_progress = state.epoch / self.epochs
                        step_progress = (step % self.total_steps) / self.total_steps if self.total_steps > 0 else 0
                        overall_progress = self.start_progress + (epoch_progress + step_progress / self.epochs) * self.training_progress_range
                        status = f"📈 Epoch {int(state.epoch) + 1}/{self.epochs}, Step {step}, Loss: {loss:.4f}"
                        self.progress_callback(status, min(overall_progress, 0.90))
            
            callbacks = []
            if progress_callback:
                steps_per_epoch = len(tokenized_dataset) // (effective_batch_size * gradient_accumulation)
                total_steps = steps_per_epoch * epochs
                callbacks.append(ProgressCallback(progress_callback, total_steps, epochs))
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                callbacks=callbacks
            )
            
            trainer.train()
            
            if progress_callback:
                progress_callback("✅ Training complete!", 0.90)
            
            # Save model - use efficient LoRA adapter saving if available
            if progress_callback:
                progress_callback("💾 Saving model...", 0.92)
            
            print("Saving model...")
            final_model_path = model_dir / "model"
            adapter_used = False
            
            # If using LoRA, save only the adapter weights (much smaller)
            if use_lora and PEFT_AVAILABLE:
                # Check if model is a PEFT model
                try:
                    from peft import PeftModel
                    if isinstance(model, PeftModel):
                        # Save only the adapter weights (much smaller - typically a few MB vs GB)
                        adapter_path = model_dir / "adapter"
                        adapter_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(str(adapter_path))
                        tokenizer.save_pretrained(str(final_model_path))
                        adapter_used = True
                        adapter_size = sum(f.stat().st_size for f in adapter_path.rglob('*') if f.is_file()) / (1024*1024)
                        print(f"✅ Saved LoRA adapter to {adapter_path} (much smaller than full model!)")
                        print(f"   Adapter size: ~{adapter_size:.2f} MB")
                        if progress_callback:
                            progress_callback(f"✅ Saved LoRA adapter (~{adapter_size:.2f} MB)", 0.96)
                    else:
                        # Not a PEFT model, save full model
                        trainer.save_model(str(final_model_path))
                        tokenizer.save_pretrained(str(final_model_path))
                        adapter_used = False
                        print("Saved full model (LoRA was not applied)")
                except Exception as e:
                    # Fallback to full model save
                    print(f"Warning: Could not save as LoRA adapter: {e}")
                    print("Falling back to full model save...")
                    trainer.save_model(str(final_model_path))
                    tokenizer.save_pretrained(str(final_model_path))
                    adapter_used = False
            else:
                # Not using LoRA, save full model
                trainer.save_model(str(final_model_path))
                tokenizer.save_pretrained(str(final_model_path))
                adapter_used = False
                print("Saved full model (LoRA not available or disabled)")
            
            # Update training count in metadata (but don't track versions)
            model_metadata = self.model_manager.get_model_metadata(self.model_name)
            if model_metadata:
                # Increment training count
                training_count = model_metadata.get("training_count", 0) + 1
                model_metadata["training_count"] = training_count
                model_metadata["last_training_date"] = datetime.now().isoformat()
                
                model_dir_meta = self.models_dir / self.model_name / "metadata.json"
                with open(model_dir_meta, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
            
            # Save fine-tuning metadata
            metadata = {
                "training_count": model_metadata.get("training_count", 1) if model_metadata else 1,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "dataset_size": len(examples),
                "base_model": self.base_model,
                "hf_model_name": hf_model_name,
                "use_lora": adapter_used,
                "adapter_path": str(model_dir / "adapter") if adapter_used else None,
                "date": datetime.now().isoformat()
            }
            
            metadata_path = model_dir / "fine_tune_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "success": True,
                "output_dir": str(model_dir),
                "model_path": str(final_model_path),
                "dataset_size": len(examples),
                "training_count": model_metadata.get("training_count", 1) if model_metadata else 1,
                "message": f"Fine-tuning completed! Model saved to {final_model_path}"
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error during fine-tuning: {error_trace}")
            return {"success": False, "error": str(e), "traceback": error_trace}
    

