"""Client for using fine-tuned models with transformers"""

from pathlib import Path
from typing import List, Dict, Optional
import torch
import os

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import PEFT for LoRA
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class FineTunedModelClient:
    """Client for interacting with fine-tuned models using transformers"""
    
    def __init__(self, model_path: str, device: str = None, hf_token: str = None):
        """
        Initialize fine-tuned model client
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to use ('cuda', 'cpu', or None for auto)
            hf_token: Optional Hugging Face token for gated models
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Please install: pip install transformers")
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Get Hugging Face token (prioritize passed token, then env, then cache)
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            try:
                from huggingface_hub import HfFolder
                hf_token = HfFolder.get_token()
            except:
                pass
        self.hf_token = hf_token
        
        # Check if this is a LoRA adapter or full model
        adapter_path = self.model_path.parent / "adapter"
        fine_tune_metadata_path = self.model_path.parent / "fine_tune_metadata.json"
        
        # Load tokenizer (always from model path)
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            token=self.hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if we have LoRA adapter
        use_adapter = False
        base_model_name = None
        
        if adapter_path.exists() and fine_tune_metadata_path.exists():
            try:
                import json
                with open(fine_tune_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get("use_lora", False) and PEFT_AVAILABLE:
                        use_adapter = True
                        base_model_name = metadata.get("hf_model_name")
                        print(f"Detected LoRA adapter. Base model: {base_model_name}")
            except:
                pass
        
        # Load model - either adapter + base or full model
        if use_adapter and base_model_name:
            print(f"Loading base model: {base_model_name}...")
            # Load base model first
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    token=self.hf_token
                )
            except OSError as e:
                error_msg = str(e)
                if "gated repo" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower():
                    raise OSError(
                        f"Authentication required for base model '{base_model_name}'. "
                        f"This is a gated model. Please provide a Hugging Face token. "
                        f"Get one at: https://huggingface.co/settings/tokens"
                    ) from e
                else:
                    raise
            
            print(f"Loading LoRA adapter from {adapter_path}...")
            # Load adapter on top of base model
            self.model = PeftModel.from_pretrained(
                base_model,
                str(adapter_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Merge adapter for faster inference (optional - can be removed if you want to keep them separate)
            # self.model = self.model.merge_and_unload()
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"LoRA adapter loaded successfully (much smaller than full model!)")
        else:
            # Load full model (backward compatibility or non-LoRA fine-tuning)
            print(f"Loading full model from {model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                token=self.hf_token
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
        
        self.model.eval()  # Set to evaluation mode
        print(f"Model loaded on {self.device}")
    
    def chat(self, messages: List[Dict[str, str]], max_length: int = 2048, temperature: float = 0.7) -> str:
        """
        Send chat messages to the fine-tuned model
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_length: Maximum length of generated response
            temperature: Sampling temperature (higher = more creative)
        
        Returns:
            Generated response text
        """
        if not messages:
            return "Error: No messages provided"
        
        # Format messages using chat template if available
        try:
            # Convert to format expected by chat template
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Apply chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback formatting
                prompt = self._format_messages_fallback(messages)
        except Exception as e:
            print(f"Error formatting messages: {e}")
            prompt = self._format_messages_fallback(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        # Find where the prompt ends and extract only the new generated text
        prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        response_ids = outputs[0][prompt_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Fallback: if extraction failed, try string replacement
        if not response and prompt in full_response:
            response = full_response.replace(prompt, "").strip()
        elif not response:
            response = full_response.strip()
        
        return response
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Fallback message formatting if chat template not available"""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        
        formatted += "Assistant: "
        return formatted
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the newly generated text
        prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        response_ids = outputs[0][prompt_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Fallback
        if not response and prompt in full_response:
            response = full_response.replace(prompt, "").strip()
        elif not response:
            response = full_response.strip()
        
        return response

