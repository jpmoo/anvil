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
    
    def __init__(self, model_path: str = None, device: str = None, hf_token: str = None, 
                 local_files_only: bool = False, remote_url: str = None, remote_model: str = "v1"):
        """
        Initialize fine-tuned model client
        
        Args:
            model_path: Path to the fine-tuned model directory (NOT USED - remote-only mode)
            device: Device to use (NOT USED - remote-only mode)
            hf_token: Optional Hugging Face token (NOT USED - remote-only mode)
            local_files_only: If True, only use cached models (NOT USED - remote-only mode)
            remote_url: REQUIRED - URL of remote inference server (e.g., "http://instance-ip:8000")
            remote_model: Model name to use on remote server ("base", "v1", "v2", etc.)
        """
        # REMOTE-ONLY MODE: This application requires remote inference server
        if not remote_url:
            error_msg = (
                "Remote inference server is required. This application is configured for remote-only mode.\n"
                "Please provide remote_url pointing to your Vast.ai inference server.\n"
                "Set up the inference server in Phase 4 of the training workflow."
            )
            print(f"[CLIENT] ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        self.remote_url = remote_url
        self.remote_model = remote_model
        self.mode = "remote"
        
        print(f"[CLIENT] ========================================")
        print(f"[CLIENT] Initializing REMOTE-ONLY client (TGI API)")
        print(f"[CLIENT] Remote URL: {remote_url}")
        print(f"[CLIENT] Remote model: {remote_model}")
        print(f"[CLIENT] API Type: Hugging Face Text Generation Inference (TGI)")
        print(f"[CLIENT] Endpoint: /v1/chat/completions (OpenAI compatible)")
        print(f"[CLIENT] Local mode is DISABLED - all inference will run on Vast.ai instance")
        print(f"[CLIENT] ========================================")
        
        # Remote mode - no local model loading needed
        return
        
        # Local mode is disabled - code below should never execute
        # Keeping for reference but raising error if somehow reached
        raise RuntimeError(
            "Local mode is disabled. This application requires a remote inference server.\n"
            "Please set up the inference server on your Vast.ai instance in Phase 4."
        )
        
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
        self.local_files_only = local_files_only
        
        # Check if this is a LoRA adapter or full model
        # adapter_path should be in the model_path directory, not parent
        adapter_path = self.model_path / "adapter"
        fine_tune_metadata_path = self.model_path / "fine_tune_metadata.json"
        
        # Also check if adapter files are directly in model_path (backward compatibility)
        has_adapter_files = (adapter_path.exists() and any(adapter_path.iterdir())) or \
                           any(self.model_path.glob("adapter_model.bin")) or \
                           any(self.model_path.glob("adapter_model.safetensors"))
        
        # Check if we have LoRA adapter first (before loading tokenizer)
        use_adapter = False
        base_model_name = None
        
        if has_adapter_files:
            # First try to get base model from fine_tune_metadata.json
            if fine_tune_metadata_path.exists():
                try:
                    import json
                    with open(fine_tune_metadata_path, 'r') as f:
                        content = f.read().strip()
                        if content:  # Check if file is not empty
                            metadata = json.loads(content)
                            if metadata.get("use_lora", False) and PEFT_AVAILABLE:
                                use_adapter = True
                                base_model_name = metadata.get("hf_model_name")
                                print(f"Detected LoRA adapter from metadata. Base model: {base_model_name}")
                except Exception as e:
                    print(f"Warning: Could not read fine_tune_metadata.json: {e}")
            
            # Fallback: try to get base model from adapter_config.json
            if not base_model_name and adapter_path.exists():
                adapter_config_path = adapter_path / "adapter_config.json"
                if adapter_config_path.exists():
                    try:
                        import json
                        with open(adapter_config_path, 'r') as f:
                            adapter_config = json.load(f)
                            base_model_name = adapter_config.get("base_model_name_or_path")
                            if base_model_name and PEFT_AVAILABLE:
                                use_adapter = True
                                print(f"Detected LoRA adapter from adapter_config.json. Base model: {base_model_name}")
                    except Exception as e:
                        print(f"Warning: Could not read adapter_config.json: {e}")
        
        # Load tokenizer - from base model if using LoRA, otherwise from model path
        if use_adapter and base_model_name:
            print(f"Loading tokenizer from base model: {base_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                token=self.hf_token
            )
        elif has_adapter_files and not base_model_name:
            # We have adapter files but couldn't determine base model
            raise ValueError(
                f"Found LoRA adapter files in {self.model_path} but could not determine base model. "
                f"Please ensure fine_tune_metadata.json or adapter/adapter_config.json contains base_model_name_or_path."
            )
        else:
            # Try to load from model path (for full models, not adapters)
            print(f"Loading tokenizer from {model_path}...")
            # Check if this looks like it might be an adapter directory
            if (self.model_path / "adapter").exists():
                raise ValueError(
                    f"Found adapter directory at {self.model_path / 'adapter'} but could not load as LoRA adapter. "
                    f"Please check that adapter_config.json or fine_tune_metadata.json is properly configured."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                token=self.hf_token
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model - either adapter + base or full model
        if use_adapter and base_model_name:
            # Check if user has the model in Ollama (can't use it, but helpful to mention)
            has_ollama_model = False
            ollama_model_name = None
            try:
                from utils.ollama_client import OllamaClient
                # Try to map HF model name to Ollama name
                hf_to_ollama = {
                    "google/gemma-3-4b-it": "gemma3:4b",
                    "google/gemma-3-4b": "gemma3:4b",
                }
                ollama_model_name = hf_to_ollama.get(base_model_name)
                if ollama_model_name:
                    ollama_client = OllamaClient(ollama_model_name)
                    has_ollama_model = ollama_client.model_exists(ollama_model_name)
            except:
                pass
            
            # Check if model is cached locally in HuggingFace cache
            try:
                from huggingface_hub import snapshot_download
                import os
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                model_cache_path = os.path.join(cache_dir, f"models--{base_model_name.replace('/', '--')}")
                is_cached = os.path.exists(model_cache_path)
                
                if is_cached:
                    print(f"Loading base model from HuggingFace cache: {base_model_name}...")
                    if has_ollama_model:
                        print(f"Note: You have {ollama_model_name} in Ollama, but LoRA adapters require the HuggingFace version (different format).")
                else:
                    print(f"Downloading base model (first time only, will be cached): {base_model_name}...")
                    if has_ollama_model:
                        print(f"Note: You have {ollama_model_name} in Ollama, but LoRA adapters require the HuggingFace PyTorch version.")
                        print(f"      Ollama uses GGUF format (quantized), which is incompatible with LoRA adapters.")
                        print(f"      This is a one-time download (~8GB). The model will be cached for future use.")
                    else:
                        print(f"Note: LoRA adapters require the exact base model. This is a one-time download (~8GB for gemma-3-4b-it).")
            except:
                print(f"Loading base model: {base_model_name}...")
                if has_ollama_model:
                    print(f"Note: You have {ollama_model_name} in Ollama, but LoRA adapters require the HuggingFace version.")
            
            # Load base model first
            # Note: from_pretrained automatically uses cache if available
            # LoRA adapters require the exact base model - cannot use Ollama models
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    token=self.hf_token,
                    local_files_only=self.local_files_only
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
            
            # Determine adapter path - prefer adapter/ subdirectory, fallback to model_path
            if adapter_path.exists() and any(adapter_path.iterdir()):
                adapter_load_path = adapter_path
            else:
                adapter_load_path = self.model_path
            
            print(f"Loading LoRA adapter from {adapter_load_path}...")
            # Load adapter on top of base model
            self.model = PeftModel.from_pretrained(
                base_model,
                str(adapter_load_path),
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
    
    def chat(self, messages: List[Dict[str, str]], max_length: int = 2048, temperature: float = 0.7, model: str = None) -> str:
        """
        Send chat messages to the fine-tuned model
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_length: Maximum length of generated response
            temperature: Sampling temperature (higher = more creative)
            model: Model to use (for remote mode: "base", "v1", etc. Overrides remote_model if set)
        
        Returns:
            Generated response text
        """
        # Use remote API if configured
        if self.mode == "remote":
            return self._chat_remote(messages, max_length, temperature, model or self.remote_model)
        
        # Local mode - existing implementation
        import time
        start_time = time.time()
        
        if not messages:
            return "Error: No messages provided"
        
        print(f"[CHAT LOG] ========================================")
        print(f"[CHAT LOG] Local Model Chat Request")
        print(f"[CHAT LOG] Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[CHAT LOG] Model path: {self.model_path}")
        print(f"[CHAT LOG] Device: {self.device}")
        print(f"[CHAT LOG] Messages: {len(messages)}")
        print(f"[CHAT LOG] Max length: {max_length}, Temperature: {temperature}")
        print(f"[CHAT LOG] Message details:")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            content_preview = content[:150] + "..." if len(content) > 150 else content
            print(f"[CHAT LOG]   [{i+1}] {role}: {len(content)} chars - {content_preview}")
        
        # Format messages using chat template if available
        format_start = time.time()
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
                print(f"[CHAT LOG] Used chat template for formatting")
            else:
                # Fallback formatting
                prompt = self._format_messages_fallback(messages)
                print(f"[CHAT LOG] Used fallback formatting (no chat template)")
        except Exception as e:
            print(f"[CHAT LOG] Error formatting messages: {e}")
            prompt = self._format_messages_fallback(messages)
        
        format_time = time.time() - format_start
        prompt_length = len(prompt)
        print(f"[CHAT LOG] Formatting took {format_time:.2f}s, prompt length: {prompt_length} chars")
        
        # Tokenize
        tokenize_start = time.time()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        tokenize_time = time.time() - tokenize_start
        input_ids_length = inputs['input_ids'].shape[1]
        print(f"[CHAT LOG] Tokenization took {tokenize_time:.2f}s, input tokens: {input_ids_length}")
        
        # Generate
        generate_start = time.time()
        print(f"[CHAT LOG] Starting generation (max_new_tokens={max_length}, temperature={temperature}, device={self.device})...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generate_time = time.time() - generate_start
        output_length = outputs[0].shape[0]
        new_tokens = output_length - input_ids_length
        print(f"[CHAT LOG] Generation took {generate_time:.2f}s, generated {new_tokens} new tokens (total output: {output_length} tokens)")
        
        # Decode response
        decode_start = time.time()
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        # Find where the prompt ends and extract only the new generated text
        prompt_length_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        response_ids = outputs[0][prompt_length_tokens:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # Fallback: if extraction failed, try string replacement
        if not response and prompt in full_response:
            response = full_response.replace(prompt, "").strip()
        elif not response:
            response = full_response.strip()
        
        decode_time = time.time() - decode_start
        response_length = len(response)
        total_time = time.time() - start_time
        
        response_preview = response[:200] + "..." if len(response) > 200 else response
        print(f"[CHAT LOG] Decoding took {decode_time:.2f}s, response length: {response_length} chars")
        print(f"[CHAT LOG] Total chat time: {total_time:.2f}s")
        print(f"[CHAT LOG] Breakdown: format={format_time:.2f}s, tokenize={tokenize_time:.2f}s, generate={generate_time:.2f}s, decode={decode_time:.2f}s")
        print(f"[CHAT LOG] Response preview: {response_preview}")
        print(f"[CHAT LOG] ========================================")
        
        return response
    
    def _chat_remote(self, messages: List[Dict[str, str]], max_length: int, temperature: float, model: str) -> str:
        """Chat using Hugging Face TGI (Text Generation Inference) API"""
        import time
        import requests
        
        start_time = time.time()
        print(f"[CHAT LOG] ========================================")
        print(f"[CHAT LOG] TGI API Request (Hugging Face Text Generation Inference)")
        print(f"[CHAT LOG] URL: {self.remote_url}/v1/chat/completions")
        print(f"[CHAT LOG] Model: {model}")
        print(f"[CHAT LOG] Messages: {len(messages)}")
        print(f"[CHAT LOG] Max tokens: {max_length}, Temperature: {temperature}")
        
        # Log message summary
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            content_preview = content[:150] + "..." if len(content) > 150 else content
            print(f"[CHAT LOG]   [{i+1}] {role}: {len(content)} chars - {content_preview}")
        
        try:
            request_start = time.time()
            # TGI uses OpenAI-compatible /v1/chat/completions endpoint
            # Note: TGI may ignore the model parameter if only one model is loaded
            request_payload = {
                "model": model,  # TGI may ignore this if single model
                "messages": messages,
                "max_tokens": max_length,  # TGI uses max_tokens, not max_length
                "temperature": temperature,
                "stream": False  # Non-streaming for simplicity
            }
            
            print(f"[CHAT LOG] Request payload: model={model}, max_tokens={max_length}, temperature={temperature}")
            
            response = requests.post(
                f"{self.remote_url}/v1/chat/completions",
                json=request_payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            request_time = time.time() - request_start
            print(f"[CHAT LOG] HTTP request completed in {request_time:.2f}s (status: {response.status_code})")
            
            response.raise_for_status()
            result = response.json()
            
            # TGI returns OpenAI-compatible format:
            # {
            #   "choices": [{"message": {"role": "assistant", "content": "..."}}],
            #   "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
            # }
            if "choices" not in result or len(result["choices"]) == 0:
                raise Exception("TGI API returned no choices in response")
            
            response_text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            tokens_generated = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            total_time = time.time() - start_time
            response_preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
            
            print(f"[CHAT LOG] Total time: {total_time:.2f}s")
            print(f"[CHAT LOG] Tokens - prompt: {prompt_tokens}, completion: {tokens_generated}, total: {total_tokens}")
            print(f"[CHAT LOG] Response length: {len(response_text)} chars")
            print(f"[CHAT LOG] Response preview: {response_preview}")
            print(f"[CHAT LOG] ========================================")
            
            return response_text
        except requests.exceptions.Timeout as e:
            error_msg = f"TGI API timeout after 120s: {str(e)}"
            print(f"[CHAT LOG] ERROR: {error_msg}")
            print(f"[CHAT LOG] ========================================")
            raise Exception(error_msg)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"TGI API connection error: {str(e)}"
            print(f"[CHAT LOG] ERROR: {error_msg}")
            print(f"[CHAT LOG] ========================================")
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"TGI API error: {str(e)}"
            print(f"[CHAT LOG] ERROR: {error_msg}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[CHAT LOG] Response status: {e.response.status_code}")
                print(f"[CHAT LOG] Response body: {e.response.text[:500]}")
            print(f"[CHAT LOG] ========================================")
            raise Exception(error_msg)
        except KeyError as e:
            error_msg = f"TGI API response format error: missing key {str(e)}"
            print(f"[CHAT LOG] ERROR: {error_msg}")
            if 'result' in locals():
                print(f"[CHAT LOG] Response structure: {list(result.keys())}")
            print(f"[CHAT LOG] ========================================")
            raise Exception(error_msg)
    
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

