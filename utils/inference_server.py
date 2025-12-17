#!/usr/bin/env python3
"""
Inference server for hosting models on Vast.ai
Supports multiple models (base + fine-tuned versions) with on-the-fly switching
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time
import json
from pathlib import Path

app = FastAPI(title="Anvil Inference Server")

# Global models and tokenizers
models = {}  # {"base": model, "v1": model, etc.}
tokenizers = {}  # {"base": tokenizer, "v1": tokenizer, etc.}
device = "cuda" if torch.cuda.is_available() else "cpu"

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "base"  # "base" or "v1", "v2", etc.
    max_length: int = 1024
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    model_used: str
    tokens_generated: int
    inference_time: float

class ModelInfo(BaseModel):
    name: str
    loaded: bool
    device: str

@app.on_event("startup")
async def load_models():
    """Load all available models on server startup"""
    global models, tokenizers
    
    print(f"[SERVER] Starting inference server on {device}...")
    
    # Get configuration from environment
    base_model_name = os.getenv("BASE_MODEL", "google/gemma-3-4b-it")
    hf_token = os.getenv("HF_TOKEN", "")
    adapter_base_path = os.getenv("ADAPTER_BASE_PATH", "/workspace/output/training")
    
    # Load base model
    print(f"[SERVER] Loading base model: {base_model_name}...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            token=hf_token
        )
        base_model.eval()
        models["base"] = base_model
        tokenizers["base"] = base_tokenizer
        print(f"[SERVER] ✓ Base model loaded")
    except Exception as e:
        print(f"[SERVER] ✗ Failed to load base model: {e}")
        raise
    
    # Load fine-tuned versions (V1, V2, etc.)
    # Look for adapter directories in the training output
    adapter_path = Path(adapter_base_path)
    if adapter_path.exists():
        # Check for adapter directory (from downloaded weights)
        adapter_dir = adapter_path / "adapter"
        if adapter_dir.exists() and any(adapter_dir.iterdir()):
            print(f"[SERVER] Loading V1 adapter from {adapter_dir}...")
            try:
                v1_model = PeftModel.from_pretrained(
                    base_model,
                    str(adapter_dir),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                v1_model.eval()
                models["v1"] = v1_model
                tokenizers["v1"] = base_tokenizer  # Use same tokenizer
                print(f"[SERVER] ✓ V1 model loaded")
            except Exception as e:
                print(f"[SERVER] ✗ Failed to load V1 adapter: {e}")
        
        # Check for versioned directories (V1, V2, etc.)
        for version_dir in adapter_path.parent.glob("V*/weights/adapter"):
            if version_dir.exists() and any(version_dir.iterdir()):
                version_name = version_dir.parent.parent.name.lower()  # "v1", "v2", etc.
                if version_name not in models:
                    print(f"[SERVER] Loading {version_name} adapter from {version_dir}...")
                    try:
                        version_model = PeftModel.from_pretrained(
                            base_model,
                            str(version_dir),
                            torch_dtype=torch.float16 if device == "cuda" else torch.float32
                        )
                        version_model.eval()
                        models[version_name] = version_model
                        tokenizers[version_name] = base_tokenizer
                        print(f"[SERVER] ✓ {version_name} model loaded")
                    except Exception as e:
                        print(f"[SERVER] ✗ Failed to load {version_name} adapter: {e}")
    
    print(f"[SERVER] Server ready! Loaded models: {list(models.keys())}")

def format_messages_fallback(messages: List[Dict[str, str]]) -> str:
    """Fallback message formatting if chat template not available"""
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            formatted += f"System: {content}\n\n"
        elif role == "user":
            formatted += f"User: {content}\n\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n\n"
    formatted += "Assistant: "
    return formatted

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with model selection"""
    start_time = time.time()
    
    # Get model (default to base if not found)
    model_name = request.model.lower()
    if model_name not in models:
        if "base" in models:
            model_name = "base"
            print(f"[CHAT] Model '{request.model}' not found, using 'base'")
        else:
            raise HTTPException(status_code=404, detail=f"Model '{request.model}' not available. Available: {list(models.keys())}")
    
    model = models[model_name]
    tokenizer = tokenizers[model_name]
    
    try:
        # Format messages
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = format_messages_fallback(request.messages)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        response_ids = outputs[0][prompt_length:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        inference_time = time.time() - start_time
        tokens_generated = len(response_ids)
        
        print(f"[CHAT] {model_name}: {tokens_generated} tokens in {inference_time:.2f}s")
        
        return ChatResponse(
            response=response,
            model_used=model_name,
            tokens_generated=tokens_generated,
            inference_time=inference_time
        )
    except Exception as e:
        print(f"[CHAT] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    return [
        ModelInfo(name=name, loaded=True, device=device)
        for name in models.keys()
    ]

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys()),
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("INFERENCE_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)












