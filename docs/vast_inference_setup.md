# Hosting Models on Vast.ai for Inference

## Overview

Hosting your model and LoRA weights on Vast.ai can significantly speed up inference if:
- You're currently running on CPU locally
- You want GPU acceleration
- You need to handle multiple requests
- You want to offload compute from your local machine

## Pros & Cons

### ✅ Pros
- **GPU Acceleration**: Much faster inference on GPU instances (10-100x faster than CPU)
- **Better Hardware**: Access to high-end GPUs (A100, RTX 3090, etc.)
- **Scalability**: Can handle multiple concurrent requests
- **Offload Local Resources**: Frees up your local machine
- **Already Have Infrastructure**: You already use Vast.ai for training

### ❌ Cons
- **Cost**: Pay per hour (~$0.20-$2.00/hour depending on GPU) even when idle
- **Network Latency**: ~50-200ms added latency per request
- **Setup Complexity**: Need to deploy API server
- **Instance Management**: Need to keep instance running or start/stop manually
- **Data Transfer**: Need to upload model weights (one-time, but ~8GB for base model)

## Implementation Approach

### Option 1: Simple API Server (Recommended)

Create a FastAPI server on Vast.ai instance that loads your model and serves inference requests.

#### Step 1: Create API Server Script

Create `inference_server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

app = FastAPI()

# Global model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_length: int = 1024
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    inference_time: float

@app.on_event("startup")
async def load_model():
    """Load model on server startup"""
    global model, tokenizer
    
    # Paths on Vast.ai instance
    base_model_name = os.getenv("BASE_MODEL", "google/gemma-3-4b-it")
    adapter_path = os.getenv("ADAPTER_PATH", "/workspace/models/adapter")
    hf_token = os.getenv("HF_TOKEN", "")
    
    print(f"Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        token=hf_token
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.eval()
    print("Model loaded and ready!")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests"""
    import time
    start_time = time.time()
    
    try:
        # Format messages
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in request.messages])
        
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
        
        return ChatResponse(
            response=response,
            tokens_generated=tokens_generated,
            inference_time=inference_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Step 2: Deploy to Vast.ai Instance

1. **Launch a GPU instance** (similar to training, but for inference)
2. **Upload files**:
   - `inference_server.py`
   - Your adapter weights (already have base model cached or download it)
3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn transformers peft torch
   ```
4. **Set environment variables**:
   ```bash
   export BASE_MODEL="google/gemma-3-4b-it"
   export ADAPTER_PATH="/workspace/models/Krabappel/V1/weights/adapter"
   export HF_TOKEN="your_token_here"
   ```
5. **Run server**:
   ```bash
   python inference_server.py
   ```

#### Step 3: Create Remote Client

Modify `FineTunedModelClient` to support remote API calls:

```python
class FineTunedModelClient:
    def __init__(self, model_path: str = None, remote_url: str = None, ...):
        if remote_url:
            self.remote_url = remote_url
            self.mode = "remote"
        else:
            # Existing local mode
            self.mode = "local"
            ...
    
    def chat(self, messages, ...):
        if self.mode == "remote":
            import requests
            response = requests.post(
                f"{self.remote_url}/chat",
                json={
                    "messages": messages,
                    "max_length": max_length,
                    "temperature": temperature
                },
                timeout=120
            )
            return response.json()["response"]
        else:
            # Existing local implementation
            ...
```

### Option 2: Use Existing Training Instance

If you have a training instance running, you could:
1. Keep it running after training completes
2. Deploy the inference server on the same instance
3. Use it for inference between training jobs

**Pros**: No extra instance cost if already running
**Cons**: Instance gets destroyed after Phase 4, so you'd need to modify the workflow

## Cost Considerations

- **GPU Instance**: ~$0.20-$2.00/hour depending on GPU
- **Idle Time**: You pay even when not using it
- **Recommendation**: 
  - Use for heavy inference workloads
  - Consider spot instances for lower cost
  - Or implement auto-start/stop based on usage

## Performance Comparison

| Setup | Speed | Latency | Cost |
|-------|-------|---------|------|
| Local CPU | Slow (10-60s) | 0ms | Free |
| Local GPU | Fast (1-5s) | 0ms | Free (if you have GPU) |
| Vast.ai GPU | Fast (1-5s) | 50-200ms | ~$0.20-2/hr |

## Alternative: Local GPU

If you have a local GPU, that's often better than Vast.ai for inference:
- No network latency
- No ongoing costs
- Faster for single-user scenarios

Check if you can use your local GPU:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Recommendation

1. **First**: Check if you can use local GPU (if available)
2. **If CPU only**: Consider Vast.ai for significant speedup
3. **For production**: Vast.ai makes sense for scalability
4. **For development**: Local is usually fine

Would you like me to implement the remote client integration?






