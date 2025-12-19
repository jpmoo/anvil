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
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
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

 # FORCE float32 for Gemma stability on Vast GPUs
model_dtype = torch.float32

# Small stability/perf toggles (safe no-ops on unsupported hardware)
if device == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "base"  # "base" or "v1", "v2", etc.
    max_length: int = 1024
    temperature: float = 0.7
    summary: Optional[str] = None
    update_summary: bool = False

class ChatResponse(BaseModel):
    response: str
    model_used: str
    tokens_generated: int
    inference_time: float
    summary: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    loaded: bool
    device: str

@app.on_event("startup")
async def load_models():
    """Load all available models on server startup"""
    global models, tokenizers
    
    print(f"[SERVER] Starting inference server on {device}...")
    print(f"[SERVER] Using dtype: {model_dtype}")
    
    # Get configuration from environment
    base_model_name = os.getenv("BASE_MODEL", "google/gemma-3-4b-it")
    hf_token = os.getenv("HF_TOKEN", "")
    # Adapter base path should be /workspace/models/ProfileName (e.g., /workspace/models/Krabappel)
    adapter_base_path = os.getenv("ADAPTER_BASE_PATH", "/workspace/models")
    profile_name = os.getenv("PROFILE_NAME", "")
    
    # Load base model
    print(f"[SERVER] Loading base model: {base_model_name}...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        # Ensure pad token is defined (important for some generation paths)
        if base_tokenizer.pad_token_id is None and base_tokenizer.eos_token_id is not None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=model_dtype,
            device_map="auto" if device == "cuda" else None,
            token=hf_token,
            trust_remote_code=True  # Required for Gemma models
        )
        base_model.eval()
        # Ensure generation config has sane pad/eos ids
        try:
            if base_model.generation_config.pad_token_id is None and base_tokenizer.eos_token_id is not None:
                base_model.generation_config.pad_token_id = base_tokenizer.eos_token_id
            if base_model.generation_config.eos_token_id is None and base_tokenizer.eos_token_id is not None:
                base_model.generation_config.eos_token_id = base_tokenizer.eos_token_id
        except Exception:
            pass
        models["base"] = base_model
        tokenizers["base"] = base_tokenizer
        print(f"[SERVER] ✓ Base model loaded")
    except Exception as e:
        print(f"[SERVER] ✗ Failed to load base model: {e}")
        raise
    
    # Load fine-tuned versions (V1, V2, etc.) as separate model instances
    # Each adapter gets its own fresh base model to avoid weight contamination
    if profile_name:
        profile_path = Path(adapter_base_path) / profile_name
        if profile_path.exists():
            for version_dir in sorted(profile_path.glob("V*/adapter")):
                if version_dir.exists() and any(version_dir.iterdir()):
                    version_name = version_dir.parent.name.lower()  # "v1", "v2", etc.
                    if version_name in models:
                        continue

                    print(f"[SERVER] Loading {version_name} adapter from {version_dir}...")
                    try:
                        adapter_base = AutoModelForCausalLM.from_pretrained(
                            base_model_name,
                            dtype=model_dtype,
                            device_map="auto" if device == "cuda" else None,
                            token=hf_token,
                            trust_remote_code=True
                        )
                        adapter_base.eval()

                        version_model = PeftModel.from_pretrained(
                            adapter_base,
                            str(version_dir),
                            is_trainable=False
                        )
                        version_model.eval()

                        models[version_name] = version_model
                        tokenizers[version_name] = base_tokenizer

                        print(f"[SERVER] ✓ {version_name} model loaded")
                    except Exception as e:
                        print(f"[SERVER] ✗ Failed to load {version_name} adapter: {e}")
    
    print(f"[SERVER] Server ready! Loaded models: {list(models.keys())}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with model selection"""
    def build_instruction(messages: List[Dict[str, str]]) -> str:
        """
        Build a single Gemma-compatible instruction string.
        System messages are treated as authoritative background context
        and are never interleaved with user turns.
        """
        system_blocks = []
        user_blocks = []

        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue

            if role == "system":
                system_blocks.append(content)
            elif role == "user":
                user_blocks.append(content)

        if not user_blocks:
            raise HTTPException(status_code=400, detail="No user instruction provided")

        instruction_parts = []

        if system_blocks:
            instruction_parts.append(
                "SYSTEM CONTEXT (background only — do not respond to this):\n"
                + "\n\n".join(system_blocks)
            )

        instruction_parts.append("USER INSTRUCTION:\n" + user_blocks[-1])

        return "\n\n".join(instruction_parts)

    instruction = build_instruction(request.messages)

    # Gemma-IT does not support system role.
    # We merge system context into the *final user instruction* explicitly.

    # Log normalized instruction (single prompt)
    print("=== NORMALIZED INSTRUCTION ===")
    print(repr(instruction[:200]))
    print("=== END NORMALIZED ===")
    
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
        # Build Gemma-compatible chat prompt and TOKENIZE DIRECTLY from the template.
        # This avoids re-encoding a rendered string, which can cause gibberish if anything
        # about special tokens/template handling differs.
        chat_messages = [
            {"role": "user", "content": instruction}
        ]

        input_ids = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # apply_chat_template() returns a tensor when tokenize=True
        if isinstance(input_ids, dict):
            # Defensive (older/newer HF variations)
            inputs = input_ids
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        else:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Prompt length from tokenized ids
        prompt_length = int(inputs["input_ids"].shape[-1])
        
        # Generation hardening: remove NaN/Inf logits and renormalize
        logits_processor = LogitsProcessorList([InfNanRemoveLogitsProcessor()])

        # Simplified EOS handling
        eos_token_id = tokenizer.eos_token_id

        # Generation settings
        max_new_tokens = int(getattr(request, "max_length", 1024))
        max_new_tokens = max(1, min(max_new_tokens, 512))  # keep responses bounded

        # Default to deterministic decode (prevents “token soup” during bring-up).
        # Only enable sampling if the caller sets temperature > 0.
        temperature = float(getattr(request, "temperature", 0.0))
        if temperature and temperature > 0.0:
            do_sample = True
            temperature = max(0.05, min(temperature, 1.0))
        else:
            do_sample = False
            temperature = 1.0

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            logits_processor=logits_processor,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

        # Constrained sampling to prevent multilingual/gibberish drift.
        if do_sample:
            gen_kwargs.update(
                top_p=0.9,
                top_k=50,
            )

        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )

        # Decode only the newly generated tokens (Gemma-safe)
        generated_tokens = outputs[0][prompt_length:]
        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        # If decoding stripped everything (can happen if the model emitted only special tokens),
        # fall back to decoding without skipping special tokens, then strip obvious markers.
        if not response:
            response_raw = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            ).strip()
            # Remove common special-token artifacts if present
            response = response_raw.replace(tokenizer.eos_token or "", "").strip()
        
        def update_rolling_summary(existing_summary: Optional[str], user_instruction: str, assistant_reply: str) -> str:
            prompt = (
                "You are maintaining a rolling conversation memory.\n"
                "Update the summary below to reflect the latest exchange.\n\n"
                "Rules:\n"
                "- 5–8 bullet points maximum\n"
                "- Preserve important context, goals, preferences, and constraints\n"
                "- Remove obsolete details\n"
                "- Do NOT include dialogue\n\n"
                f"Existing summary:\n{existing_summary or '(none)'}\n\n"
                f"Latest user instruction:\n{user_instruction}\n\n"
                f"Assistant response:\n{assistant_reply}\n\n"
                "Updated summary:"
            )

            chat_messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if not isinstance(input_ids, dict):
                input_ids = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

            input_ids = {k: v.to(device) for k, v in input_ids.items()}

            outputs = model.generate(
                **input_ids,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            summary_tokens = outputs[0][input_ids["input_ids"].shape[-1]:]
            return tokenizer.decode(summary_tokens, skip_special_tokens=True).strip()

        new_summary = None
        if request.update_summary:
            new_summary = update_rolling_summary(
                request.summary,
                request.messages[-1]["content"],
                response
            )
        
        inference_time = time.time() - start_time
        tokens_generated = len(outputs[0]) - prompt_length
        
        print(f"[CHAT] {model_name}: {tokens_generated} tokens in {inference_time:.2f}s")
        
        return ChatResponse(
            response=response,
            model_used=model_name,
            tokens_generated=tokens_generated,
            inference_time=inference_time,
            summary=new_summary
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
