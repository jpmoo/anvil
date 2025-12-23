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
import re

app = FastAPI(title="Anvil Inference Server")

# ---------------------------------------------------------------------
# Readiness state
# ---------------------------------------------------------------------
MODEL_READY = False
MODEL_ERROR = None

models = {}
tokenizers = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

# FORCE float32 for Gemma stability on Vast GPUs
model_dtype = torch.float32

if device == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

# ---------------------------------------------------------------------
# Environment-controlled safety limits
# ---------------------------------------------------------------------
MAX_NEW_TOKENS_CAP = int(os.getenv("MAX_NEW_TOKENS_CAP", "1536"))
# Tier 1 Fix: Set MAX_GENERATION_SECONDS to 25-30 seconds to force graceful stopping
MAX_GENERATION_SECONDS = int(os.getenv("MAX_GENERATION_SECONDS", "30"))

# ✅ CHANGE 1: allow server-side completion buffer
COMPLETION_BUFFER_RATIO = float(os.getenv("COMPLETION_BUFFER_RATIO", "1.25"))

# ---------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "base"
    max_length: int = 512  # deprecated
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None
    repetition_penalty: Optional[float] = None
    summary: Optional[str] = None
    update_summary: bool = False
    verbosity: Optional[str] = None
    min_paragraphs: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    model_used: str
    tokens_generated: int
    inference_time: float
    finish_reason: str
    summary: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    loaded: bool
    device: str

# ---------------------------------------------------------------------
# Startup: load models
# ---------------------------------------------------------------------
@app.on_event("startup")
async def load_models():
    global models, tokenizers, MODEL_READY, MODEL_ERROR
    MODEL_READY = False
    MODEL_ERROR = None

    base_model_name = os.getenv("BASE_MODEL", "google/gemma-3-4b-it")
    hf_token = os.getenv("HF_TOKEN", "")
    adapter_base_path = os.getenv("ADAPTER_BASE_PATH", "/workspace/models")
    profile_name = os.getenv("PROFILE_NAME", "")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            token=hf_token,
            trust_remote_code=True,
        )

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=model_dtype,
            device_map="auto" if device == "cuda" else None,
            token=hf_token,
            trust_remote_code=True,
        )
        base_model.eval()

        base_model.generation_config.pad_token_id = tokenizer.eos_token_id
        base_model.generation_config.eos_token_id = tokenizer.eos_token_id

        models["base"] = base_model
        tokenizers["base"] = tokenizer

    except Exception as e:
        MODEL_ERROR = str(e)
        raise

    # Load adapters
    if profile_name:
        profile_path = Path(adapter_base_path) / profile_name
        if profile_path.exists():
            for version_dir in sorted(profile_path.glob("V*/adapter")):
                version_name = version_dir.parent.name.lower()
                if version_name in models:
                    continue

                adapter_base = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    dtype=model_dtype,
                    device_map="auto" if device == "cuda" else None,
                    token=hf_token,
                    trust_remote_code=True,
                )
                adapter_base.eval()

                model = PeftModel.from_pretrained(
                    adapter_base,
                    str(version_dir),
                    is_trainable=False,
                )
                model.eval()

                models[version_name] = model
                tokenizers[version_name] = tokenizer

    MODEL_READY = True

# ---------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------
@app.get("/ready")
async def ready():
    if MODEL_READY:
        return {"status": "ready", "models_loaded": list(models.keys()), "device": device}
    if MODEL_ERROR:
        raise HTTPException(status_code=500, detail=MODEL_ERROR)
    raise HTTPException(status_code=503, detail="loading")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ready": MODEL_READY,
        "models_loaded": list(models.keys()),
        "device": device,
    }

# ---------------------------------------------------------------------
# Core chat endpoint
# ---------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not ready")

    def sanitize(messages):
        cleaned = []
        last_assistant = None
        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant" and content == last_assistant:
                continue
            cleaned.append({"role": role, "content": content})
            last_assistant = content if role == "assistant" else None
        return cleaned

    request.messages = sanitize(request.messages)

    def build_instruction(messages):
        system = []
        dialogue = []
        for m in messages:
            if m["role"] == "system":
                system.append(m["content"])
            else:
                dialogue.append(m["content"])

        if not dialogue:
            raise HTTPException(status_code=400, detail="No user instruction")

        parts = []
        if system:
            parts.append("\n".join(system))
        if len(dialogue) > 1:
            parts.append("\n".join(dialogue[:-1][-4:]))
        parts.append(dialogue[-1])
        return "\n\n".join(parts)

    instruction = build_instruction(request.messages)

    model_name = request.model.lower()
    if model_name not in models:
        model_name = "base"

    model = models[model_name]
    tokenizer = tokenizers[model_name]

    start_time = time.time()

    chat_messages = [{"role": "user", "content": instruction}]
    input_ids = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if not isinstance(input_ids, dict):
        input_ids = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    input_ids = {k: v.to(device) for k, v in input_ids.items()}
    prompt_length = input_ids["input_ids"].shape[-1]

    # -----------------------------------------------------------------
    # ✅ CHANGE 2: buffered max_new_tokens
    # -----------------------------------------------------------------
    requested = (
        int(request.max_new_tokens)
        if request.max_new_tokens is not None
        else int(request.max_length)
    )

    max_new_tokens = min(
        int(requested * COMPLETION_BUFFER_RATIO),
        MAX_NEW_TOKENS_CAP,
    )

    temperature = request.temperature
    if temperature is None:
        do_sample = False
        temperature = 1.0
    else:
        do_sample = request.do_sample if request.do_sample is not None else temperature > 0

    logits_processor = LogitsProcessorList([InfNanRemoveLogitsProcessor()])

    # Use repetition_penalty from request if provided, otherwise default to 1.1
    # Fix: Allow lower repetition_penalty (1.05) for expansion requests
    repetition_penalty = float(request.repetition_penalty) if request.repetition_penalty is not None else 1.1
    
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(temperature),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        logits_processor=logits_processor,
        # -----------------------------------------------------------------
        # ✅ CHANGE 3: allow Gemma to finish structure
        # -----------------------------------------------------------------
        early_stopping=False,
    )

    if request.top_p is not None:
        gen_kwargs["top_p"] = float(request.top_p)
    if request.top_k is not None:
        gen_kwargs["top_k"] = int(request.top_k)

    if time.time() - start_time > MAX_GENERATION_SECONDS:
        raise HTTPException(status_code=504, detail="Generation time exceeded")

    outputs = model.generate(**input_ids, **gen_kwargs)

    generated = outputs[0][prompt_length:]
    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    tokens_generated = len(generated)
    inference_time = time.time() - start_time

    # -----------------------------------------------------------------
    # Finish reason: do NOT penalize near-cap completions
    # -----------------------------------------------------------------
    finish_reason = "stop"

    print(
        f"[CHAT] {model_name}: requested={requested}, "
        f"buffered={max_new_tokens}, generated={tokens_generated}, "
        f"time={inference_time:.2f}s"
    )

    return ChatResponse(
        response=response,
        model_used=model_name,
        tokens_generated=tokens_generated,
        inference_time=inference_time,
        finish_reason=finish_reason,
        summary=None,
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    return [
        ModelInfo(name=name, loaded=True, device=device)
        for name in models.keys()
    ]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("INFERENCE_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)