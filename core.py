# ===== CORE: Production Adaptive LLM Router =====

import os
import time
import random
import torch
import torch.nn.functional as F
import pandas as pd
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================= INIT =================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = "Mokshil/adaptive-router-model"

MODEL = None
TOKENIZER = None

# ================= LOAD MODEL =================
def get_classifier():
    global MODEL, TOKENIZER
    if MODEL is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
        MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return MODEL, TOKENIZER

# ================= LABEL MAP =================
id2level = {0: "L1", 1: "L2", 2: "L3", 3: "L4", 4: "L5"}
level_order = ["L1", "L2", "L3", "L4", "L5"]

# ================= MODEL PREDICTION =================
def predict_with_confidence(prompt):
    model, tokenizer = get_classifier()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    confidence, pred = torch.max(probs, dim=1)
    return id2level[pred.item()], confidence.item()

# ================= NORMALIZATION =================
def normalize_level(prompt, level):
    p = prompt.lower()
    wc = len(p.split())

    if "code" in p or "implement" in p:
        return min(level, "L4", key=lambda x: level_order.index(x))

    return level

# ================= INTENT DETECTION =================
def detect_intent(prompt):
    p = prompt.lower()

    if any(k in p for k in ["design", "architecture", "system design", "research", "transformer", "deep learning"]):
        return "L5"

    if any(k in p for k in ["code", "implement", "write" , "program"]):
        return "L4"

    if any(k in p for k in ["compare", "difference", "vs"]):
        return "L3"

    if any(k in p for k in ["what is", "define", "explain briefly"]):
        return "L2"

    if any(k in p for k in ["rephrase" , "grammar"]):
        return "L1"

    return "L3"

# ================= COST =================
def estimate_cost(prompt, model_type):
    tokens = len(prompt.split())
    factors = {"small": 1, "medium": 3, "large": 5}
    return tokens * factors[model_type]

# ================= MODEL POOLS =================
MODEL_POOLS = {
    "small": ["llama-3.1-8b-instant"],
    "medium": ["llama-3.3-70b-versatile"],
    "large": ["openai/gpt-oss-120b"]
}

# ================= CALL LLM (FIXED) =================
def call_llm(model_type, prompt):

    full_response = ""
    messages = [{"role": "user", "content": prompt}]

    for _ in range(3):  # continuation loop
        for model in MODEL_POOLS[model_type]:
            try:
                start = time.time()

                res = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=600,
                    temperature=0.7
                )

                latency = time.time() - start

                # skip slow models
                if latency > 12:
                    continue

                chunk = res.choices[0].message.content
                full_response += chunk

                messages.append({"role": "assistant", "content": chunk})
                messages.append({"role": "user", "content": "continue"})

                if len(chunk.split()) < 100:
                    return full_response, model

                break

            except:
                continue

    return full_response, model

# ===============COMPRESS RESPONSE =================
def detect_response_type(response):
    if "```" in response:
        return "code"
    if len(response.split()) > 800:
        return "long_text"
    return "normal"


def compress_response(response):
    r_type = detect_response_type(response)

    # Don't compress code
    if r_type == "code":
        return response

    if r_type == "long_text":
        try:
            summary_prompt = f"""
Summarize the following response clearly while preserving key technical details.
Keep structure (headings, bullets).
Limit to ~600 tokens.

Response:
{response}
"""

            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=600
            )

            return res.choices[0].message.content

        except:
            return response

    return response

# ================= ADAPTIVE MEMORY =================
adaptive_memory = {}

def update_memory(prompt, model, latency):
    key = prompt[:50].lower()

    if key not in adaptive_memory:
        adaptive_memory[key] = {"model": model, "latency": latency, "count": 1}
    else:
        entry = adaptive_memory[key]
        entry["latency"] = (entry["latency"] * entry["count"] + latency) / (entry["count"] + 1)
        entry["count"] += 1

def get_memory_hint(prompt):
    return adaptive_memory.get(prompt[:50].lower(), None)

# ================= ROUTING =================
def choose_model(level, confidence, prompt):

    intent = detect_intent(prompt)

    # ===== INTENT PRIORITY =====
    if intent == "[L5,L4]":
        return "large"
    elif intent == "L3":
        return "medium"
    elif intent == "[L1,L2]":
        return "small"

    # ===== LEVEL BASELINE =====
    level_map = {
        "L1": "small",
        "L2": "small",
        "L3": "medium",
        "L4": "large",
        "L5": "large"
    }

    selected = level_map[level]

    # ===== ADAPTIVE MEMORY =====
    hint = get_memory_hint(prompt)
    if hint and hint["latency"] < 2:
        return hint["model"]

    # ===== CONFIDENCE SAFETY =====
    if confidence < 0.6:
        return "large"

    return selected

# ================= LOGS =================
logs = []

# ================= MAIN =================
def full_system(prompt):

    level, confidence = predict_with_confidence(prompt)
    level = normalize_level(prompt, level)

    selected_model = choose_model(level, confidence, prompt)

    start = time.time()
    response, actual_model = call_llm(selected_model, prompt)
    latency = time.time() - start

    response = compress_response(response)

    cost = estimate_cost(prompt, selected_model)

    update_memory(prompt, selected_model, latency)

    log_entry = {
        "query": prompt,
        "level": level,
        "confidence": confidence,
        "model_type": selected_model,
        "actual_model": actual_model,
        "latency": latency,
        "cost": cost
    }

    logs.append(log_entry)

    return response, selected_model, actual_model, latency, cost, log_entry

# ================= ANALYTICS =================
def get_logs_df():
    if not logs:
        return pd.DataFrame()
    return pd.DataFrame(logs)

def compute_efficiency_score(row):
    quality = {"small": 0.6, "medium": 0.8, "large": 1.0}
    return quality[row["model_type"]] / (row["cost"] * row["latency"] + 1e-6)

def compute_dashboard_metrics():
    df = get_logs_df()
    if df.empty:
        return None

    total = len(df)
    avg_latency = df["latency"].mean()
    avg_cost = df["cost"].mean()

    baseline = sum(estimate_cost(q, "large") for q in df["query"])
    optimized = df["cost"].sum()

    reduction = ((baseline - optimized) / baseline) * 100

    df["eff"] = df.apply(compute_efficiency_score, axis=1)

    return {
        "total_queries": total,
        "avg_latency": avg_latency,
        "avg_cost": avg_cost,
        "cost_reduction": reduction,
        "avg_efficiency": df["eff"].mean()
    }

def get_cost_comparison_df():
    df = get_logs_df()
    if df.empty:
        return None

    df["baseline"] = df["query"].apply(lambda q: estimate_cost(q, "large"))
    df["opt_cum"] = df["cost"].cumsum()
    df["base_cum"] = df["baseline"].cumsum()

    return pd.DataFrame({
        "Optimized Cost": df["opt_cum"],
        "Baseline Cost": df["base_cum"]
    })