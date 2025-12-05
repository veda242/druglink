# main.py - LLaMA3 (Groq API) Version with bullet-point explanations + CORS
"""
Enhanced KnowDDI API using LLaMA 3 (Groq) + rule-based dosage risk.

- Uses models/ddi_baseline_model.joblib for interaction classification.
- Uses dosage_database.csv + ddi_utils.aggregate_dosage_risk for dosage safety.
- Explanations are bullet points (LLM or deterministic fallback).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Any, Dict
import joblib
import os
import json
import hashlib
import shelve
import traceback

# Local utilities
from ddi_utils import extract_drug_dosages, aggregate_dosage_risk, get_confidence

# Groq (LLaMA3 client)
try:
    from groq import Groq

    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False
    Groq = None

CACHE_PATH = "llm_cache.db"
app = FastAPI(title="KnowDDI - LLaMA3 DDI API (bullet points)")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: allow all
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ------------- Model loading -------------

MODEL_PATH = Path("models/ddi_baseline_model.joblib")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

LABEL_MAP = {0: "no_interaction", 1: "interaction"}


class Query(BaseModel):
    texts: List[str]


# ------------- Cache helpers -------------

def make_cache_key(text: str, label: str, dosage_summary: dict) -> str:
    payload = json.dumps(
        {"text": text, "label": label, "dosage": dosage_summary},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_get(key: str):
    try:
        with shelve.open(CACHE_PATH) as db:
            return db.get(key)
    except Exception:
        return None


def cache_set(key: str, value: str):
    try:
        with shelve.open(CACHE_PATH) as db:
            db[key] = value
    except Exception:
        pass


# ------------- Bullet-point fallback -------------

def generate_bullet_fallback(text, label, conf, dosage_summary):
    """
    Deterministic explanation if LLaMA is unavailable.
    """
    try:
        conf_txt = f"{conf * 100:.0f}%" if conf is not None else "unknown"
        overall = dosage_summary.get("overall_risk", "unknown")
        per = dosage_summary.get("per_drug", [])

        per_lines = []
        for d in per:
            drug = d.get("drug") or "drug"
            qty = d.get("qty_mg")
            qty_text = f"{qty} mg" if qty is not None else "no dosage specified"
            risk = d.get("risk", "unknown")
            details = d.get("details", "")
            per_lines.append(f"{drug} - {qty_text} — {risk}. {details}".strip())

        txt_lower = text.lower() if text else ""
        if label.lower().startswith("interaction"):
            mech = "Potential interaction or overdose risk."
            side_effects = "Depends on drugs; may include bleeding, organ toxicity, or CNS effects."
            checks = "Review medication list, labs, and dosing; consider adjusting therapy."
        else:
            mech = "No major interaction flagged by the model."
            side_effects = "Minimal expected interaction-related side effects."
            checks = "Verify with clinical references or a pharmacist if concerned."

        bullets = []
        bullets.append(f"- Interaction status: {label} (confidence: {conf_txt}).")
        bullets.append(f"- Overall dosage risk: {overall}.")
        if per_lines:
            bullets.append("- Key drug details:")
            for pl in per_lines:
                bullets.append(f"  • {pl}")
        bullets.append(f"- Likely mechanism: {mech}")
        bullets.append(f"- Possible side effects: {side_effects}")
        bullets.append(
            "- Recommendation: "
            + (
                "Avoid or adjust combination unless clearly indicated."
                if label.lower().startswith("interaction")
                else "Combination appears low-risk; confirm with clinician if in doubt."
            )
        )
        bullets.append(f"- Precautions / checks: {checks}")

        return "\n".join(bullets)
    except Exception as e:
        return f"- Unable to generate fallback explanation; verify clinically. (Error: {e})"


# ------------- LLaMA call + cache -------------

def call_llama_with_bullets(prompt, text, label, conf, dosage_summary):
    """
    Try LLaMA via Groq. If not available, use deterministic fallback.
    Caching used to avoid repeated calls.
    """
    key = make_cache_key(text, label, dosage_summary)
    cached = cache_get(key)
    if cached:
        # return cached text as-is (no [cached] prefix)
        return cached

    if not GROQ_AVAILABLE:
        fb = generate_bullet_fallback(text, label, conf, dosage_summary)
        cache_set(key, fb)
        return fb

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        fb = generate_bullet_fallback(text, label, conf, dosage_summary)
        cache_set(key, fb)
        return fb

    try:
        client = Groq(api_key=api_key)
        system_msg = (
            "You are a clinical pharmacology assistant. "
            "Provide a brief, clear explanation in bullet points only. "
            "Do not write paragraphs."
        )
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=220,
            temperature=0.2,
        )
        msg = resp.choices[0].message.content.strip()
        cache_set(key, msg)
        return msg
    except Exception as e:
        print("Groq error:", e)
        fb = generate_bullet_fallback(text, label, conf, dosage_summary)
        cache_set(key, fb)
        return fb


# ------------- OPTIONS handler (for CORS preflight) -------------

@app.options("/predict_enhanced")
def options_predict():
    return {}


# ------------- Main endpoint -------------

@app.post("/predict_enhanced")
async def predict_enhanced(q: Query):
    texts = q.texts
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    # ---- model predictions ----
    try:
        raw_preds = model.predict(texts)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    labels = [LABEL_MAP.get(int(x), str(x)) for x in raw_preds]

    # ---- confidence ----
    try:
        confidences = get_confidence(model, texts)
    except Exception as e:
        print("confidence error:", e)
        confidences = [0.5 for _ in texts]

    # ---- dosage extraction + risk ----
    dosage_infos = []
    for idx, txt in enumerate(texts):
        try:
            entries = extract_drug_dosages(txt)
            agg = aggregate_dosage_risk(entries)

            # If model says "interaction" but risk is still likely_safe,
            # bump overall_risk to interaction_risk for UI clarity.
            label_now = labels[idx]
            if "interaction" in label_now.lower() and agg.get("overall_risk") == "likely_safe":
                agg["overall_risk"] = "interaction_risk"

            formatted_entries = [
                {"drug_phrase": e[0], "qty_mg": e[1], "unit": e[2]} for e in entries
            ]
            dosage_infos.append({"entries": formatted_entries, "aggregated": agg})
        except Exception as e:
            print("dosage error:", e)
            dosage_infos.append({"entries": [], "aggregated": {"overall_risk": "unknown", "per_drug": []}})

    # ---- build prompts + explanations ----
    explanations = []
    for i, txt in enumerate(texts):
        label = labels[i]
        conf = confidences[i]
        dosage_summary = dosage_infos[i]["aggregated"]

        prompt = (
            f"Input: \"{txt}\".\n"
            f"Prediction: {label} (confidence: {conf}).\n"
            f"Dosage Summary: {json.dumps(dosage_summary)}\n\n"
            "Provide explanation strictly as bullet points only. Use this exact checklist of bullets:\n"
            "- Are these drugs interacting? (yes/no)\n"
            "- Why (mechanism)?\n"
            "- What side effects can they cause?\n"
            "- Recommendation (avoid/ok with monitoring):\n"
            "- Precautions to follow:\n"
            "- When to seek medical help:\n"
            "Keep each bullet short and clinician-focused."
        )

        expl = call_llama_with_bullets(prompt, txt, label, conf, dosage_summary)
        explanations.append(expl)

    return {
        "predictions": labels,
        "raw": [int(x) for x in raw_preds],
        "confidence": confidences,
        "dosage_checks": dosage_infos,
        "llm_explanations": explanations,
    }


@app.get("/")
def root():
    return {"status": "LLaMA3 DDI API (bullet points) running"}
