# main.py - LLaMA3 (Groq API) Version with bullet-point explanations + CORS
"""
Enhanced KnowDDI API using LLaMA 3 (Groq).
Explanations are returned as bullet points (both LLaMA output and fallback).
Caching via shelve included. CORS enabled for frontend calls.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib
from typing import List, Any, Dict
import os
import json
import hashlib
import shelve
import traceback

# Local utilities - ensure these exist in repo
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

# ---- CORS (allow frontend to call this API) ----
# For development allow all origins. In production replace ["*"] with your exact origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# ------------------------------------------------

# Load your ML model
MODEL_PATH = Path("models/ddi_baseline_model.joblib")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

LABEL_MAP = {0: "no_interaction", 1: "interaction"}

class Query(BaseModel):
    texts: List[str]

# -----------------------
# Cache helpers
# -----------------------
def make_cache_key(text: str, label: str, dosage_summary: dict) -> str:
    payload = json.dumps({"text": text, "label": label, "dosage": dosage_summary}, sort_keys=True, ensure_ascii=False)
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

# -----------------------
# Bullet-point fallback explanation
# -----------------------
def generate_bullet_fallback(text, label, conf, dosage_summary):
    """
    Return a concise bullet-point explanation built deterministically.
    """
    try:
        conf_txt = f"{conf*100:.0f}%" if conf is not None else "unknown"
        overall = dosage_summary.get("overall_risk", "unknown")
        per = dosage_summary.get("per_drug", [])

        # Build per-drug bullet lines
        per_lines = []
        for d in per:
            drug = d.get("drug") or d.get("drug_phrase") or "drug"
            qty = d.get("qty_mg")
            qty_text = f"{qty} mg" if qty is not None else "no dosage specified"
            risk = d.get("risk", "unknown")
            details = d.get("details", "")
            per_lines.append(f"{drug} - {qty_text} — {risk}. {details}".strip())

        # Heuristic mechanism detection
        txt_lower = text.lower() if text else ""
        if label.lower().startswith("interaction"):
            if "warfarin" in txt_lower or "anticoagulant" in txt_lower:
                mech = "Potential bleeding risk (anticoagulant interaction)."
                side_effects = "Bleeding, bruising, anemia."
                checks = "Check INR, bleeding signs; avoid NSAIDs if possible."
            elif "acetaminophen" in txt_lower or "paracetamol" in txt_lower:
                mech = "Possible hepatotoxicity with high cumulative doses."
                side_effects = "Liver injury, nausea, elevated LFTs."
                checks = "Check liver function tests; avoid alcohol."
            elif "ibuprofen" in txt_lower or "nsaid" in txt_lower:
                mech = "NSAID-related GI/renal risk; may increase bleeding risk."
                side_effects = "Stomach bleeding, renal impairment."
                checks = "Check renal function; monitor GI symptoms."
            else:
                mech = "Possible pharmacodynamic or pharmacokinetic interaction."
                side_effects = "Depends on drugs involved; monitor clinically."
                checks = "Review full medication list and relevant labs."
        else:
            mech = "No major interaction flagged by the model."
            side_effects = "Minimal expected interaction-related side effects."
            checks = "Verify with clinical references if concerned."

        bullets = []
        bullets.append(f"- Interaction status: {label} (confidence: {conf_txt}).")
        bullets.append(f"- Overall dosage risk: {overall}.")
        if per_lines:
            bullets.append("- Key drug details:")
            for pl in per_lines:
                bullets.append(f"  • {pl}")
        bullets.append(f"- Likely mechanism: {mech}")
        bullets.append(f"- Possible side effects: {side_effects}")
        bullets.append(f"- Recommendation: {'Avoid combining unless prescribed by a clinician.' if label.lower().startswith('interaction') else 'Combination appears low-risk; confirm with clinician if in doubt.'}")
        bullets.append(f"- Precautions / checks: {checks}")

        return "\n".join(bullets)
    except Exception as e:
        return f"- Unable to generate fallback explanation; verify clinically. (Error: {e})"

# -----------------------
# LLaMA call + fallback + cache
# -----------------------
def call_llama_with_bullets(prompt, text, label, conf, dosage_summary):
    """
    Try to fetch explanation from Groq LLaMA. On failure or missing client/key, use fallback.
    Use caching to avoid repeated calls.
    """
    key = make_cache_key(text, label, dosage_summary)
    cached = cache_get(key)
    if cached:
        # Just return the cached explanation, no debug prefix
        return cached

    # If Groq client is not available, use deterministic fallback
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
            "You are a clinical pharmacology assistant. Provide a brief, clear explanation in bullet points only. "
            "Do not write paragraphs. Use short bullets that a clinician can use to decide next steps."
        )
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.2,
        )
        msg = resp.choices[0].message.content.strip()
        cache_set(key, msg)
        return msg
    except Exception:
        fb = generate_bullet_fallback(text, label, conf, dosage_summary)
        cache_set(key, fb)
        return fb

# -----------------------
# Endpoint
# -----------------------
@app.options("/predict_enhanced")
def options_predict():
    return {}

@app.post("/predict_enhanced")
async def predict_enhanced(q: Query):
    texts = q.texts
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    # Predictions
    try:
        raw_preds = model.predict(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    labels = [LABEL_MAP.get(int(x), str(x)) for x in raw_preds]

    # Confidence
    try:
        confidences = get_confidence(model, texts)
    except Exception:
        confidences = [0.5 for _ in texts]

    # Dosage extraction & aggregation
    dosage_infos = []
    for txt in texts:
        try:
            entries = extract_drug_dosages(txt)
            agg = aggregate_dosage_risk(entries)
            formatted_entries = [
                {"drug_phrase": e[0], "qty_mg": e[1], "unit": e[2]}
                for e in entries
            ]
            dosage_infos.append({"entries": formatted_entries, "aggregated": agg})
        except Exception:
            dosage_infos.append({"entries": [], "aggregated": {}})

    # Build prompts and request explanations (bullet points)
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
        "llm_explanations": explanations
    }

@app.get("/")
def root():
    return {"status": "LLaMA3 DDI API (bullet points) running"}
