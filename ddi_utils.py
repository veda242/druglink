# ddi_utils.py
#
# Utilities for dosage extraction, dosage risk, and confidence.

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple, Any

import pandas as pd
import numpy as np

# -----------------------------------------
# Load dosage database (dosage_database.csv)
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOSAGE_DB_PATH = BASE_DIR / "dosage_database.csv"

print("Loading dosage DB from:", DOSAGE_DB_PATH)
try:
    _dosage_df = pd.read_csv(DOSAGE_DB_PATH)
    _dosage_df["drug_norm"] = _dosage_df["drug"].astype(str).str.lower().str.strip()
    print("Dosage DB rows:", len(_dosage_df))
except Exception as e:
    print("Error loading dosage_database.csv:", e)
    _dosage_df = pd.DataFrame(
        columns=["drug", "usual_mg", "toxic_mg", "notes", "drug_norm"]
    )


# -----------------------------------------
# Extraction
# -----------------------------------------

def extract_drug_dosages(text: str) -> List[Tuple[str, float | None, str]]:
    """
    Very simple extractor: finds 'drug 500 mg', 'ibuprofen 400mg', etc.

    Returns list of (drug_phrase, qty_mg, "mg").
    If no dose found at all, returns [(full_text, None, "mg")] as fallback.
    """
    if not text:
        return []

    pattern = re.compile(
        r"([A-Za-z0-9\+\-\s]+?)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|g)\b",
        re.IGNORECASE,
    )

    entries: List[Tuple[str, float | None, str]] = []

    for match in pattern.finditer(text):
        phrase = match.group(1).strip()
        qty = float(match.group(2))
        unit_raw = match.group(3).lower()

        # normalize to mg
        if unit_raw == "g":
            qty_mg = qty * 1000.0
        elif unit_raw in ("mcg", "µg", "ug"):
            qty_mg = qty / 1000.0
        else:
            qty_mg = qty

        entries.append((phrase, qty_mg, "mg"))

    if not entries:
        # no dosage detected → just return the text as a single "drug"
        entries.append((text.strip(), None, "mg"))

    return entries


# -----------------------------------------
# Dosage risk helpers
# -----------------------------------------

def _lookup_limits(drug_phrase: str) -> tuple[float | None, float | None]:
    """
    Use dosage_database.csv to get (usual_mg, toxic_mg) for a given phrase.
    Returns (None, None) if not found.
    """
    if _dosage_df.empty:
        return None, None

    name = str(drug_phrase).lower().strip()
    rows = _dosage_df[_dosage_df["drug_norm"] == name]

    if rows.shape[0] == 0:
        # fallback: substring match either way
        rows = _dosage_df[
            _dosage_df["drug_norm"].apply(
                lambda x: name in x or x in name  # type: ignore[arg-type]
            )
        ]

    if rows.shape[0] == 0:
        return None, None

    row = rows.iloc[0]
    try:
        usual = float(row["usual_mg"])
    except Exception:
        usual = None
    try:
        toxic = float(row["toxic_mg"])
    except Exception:
        toxic = None

    return usual, toxic


def aggregate_dosage_risk(entries: List[Tuple[str, float | None, str]]) -> dict:
    """
    entries: list of (drug_phrase, qty_mg, unit)

    Steps:
    - Normalize name (lower/strip).
    - Sum doses for same drug.
    - Use dosage_database.csv:
        * total <= usual_mg        → likely_safe
        * usual_mg–toxic_mg        → caution
        * total  > toxic_mg        → likely_risky
    """
    if not entries:
        return {"overall_risk": "unknown", "per_drug": []}

    # Sum doses per normalized drug name
    totals: dict[str, dict] = {}
    for phrase, qty_mg, unit in entries:
        name = str(phrase).lower().strip()
        if name not in totals:
            totals[name] = {
                "phrases": [phrase.strip()],
                "total_mg": float(qty_mg) if qty_mg is not None else 0.0,
                "unit": unit,
            }
        else:
            totals[name]["phrases"].append(phrase.strip())
            if qty_mg is not None:
                totals[name]["total_mg"] += float(qty_mg)

    per_drug: list[dict] = []
    worst = "likely_safe"
    order = ["likely_safe", "caution", "likely_risky"]

    for name, info in totals.items():
        total = info["total_mg"]
        unit = info["unit"]
        phrase = ", ".join(info["phrases"])

        usual, toxic = _lookup_limits(name)

        if usual is None or toxic is None:
            risk = "unknown"
            details = "No reference dose found in dosage database."
        else:
            if total <= usual:
                risk = "likely_safe"
                details = f"Within usual single dose (~{usual} mg). Total: {total} mg."
            elif total <= toxic:
                risk = "caution"
                details = (
                    f"Above usual dose but below toxic range "
                    f"(usual {usual} mg, toxic {toxic} mg). Total: {total} mg."
                )
            else:
                risk = "likely_risky"
                details = f"Exceeds toxic dose threshold ({toxic} mg). Total: {total} mg."

        per_drug.append(
            {
                "drug": phrase,
                "qty_mg": total,
                "unit": unit,
                "risk": risk,
                "details": details,
            }
        )

        if risk in order and order.index(risk) > order.index(worst):
            worst = risk

    if all(d["risk"] == "unknown" for d in per_drug):
        overall = "unknown"
    else:
        overall = worst

    return {"overall_risk": overall, "per_drug": per_drug}


# -----------------------------------------
# Confidence
# -----------------------------------------

def get_confidence(model: Any, texts: List[str]) -> List[float]:
    """
    Return confidence scores for each text based on model.predict_proba.
    Falls back to 0.5 if anything fails.
    """
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(texts)  # shape (n_samples, n_classes)
            probs = np.asarray(probs)
            # take the max probability per sample
            return probs.max(axis=1).tolist()
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(texts)
            scores = np.asarray(scores)
            # crude squashing
            scores = 1.0 / (1.0 + np.exp(-scores))
            if scores.ndim > 1:
                scores = scores.max(axis=1)
            return scores.tolist()
    except Exception as e:
        print("get_confidence error:", e)

    return [0.5 for _ in texts]
