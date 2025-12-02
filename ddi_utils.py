# ddi_utils.py
import re
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Load a small dosage database (drug -> (usual_dose_mg, toxic_threshold_mg, notes))
# You can expand dosage_database.csv with more rows.
DOSAGE_DB_PATH = Path("dosage_database.csv")

def load_dosage_db(path: Path = DOSAGE_DB_PATH) -> Dict[str, Dict]:
    db = {}
    if not path.exists():
        return db
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r.get("drug").strip().lower()
            try:
                usual = float(r.get("usual_mg") or 0)
            except Exception:
                usual = 0.0
            try:
                toxic = float(r.get("toxic_mg") or 0)
            except Exception:
                toxic = 0.0
            notes = r.get("notes") or ""
            db[name] = {"usual_mg": usual, "toxic_mg": toxic, "notes": notes}
    return db

DOSAGE_DB = load_dosage_db()

# Simple regex to detect tokens like "dolo 650", "warfarin 5 mg", "500mg", "5 mg"
DOSAGE_RE = re.compile(r"([A-Za-z0-9\-\s]+?)\s+(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg)?\b", re.IGNORECASE)

def normalize_drug_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", " ", name.lower()).strip()

def extract_drug_dosages(text: str) -> List[Tuple[str, Optional[float], Optional[str]]]:
    """
    Extract drug-like phrases and numeric dosages from a text.
    Returns list of (drug_phrase, quantity_in_mg_or_none, unit_str_or_none)
    """
    results = []
    for match in DOSAGE_RE.finditer(text):
        raw_name = match.group(1).strip()
        qty = float(match.group(2))
        unit = match.group(3) or "mg"
        unit = unit.lower()
        # normalize quantity to mg (very simplistic)
        qty_mg = qty
        if unit in ("g",):
            qty_mg = qty * 1000.0
        elif unit in ("mcg", "µg"):
            qty_mg = qty / 1000.0
        results.append((raw_name, qty_mg, unit))
    # If no explicit dosage found, try to split by 'and' or ',' to get drug names
    if not results:
        # split tokens and return plain drug guesses without dosages
        items = re.split(r"[,\n/;]| and | & ", text)
        for item in items:
            token = item.strip()
            if token:
                results.append((token, None, None))
    return results

def check_dosage_risk_for_entry(drug_phrase: str, qty_mg: Optional[float]) -> Dict:
    """
    Using the dosage DB (DOSAGE_DB) decide whether qty is likely harmful.
    Returns dict: {'drug':..., 'qty_mg':..., 'risk':'safe'|'possible_harm'|'unknown', 'details':...}
    """
    norm = normalize_drug_name(drug_phrase)
    # try exact match
    info = DOSAGE_DB.get(norm)
    if info is None:
        # try partial match: any key contained in norm or vice versa
        for k, v in DOSAGE_DB.items():
            if k in norm or norm in k:
                info = v
                break

    if qty_mg is None:
        return {"drug": drug_phrase, "qty_mg": None, "risk": "unknown", "details": "No dosage specified"}

    if info:
        toxic = info.get("toxic_mg") or 0.0
        usual = info.get("usual_mg") or 0.0
        # simple thresholds:
        if toxic > 0 and qty_mg >= toxic:
            return {"drug": drug_phrase, "qty_mg": qty_mg, "risk": "harmful", "details": f"above toxic threshold ({toxic} mg). {info.get('notes')}"}
        elif usual > 0 and qty_mg > 2.5 * usual:
            return {"drug": drug_phrase, "qty_mg": qty_mg, "risk": "possible_harm", "details": f"more than 2.5x usual dose ({usual} mg). {info.get('notes')}"}
        else:
            return {"drug": drug_phrase, "qty_mg": qty_mg, "risk": "likely_safe", "details": f"within expected dose range ({usual} mg typical). {info.get('notes')}"}
    else:
        return {"drug": drug_phrase, "qty_mg": qty_mg, "risk": "unknown", "details": "No dosage data for this drug in database"}

def aggregate_dosage_risk(entries: List[Tuple[str, Optional[float], Optional[str]]]) -> Dict:
    results = []
    highest = "likely_safe"
    order = {"unknown": 0, "likely_safe": 1, "possible_harm": 2, "harmful": 3}
    for drug, qty, unit in entries:
        r = check_dosage_risk_for_entry(drug, qty)
        results.append(r)
        if order.get(r["risk"], 0) > order.get(highest, 0):
            highest = r["risk"]
    return {"overall_risk": highest, "per_drug": results}

# Confidence helper
def get_confidence(model, texts: List[str]) -> List[float]:
    """
    Return confidence scores in range [0,1].
    If predict_proba available -> max class probability.
    Else if decision_function available -> convert to 0-1 via logistic.
    Else return 0.5 for each (unknown).
    """
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(texts)
            # take max probability per sample
            confidences = [float(max(p)) for p in probs]
            return confidences
        elif hasattr(model, "decision_function"):
            import math
            df = model.decision_function(texts)
            # for binary, df shape (n,)
            if hasattr(df[0], "__iter__"):
                # multiclass: softmax-like conversion
                def softmax_row(row):
                    ex = [math.exp(x) for x in row]
                    s = sum(ex)
                    return max(ex) / s
                return [float(softmax_row(row)) for row in df]
            else:
                # scalar decision function -> map via sigmoid
                def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
                return [float(sigmoid(x)) for x in df]
    except Exception:
        pass
    # fallback
    return [0.5 for _ in texts]
