# drug_validator.py
import csv
from pathlib import Path

def load_valid_drugs(path="valid_drug_names.csv"):
    valid = set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("valid_drug_names.csv not found")
    with p.open("r", encoding="utf-8") as f:
        for row in f:
            name = row.strip().lower()
            if name:
                valid.add(name)
    return valid

VALID_DRUGS = load_valid_drugs()

def is_valid_drug(name: str) -> bool:
    return name.lower().strip() in VALID_DRUGS
