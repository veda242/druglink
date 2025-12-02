#!/usr/bin/env python3
"""
preprocess_dailymed.py
- Walks KnowDDI-main/daily med/ and subfolders
- Parses SPL XMLs (.xml) from DailyMed
- Extracts (drug_name, interacting_drug, interaction_text, dosage_text)
- Computes a heuristic confidence score [0..1]
- Normalizes dosage into mg (when possible) and creates a numeric dosage feature
- Writes CSV at data/dailymed_interactions.csv and JSON at data/dailymed_interactions.json
Requirements: lxml (pip install lxml) or fallback to builtin xml.etree
"""
import os
import re
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from lxml import etree as LET
    XML_PARSER = "lxml"
except Exception:
    import xml.etree.ElementTree as ET
    LET = None
    XML_PARSER = "etree"

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "daily med"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

# keywords for interaction sections and high-confidence sections
INTERACTION_KEYWORDS = [
    "drug interaction", "interactions", "interaction", "contraindications",
    "warnings", "precautions", "drug interactions", "clinical pharmacology",
    "adverse reactions", "clinical studies"
]
HIGH_CONF_SECTIONS = ["contraindication", "boxed warning", "black box", "warnings", "contraindications"]

# regex patterns to catch dosages
DOSAGE_PATTERNS = [
    re.compile(r'(\d+(?:\.\d+)?\s*(?:mg|milligram|mcg|µg|g|gram|units|tablet|tab|capsule|mcg/kg|mg/kg))', re.I),
    re.compile(r'(\d+(?:-\d+)?\s*(?:to)\s*\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g))', re.I)
]

UNIT_TO_MG = {
    'mg': 1.0,
    'milligram': 1.0,
    'g': 1000.0,
    'gram': 1000.0,
    'mcg': 0.001,
    'µg': 0.001,
    'units': None,  # keep None, domain-specific
    'tablet': None,
    'tab': None,
    'capsule': None
}

def list_xml_files(folder):
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith('.xml'):
                yield Path(root) / f

def extract_text_from_xml(path):
    """Return concatenated text content of meaningful nodes"""
    try:
        if XML_PARSER == "lxml":
            tree = LET.parse(str(path))
            root = tree.getroot()
            # Many SPL files use /spl/section or /component/structuredBody ...
            texts = []
            for el in root.iter():
                if el.text:
                    texts.append(el.text.strip())
                # also check tail
                if el.tail:
                    texts.append(el.tail.strip())
            return " ".join(t for t in texts if t)
        else:
            tree = ET.parse(str(path))
            root = tree.getroot()
            texts = []
            for el in root.iter():
                if el.text:
                    texts.append(el.text.strip())
                if el.tail:
                    texts.append(el.tail.strip())
            return " ".join(t for t in texts if t)
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}")
        return ""

def find_sections_by_heading(xml_path):
    """
    Attempt to find section headings and section text by searching for <heading> or <title> tags.
    Returns list of (heading_lower, section_text)
    Fallback: whole document as single section.
    """
    try:
        raw = open(xml_path, 'rb').read().decode('utf-8', errors='ignore')
    except:
        raw = ""
    # naive split by common section titles
    sections = []
    # pattern: <heading>Some Title</heading> ... content ...
    heading_re = re.compile(r'<heading[^>]*>(.*?)</heading>', re.I|re.S)
    title_re = re.compile(r'<title[^>]*>(.*?)</title>', re.I|re.S)
    # If we find headings, split at them
    headings = heading_re.findall(raw) or title_re.findall(raw)
    if headings:
        # simple: split by heading tags and pair
        parts = re.split(r'<heading[^>]*>.*?</heading>|<title[^>]*>.*?</title>', raw, flags=re.I|re.S)
        # the headings array may be shorter/longer; zip nearest
        for i, h in enumerate(headings):
            text = parts[i+1] if i+1 < len(parts) else parts[-1]
            sections.append((h.strip().lower(), re.sub(r'<[^>]+>', ' ', text).strip()))
        return sections
    # fallback: try to find section headings in plain text (very naive)
    lower = raw.lower()
    for kw in INTERACTION_KEYWORDS:
        if kw in lower:
            # split on the keyword
            idx = lower.find(kw)
            snippet = raw[max(0, idx-500): idx+2000]
            sections.append((kw, re.sub(r'<[^>]+>', ' ', snippet)))
    if sections:
        return sections
    # last fallback: whole text
    txt = re.sub(r'<[^>]+>', ' ', raw)
    return [("full_text", txt[:5000])]

def find_drug_name(xml_path):
    """
    Try to extract manufacturer/brand/drug name from specific SPL tags such as <title> or <brand_name>.
    Fallback: filename basename
    """
    fname = Path(xml_path).stem
    try:
        raw = open(xml_path, 'rb').read().decode('utf-8', errors='ignore')
    except:
        raw = ""
    # look for <product> or <brand_name> or <title>
    m = re.search(r'<brand_name[^>]*>([^<]+)</brand_name>', raw, re.I)
    if not m:
        m = re.search(r'<title[^>]*>([^<]+)</title>', raw, re.I)
    if m:
        return re.sub(r'\s+', ' ', m.group(1)).strip()
    # fallback to file stem
    return fname

def extract_dosage(snippet):
    """Return first detected dosage text and numeric mg if convertible"""
    # search patterns
    for pat in DOSAGE_PATTERNS:
        m = pat.search(snippet)
        if m:
            txt = m.group(1)
            # parse number and unit
            num_m = re.search(r'(\d+(?:\.\d+)?)', txt)
            unit_m = re.search(r'(mg|milligram|g|gram|mcg|µg|units|tablet|tab|capsule|mg/kg|mcg/kg)', txt, re.I)
            num = float(num_m.group(1)) if num_m else None
            unit = unit_m.group(1).lower() if unit_m else None
            mg_val = None
            if num is not None and unit:
                # handle per-kg heuristics by ignoring /kg for now (we leave as None)
                if '/kg' in txt or 'per kg' in txt:
                    mg_val = None
                else:
                    conv = UNIT_TO_MG.get(unit.split('/')[0], None)
                    if conv is not None:
                        mg_val = num * conv
            return txt, mg_val
    return None, None

def compute_confidence(section_heading, snippet):
    """
    Heuristic confidence scoring [0..1]
    - If heading is contraindications/boxed warning -> high
    - If explicit 'avoid' 'should not' 'contraindicat' -> high
    - If 'may interact' or 'coadministration' -> medium
    - If only mechanistic speculation -> medium-low
    - Base score starts at 0.3
    """
    score = 0.3
    h = (section_heading or "").lower()
    s = (snippet or "").lower()
    # heading cues
    for hf in HIGH_CONF_SECTIONS:
        if hf in h:
            score = max(score, 0.9)
    # keywords
    if any(w in s for w in ["contraindicat", "avoid", "do not", "should not", "not recommended", "boxed warning", "black box"]):
        score = max(score, 0.95)
    if any(w in s for w in ["may increase", "may decrease", "coadministration", "interact", "interaction", "concomitant"]):
        score = max(score, 0.7)
    # presence of explicit effect words increases score
    if any(w in s for w in ["increase concentration", "decrease concentration", "dose adjustment", "dose-dependent"]):
        score = max(score, 0.8)
    # lower confidence if speculative language
    if any(w in s for w in ["may be", "there is limited", "reported rarely", "theoretical"]):
        score = min(score, 0.6)
    return round(float(score), 3)

def extract_interactions_from_text(text):
    """Very naive extractor: look for lines with pattern 'DrugA ... with ... DrugB' using word heuristics.
       We'll return tuples (other_drug_mention, snippet). This is approximate; human review recommended.
    """
    # Look for uppercase-cased drug-like tokens or words that look like drug names (capitalized tokens)
    # We'll use simple rule: any capitalized token of length >=3 that is not start of sentence.
    candidates = set()
    # tokenization
    tokens = re.findall(r'\b[A-Z][A-Za-z0-9\-]{2,}\b', text)
    for t in tokens:
        if len(t) > 2 and not re.match(r'(The|This|And|For|Other|With|Not|It|On)$', t):
            candidates.add(t)
    # also try to extract phrases like "coadministration with X" or "interaction with X"
    others = []
    for m in re.finditer(r'(?:interaction with|coadministration with|with|when administered with|when coadministered with)\s+([A-Z][A-Za-z0-9\-\s]{2,40})', text, re.I):
        other = m.group(1).strip().split('.')[0].split(',')[0]
        others.append(other)
    # combine
    for o in others:
        candidates.add(o.strip())
    return list(candidates)[:20]

def main(input_folder, out_csv, out_json):
    rows = []
    files_processed = 0
    for p in list_xml_files(input_folder):
        files_processed += 1
        drug_name = find_drug_name(p)
        sections = find_sections_by_heading(p)
        doc_text = extract_text_from_xml(p)
        # iterate sections, find ones with interaction keywords
        found_any = False
        for heading, snippet in sections:
            if any(kw in heading for kw in INTERACTION_KEYWORDS) or any(kw in snippet.lower() for kw in INTERACTION_KEYWORDS):
                # extract other drug mentions
                others = extract_interactions_from_text(snippet)
                if not others:
                    # try doc-level extraction
                    others = extract_interactions_from_text(doc_text)
                # extract dosage mention for this section
                dosage_text, dosage_mg = extract_dosage(snippet or doc_text)
                conf = compute_confidence(heading, snippet)
                for o in others:
                    row = {
                        "source_file": str(p),
                        "drug": drug_name,
                        "other_drug_mention": o,
                        "section_heading": heading,
                        "interaction_text_snippet": (snippet[:2000] if snippet else "") ,
                        "dosage_text": dosage_text or "",
                        "dosage_mg": dosage_mg if dosage_mg is not None else "",
                        "confidence": conf
                    }
                    rows.append(row)
                    found_any = True
        # if nothing found, optionally try doc-level weak extraction
        if not found_any:
            others = extract_interactions_from_text(doc_text)
            if others:
                dosage_text, dosage_mg = extract_dosage(doc_text)
                conf = compute_confidence("full_text", doc_text)
                for o in others:
                    rows.append({
                        "source_file": str(p),
                        "drug": drug_name,
                        "other_drug_mention": o,
                        "section_heading": "full_text",
                        "interaction_text_snippet": doc_text[:2000],
                        "dosage_text": dosage_text or "",
                        "dosage_mg": dosage_mg if dosage_mg is not None else "",
                        "confidence": conf
                    })
    # write CSV and JSON
    if rows:
        fieldnames = ["source_file","drug","other_drug_mention","section_heading","interaction_text_snippet","dosage_text","dosage_mg","confidence"]
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Processed {files_processed} files. Extracted {len(rows)} candidate interactions.")
    print(f"Wrote CSV -> {out_csv}")
    print(f"Wrote JSON -> {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="DailyMed folder (default: daily med/)")
    ap.add_argument("--out_csv", default=str(OUT_DIR / "dailymed_interactions.csv"))
    ap.add_argument("--out_json", default=str(OUT_DIR / "dailymed_interactions.json"))
    args = ap.parse_args()
    main(Path(args.input), Path(args.out_csv), Path(args.out_json))
