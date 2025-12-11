# train_baseline.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

CSV_PATH = os.path.join("data", "dailymed_interactions.csv")
OUT_MODEL = "ddi_baseline_model.joblib"
OUT_VECT = "ddi_baseline_vectorizer.joblib"

print("Loading", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Prepare text: prefer interaction_text_snippet, fallback to section_heading, include dosage_text
def make_text(row):
    parts = []
    if pd.notna(row.get("interaction_text_snippet")):
        parts.append(str(row["interaction_text_snippet"]))
    elif pd.notna(row.get("section_heading")):
        parts.append(str(row["section_heading"]))
    if pd.notna(row.get("dosage_text")):
        parts.append(str(row["dosage_text"]))
    return " . ".join(parts).strip()

df["text"] = df.apply(make_text, axis=1)

# Create binary label from confidence (adjust threshold if you prefer)
df["label"] = (df.get("confidence", 0.0) >= 0.8).astype(int)
print("Total rows:", len(df), " Positive labels:", int(df["label"].sum()))

# Drop rows with no text
df = df[df["text"].str.strip().astype(bool)]
print("After dropping empty text rows:", len(df))

# Train / test split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print("Train size:", len(train), "Test size:", len(test))

# Build pipeline
vect = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga")
pipe = make_pipeline(vect, clf)

# Fit
print("Fitting TF-IDF + LogisticRegression ...")
pipe.fit(train["text"], train["label"])

# Evaluate
y_pred = pipe.predict(test["text"])
y_true = test["label"].values
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1:", f1_score(y_true, y_pred, zero_division=0))
print("\nClassification report:\n", classification_report(y_true, y_pred, zero_division=0))

# Save model and vectorizer
print("Saving model ->", OUT_MODEL)
joblib.dump(pipe, OUT_MODEL)
print("Done.")
