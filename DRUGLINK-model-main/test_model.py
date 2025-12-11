import joblib
from pathlib import Path

print("🔍 Checking model file...")

model_path = Path("models/ddi_baseline_model.joblib")

if not model_path.exists():
    print("❌ Model file not found at:", model_path)
    exit()

print("✅ Model file found at:", model_path)
print("📦 Loading model...")


model = joblib.load(model_path)

print("\n🔮 Enter drug interaction samples (type 'done' to finish):")
samples = []
while True:
    user_input = input("Enter sample: ").strip()
    if user_input.lower() == 'done':
        break
    if user_input:
        samples.append(user_input)

if not samples:
    print("❌ No samples provided. Exiting...")
    exit()

print("\n🔮 Making predictions...")
predictions = model.predict(samples)

print("\n=============================")
print("INPUT:", samples)
print("OUTPUT:", predictions)
print("=============================\n")
