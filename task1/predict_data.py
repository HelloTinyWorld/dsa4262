import os
import json
import csv
import gzip
from tqdm import tqdm
import pandas as pd
import joblib

# -----------------------------
# Settings
# -----------------------------
INPUT_DIR = "input data"
TEMP_DIR = "temp_cleaned"
OUTPUT_DIR = "output_predictions"
CHUNK_SIZE = 5000  # for JSON flattening
PRED_CHUNK_SIZE = 10000  # for prediction chunking
MODEL_PATH = "full_xgb_model.pkl"

# -----------------------------
# Check input directory
# -----------------------------
if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(f"Input folder '{INPUT_DIR}' does not exist.")

# Find all .json or .json.gz files in input folder
json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json") or f.endswith(".json.gz")]

if not json_files:
    raise FileNotFoundError(f"No .json or .json.gz files found in '{INPUT_DIR}'.")

# -----------------------------
# Helper function to flatten one JSON object
# -----------------------------
def flatten_json_line(obj):
    rows = []
    for transcript_id, positions in obj.items():
        for position, kmers in positions.items():
            for kmer, feature_lists in kmers.items():
                for feats in feature_lists:
                    if len(feats) == 9:
                        rows.append({
                            "transcript_id": transcript_id,
                            "position": position,
                            "kmer": kmer,
                            **{f"feat{i+1}": feats[i] for i in range(9)}
                        })
    return rows

# -----------------------------
# Load model once
# -----------------------------
model = joblib.load(MODEL_PATH)
feature_cols = [f"feat{i}" for i in range(1, 10)]

# -----------------------------
# Process each JSON file
# -----------------------------
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for json_filename in json_files:
    print(f"\nProcessing file '{json_filename}' ...")

    input_path = os.path.join(INPUT_DIR, json_filename)
    cleaned_csv_path = os.path.join(TEMP_DIR, f"{os.path.splitext(json_filename)[0]}.csv")
    prediction_out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(json_filename)[0]}_prediction.csv")

    # Step 1: Flatten JSON to CSV
    open_func = gzip.open if json_filename.endswith(".gz") else open

    with open_func(input_path, "rt") as f_in, open(cleaned_csv_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[
            "transcript_id", "position", "kmer", *[f"feat{i+1}" for i in range(9)]
        ])
        writer.writeheader()

        batch = []
        for i, line in enumerate(tqdm(f_in, desc=f"Flattening {json_filename}")):
            try:
                obj = json.loads(line)
                batch.extend(flatten_json_line(obj))
            except Exception as e:
                print(f"⚠️ Error parsing line {i}: {e}")
                continue

            if len(batch) >= CHUNK_SIZE:
                writer.writerows(batch)
                batch.clear()

        if batch:
            writer.writerows(batch)

    print(f"Flattened data saved to '{cleaned_csv_path}'")

    # Step 2: Predict on cleaned CSV
    if os.path.exists(prediction_out_path):
        print(f"Warning: Output prediction file '{prediction_out_path}' exists and will be overwritten.")
        os.remove(prediction_out_path)

    print(f"Running chunked prediction on cleaned data for '{json_filename}'...")

    for i, chunk in enumerate(pd.read_csv(cleaned_csv_path, chunksize=PRED_CHUNK_SIZE)):
        agg_funcs = {col: "mean" for col in feature_cols}
        grouped = chunk.groupby(["transcript_id", "position", "kmer"], as_index=False).agg(agg_funcs)

        X_new_features = grouped[feature_cols]

        preds = model.predict_proba(X_new_features)[:, 1]

        result = pd.DataFrame({
            "transcript_id": grouped["transcript_id"],
            "transcript_position": grouped["position"],
            "kmer": grouped["kmer"],
            "score": preds
        })

        if i == 0:
            result.to_csv(prediction_out_path, index=False)
        else:
            result.to_csv(prediction_out_path, mode='a', header=False, index=False)

    print(f"Predictions saved to '{prediction_out_path}'")

print("\nAll files processed.")
