%pip install -r requirements.txt


import os
import json
import csv
from tqdm import tqdm  # for progress bar

# -----------------------------
# Settings
# -----------------------------
BASE_DIR = "rna_modification_data"
OUTPUT_DIR = "prepared_data"
CHUNK_SIZE = 5000  # number of lines to process at once
os.makedirs(OUTPUT_DIR, exist_ok=True)

skipped_files = []

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
# Process all folders
# -----------------------------
for folder in sorted(os.listdir(BASE_DIR)):
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    json_path = os.path.join(folder_path, "data.json")
    if not os.path.exists(json_path):
        print(f"⚠️ Missing data.json in {folder}")
        skipped_files.append(folder)
        continue

    output_path = os.path.join(OUTPUT_DIR, f"{folder}.csv")

    try:
        with open(json_path, "r") as f_in, open(output_path, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=[
                "transcript_id", "position", "kmer",
                *[f"feat{i+1}" for i in range(9)]
            ])
            writer.writeheader()

            batch = []
            for i, line in enumerate(tqdm(f_in, desc=f"Processing {folder}")):
                try:
                    obj = json.loads(line)
                    batch.extend(flatten_json_line(obj))
                except Exception as e:
                    print(f"⚠️ Error parsing line {i} in {folder}: {e}")
                    continue

                # write batch to CSV and clear memory
                if len(batch) >= CHUNK_SIZE:
                    writer.writerows(batch)
                    batch.clear()

            # write any remaining data
            if batch:
                writer.writerows(batch)

        print(f"✅ Finished {folder}, saved to {output_path}")

    except Exception as e:
        print(f"❌ Failed to process {folder}: {e}")
        skipped_files.append(folder)

# -----------------------------
# Summary
# -----------------------------
print("\n=== Summary ===")
if skipped_files:
    print("Skipped JSONs:")
    for name in skipped_files:
        print("-", name)
else:
    print("All JSON files processed successfully.")


import os
import joblib
import pandas as pd

def predict_new_data_chunked(input_csv, chunksize=10000):
    # Load model once
    model = joblib.load("full_xgb_model.pkl")
    
    feature_cols = [f"feat{i}" for i in range(1, 10)]
    
    # Prepare output folder
    out_folder = "predictions"
    os.makedirs(out_folder, exist_ok=True)
    
    # Prepare output file path
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    out_path = os.path.join(out_folder, f"{base_name}_prediction.csv")
    
    # Remove existing output file if exists (to overwrite fresh)
    if os.path.exists(out_path):
        os.remove(out_path)
    
    # Process input CSV in chunks
    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunksize)):
        # Group by transcript_id, position, kmer; aggregate feature columns by mean
        agg_funcs = {col: "mean" for col in feature_cols}
        grouped = chunk.groupby(["transcript_id", "position", "kmer"], as_index=False).agg(agg_funcs)
        
        # Extract features for prediction
        X_new_features = grouped[feature_cols]
        
        # Predict probabilities for positive class
        preds = model.predict_proba(X_new_features)[:, 1]
        
        # Create prediction DataFrame
        result = pd.DataFrame({
            "transcript_id": grouped["transcript_id"],
            "transcript_position": grouped["position"],
            "kmer": grouped["kmer"],
            "score": preds
        })
        
        # Append chunk predictions to CSV file
        if i == 0:
            result.to_csv(out_path, index=False)
        else:
            result.to_csv(out_path, mode='a', header=False, index=False)
        
        
    
    print(f"Predictions saved to: {out_path}")

# Example to run over all CSV files in prepared_data folder:
input_folder = "prepared_data"
files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

for f in files:
    file_path = os.path.join(input_folder, f)
    print(f"🔹 Running predictions on {file_path}...")
    predict_new_data_chunked(file_path)


