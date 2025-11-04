import os
import gzip
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import warnings

# --------------------------
# CONFIGURATION
# --------------------------
TASK1_DIR = "task1"
JSON_FILE = os.path.join(TASK1_DIR, "dataset0.json.gz")
INFO_LABEL_FILE = os.path.join(TASK1_DIR, "data.info.labelled")
PARSED_CSV_OUTPUT = os.path.join(TASK1_DIR, "parsed_dataset0.csv")
CHUNK_SIZE = 1000  # smaller chunk size

# Ensure files exist
if not os.path.isdir(TASK1_DIR):
    raise FileNotFoundError(f"Directory '{TASK1_DIR}' not found.")
if not os.path.exists(JSON_FILE):
    raise FileNotFoundError(f"JSON file not found: {JSON_FILE}")
if not os.path.exists(INFO_LABEL_FILE):
    raise FileNotFoundError(f"Info/label file not found: {INFO_LABEL_FILE}")

# --------------------------
# STEP 1: Parse dataset0.json.gz in small chunks
# --------------------------
rows = []
parsed_count = 0
with gzip.open(JSON_FILE, "rt") as f:
    buffer = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        for transcript_id, positions in obj.items():
            for position, kmers in positions.items():
                try:
                    pos_int = int(position)
                except Exception:
                    continue
                for kmer, feature_lists in kmers.items():
                    for feats in feature_lists:
                        if isinstance(feats, (list, tuple)) and len(feats) == 9:
                            buffer.append({
                                "transcript_id": transcript_id,
                                "position": pos_int,
                                "kmer": kmer,
                                "feat1": feats[0],
                                "feat2": feats[1],
                                "feat3": feats[2],
                                "feat4": feats[3],
                                "feat5": feats[4],
                                "feat6": feats[5],
                                "feat7": feats[6],
                                "feat8": feats[7],
                                "feat9": feats[8]
                            })
                            parsed_count += 1
        # Periodically convert buffer to df and append to disk to avoid memory issues
        if len(buffer) >= CHUNK_SIZE:
            df_chunk = pd.DataFrame(buffer)
            if parsed_count == len(buffer):  # first write (overwrite)
                df_chunk.to_csv(PARSED_CSV_OUTPUT, index=False)
            else:  # append
                df_chunk.to_csv(PARSED_CSV_OUTPUT, mode='a', header=False, index=False)
            buffer = []
if buffer:
    df_chunk = pd.DataFrame(buffer)
    if parsed_count == len(buffer):
        df_chunk.to_csv(PARSED_CSV_OUTPUT, index=False)
    else:
        df_chunk.to_csv(PARSED_CSV_OUTPUT, mode='a', header=False, index=False)

print(f"Parsed {parsed_count} feature rows saved to {PARSED_CSV_OUTPUT}")

# --------------------------
# STEP 2: Read data.info.labelled in chunks and clean
# --------------------------
info_chunks = []
for chunk in pd.read_csv(INFO_LABEL_FILE, chunksize=CHUNK_SIZE):
    # rename column for merge
    chunk = chunk.rename(columns={"transcript_position": "position"})
    chunk["position"] = pd.to_numeric(chunk["position"], errors="coerce")
    chunk = chunk.dropna(subset=["position"])
    chunk["position"] = chunk["position"].astype(int)
    chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce").astype("Int64")
    info_chunks.append(chunk)

info_df = pd.concat(info_chunks, ignore_index=True)
print("Info DataFrame shape (after cleaning):", info_df.shape)

# --------------------------
# STEP 3: Merge parsed data and info file in chunks to avoid memory blowup
# --------------------------
merged_chunks = []
for parsed_chunk in pd.read_csv(PARSED_CSV_OUTPUT, chunksize=CHUNK_SIZE):
    merged_chunk = parsed_chunk.merge(
        info_df[["gene_id", "transcript_id", "position", "label"]],
        on=["transcript_id", "position"],
        how="left",
        validate="m:1"
    )
    merged_chunks.append(merged_chunk)

merged_df = pd.concat(merged_chunks, ignore_index=True)
print("Merged DataFrame shape:", merged_df.shape)
num_with_info = merged_df["gene_id"].notna().sum()
print(f"Rows with matching gene_id/label from info file: {num_with_info} / {len(merged_df)}")

# Save merged
merged_df.to_csv(PARSED_CSV_OUTPUT, index=False)
print(f"Saved merged dataframe to: {PARSED_CSV_OUTPUT}")

# --------------------------
# STEP 4: Split genes into train/test with stratify, process in chunks
# --------------------------
gene_labels = merged_df.groupby('gene_id')['label'].agg(lambda x: x.mode()[0]).reset_index()

train_genes, test_genes = train_test_split(
    gene_labels['gene_id'],
    test_size=0.2,
    stratify=gene_labels['label'],
    random_state=42
)

train_rows = []
test_rows = []

for chunk in pd.read_csv(PARSED_CSV_OUTPUT, chunksize=CHUNK_SIZE):
    train_chunk = chunk[chunk['gene_id'].isin(train_genes)]
    test_chunk = chunk[chunk['gene_id'].isin(test_genes)]
    train_rows.append(train_chunk)
    test_rows.append(test_chunk)

train_df = pd.concat(train_rows, ignore_index=True)
test_df = pd.concat(test_rows, ignore_index=True)

train_df.to_csv("task1/train_set.csv", index=False)
test_df.to_csv("task1/test_set.csv", index=False)

overlap = set(train_df['gene_id']).intersection(set(test_df['gene_id']))
if len(overlap) == 0:
    print("Check passed ✅: No gene_ids in test are in train.")
else:
    print("Check failed ❌: Overlapping gene_ids found:", overlap)

# --------------------------
# STEP 5: Aggregate by mean for each group in chunks
# --------------------------
def aggregate_by_mean(df, group_cols, agg_cols, chunksize=CHUNK_SIZE):
    agg_dfs = []
    for chunk in (df[i:i+chunksize] for i in range(0, len(df), chunksize)):
        agg_chunk = chunk.groupby(group_cols)[agg_cols].mean().reset_index()
        agg_dfs.append(agg_chunk)
    return pd.concat(agg_dfs, ignore_index=True)

feature_cols = [f"feat{i}" for i in range(1, 10)] + ["label"]

train_df = aggregate_by_mean(train_df, ['transcript_id','position','kmer','gene_id'], feature_cols)
test_df = aggregate_by_mean(test_df, ['transcript_id','position','kmer','gene_id'], feature_cols)

full_df = pd.concat([train_df, test_df], ignore_index=True)

# --------------------------
# STEP 6: Prepare training/testing data
# --------------------------
X_train = train_df[feature_cols[:-1]]
y_train = train_df["label"]
X_test = test_df[feature_cols[:-1]]
y_test = test_df["label"]

# --------------------------
# STEP 7: Train XGBoost with GridSearchCV
# --------------------------
warnings.filterwarnings("ignore")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=21
)
param_grid = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best AUCROC: {:.4f}".format(grid_search.best_score_))
print("Best Parameters:", grid_search.best_params_)

# --------------------------
# STEP 8: Fit best model on full data
# --------------------------
best_params = grid_search.best_params_
full_model = xgb.XGBClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=21
)

X_full = full_df[feature_cols[:-1]]
y_full = full_df['label']

full_model.fit(X_full, y_full)

y_score = full_model.predict_proba(X_full)[:,1]
auc_score = roc_auc_score(y_full, y_score)
precision, recall, thresholds = precision_recall_curve(y_full, y_score)
auc_precision_recall = auc(recall, precision)

print(f"Full data ROC AUC: {auc_score:.4f}")
print(f"Full data PR AUC: {auc_precision_recall:.4f}")

joblib.dump(full_model, "full_xgb_model.pkl")

# --------------------------
# STEP 9: Prediction function using chunks
# --------------------------
def predict_new_data(input_csv, chunk_size=CHUNK_SIZE):
    model = joblib.load("full_xgb_model.pkl")
    results = []

    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        grouped = chunk.groupby(["transcript_id", "position"], as_index=False).mean()
        X_new = grouped[feature_cols[:-1]]
        preds = model.predict_proba(X_new)[:,1]
        chunk_res = pd.DataFrame({
            "transcript_id": grouped["transcript_id"],
            "transcript_position": grouped["position"],
            "score": preds
        })
        results.append(chunk_res)

    final_res = pd.concat(results, ignore_index=True)
    os.makedirs("xgboost_predictions", exist_ok=True)
    out_path = f"xgboost_predictions/predictions_{os.path.basename(input_csv)}"
    final_res.to_csv(out_path, index=False)
    print(f"Predictions saved to: {out_path}")

# --------------------------
# Run predictions for files
# --------------------------
files = ["task1/parsed_dataset2.csv"]
for f in files:
    print(f"🔹 Running predictions on {f}...")
    predict_new_data(f)

