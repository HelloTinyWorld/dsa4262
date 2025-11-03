import os
import gzip
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc
import numpy as np
import warnings
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import joblib



# --------------------------
# CONFIGURATION
# --------------------------
TASK1_DIR = "task1"
JSON_FILE = os.path.join(TASK1_DIR, "dataset0.json.gz")
INFO_LABEL_FILE = os.path.join(TASK1_DIR, "data.info.labelled")
PARSED_CSV_OUTPUT = os.path.join(TASK1_DIR, "parsed_dataset0.csv")

# Ensure task1 exists and files are present
if not os.path.isdir(TASK1_DIR):
    raise FileNotFoundError(f"Directory '{TASK1_DIR}' not found.")
if not os.path.exists(JSON_FILE):
    raise FileNotFoundError(f"JSON file not found: {JSON_FILE}")
if not os.path.exists(INFO_LABEL_FILE):
    raise FileNotFoundError(f"Info/label file not found: {INFO_LABEL_FILE}")

# --------------------------
# STEP 1: PARSE dataset0.json.gz INTO ROWS
# --------------------------
rows = []
parsed_count = 0
with gzip.open(JSON_FILE, "rt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # Each line is a JSON object mapping transcript_id -> positions
        for transcript_id, positions in obj.items():
            for position, kmers in positions.items():
                # position in JSON may be string or number; keep as int
                try:
                    pos_int = int(position)
                except Exception:
                    # skip malformed position
                    continue
                for kmer, feature_lists in kmers.items():
                    for feats in feature_lists:
                        # each feats should be list of length 9
                        if isinstance(feats, (list, tuple)) and len(feats) == 9:
                            rows.append({
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

print(f"Parsed {parsed_count} feature rows from {JSON_FILE}")

# Convert to DataFrame
df = pd.DataFrame(rows)
print("Parsed DataFrame shape:", df.shape)
if df.shape[0] == 0:
    raise ValueError("No rows parsed from JSON. Check the JSON file and parsing logic.")

# --------------------------
# STEP 2: READ data.info.labelled (CSV with header)
# Expect header: gene_id,transcript_id,transcript_position,label
# --------------------------
# Read with pandas; it will automatically parse the header if present.
info_df = pd.read_csv(INFO_LABEL_FILE)

# Validate expected columns exist
expected_cols = {"gene_id", "transcript_id", "transcript_position", "label"}
if not expected_cols.issubset(set(info_df.columns)):
    raise ValueError(
        f"Info file columns mismatch. Expected at least {expected_cols}; found {set(info_df.columns)}"
    )

# Rename the position column to 'position' to match parsed df
info_df = info_df.rename(columns={"transcript_position": "position"})

# Coerce position to numeric, drop invalid rows
info_df["position"] = pd.to_numeric(info_df["position"], errors="coerce")
before_drop = len(info_df)
info_df = info_df.dropna(subset=["position"])
dropped = before_drop - len(info_df)
if dropped:
    print(f"Warning: Dropped {dropped} rows from info file due to invalid 'position' values.")
# convert to int now (safe)
info_df["position"] = info_df["position"].astype(int)

# Ensure label is integer if possible
if info_df["label"].dtype != int:
    info_df["label"] = pd.to_numeric(info_df["label"], errors="coerce").astype("Int64")

print("Info DataFrame shape (after cleaning):", info_df.shape)

# --------------------------
# STEP 3: MERGE parsed JSON data with info_df
# --------------------------
# Merge on transcript_id and position
merged = df.merge(
    info_df[["gene_id", "transcript_id", "position", "label"]],
    on=["transcript_id", "position"],
    how="left",
    validate="m:1"  # many parsed rows can map to single info row
)

print("Merged DataFrame shape:", merged.shape)

# Report merge stats
num_with_info = merged["gene_id"].notna().sum()
print(f"Rows with matching gene_id/label from info file: {num_with_info} / {len(merged)}")

# --------------------------
# STEP 4: SAVE RESULT
# --------------------------
merged.to_csv(PARSED_CSV_OUTPUT, index=False)
print(f"Saved merged dataframe to: {PARSED_CSV_OUTPUT}")


df = merged


import pandas as pd
from sklearn.model_selection import train_test_split

# Assume df has 'gene_id' and 'label' columns

# First, get unique gene_ids with their label (majority label for that gene)
gene_labels = df.groupby('gene_id')['label'].agg(lambda x: x.mode()[0]).reset_index()

# Split gene_ids into train/test while stratifying by label
train_genes, test_genes = train_test_split(
    gene_labels['gene_id'],
    test_size=0.2,                # 20% genes for test
    stratify=gene_labels['label'],
    random_state=42
)

# Select rows corresponding to train/test genes
train_df = df[df['gene_id'].isin(train_genes)].reset_index(drop=True)
test_df = df[df['gene_id'].isin(test_genes)].reset_index(drop=True)

# Save to CSV
train_df.to_csv("task1/train_set.csv", index=False)
test_df.to_csv("task1/test_set.csv", index=False)


# Find overlapping gene_ids
overlap_genes = set(train_df['gene_id']).intersection(set(test_df['gene_id']))

if len(overlap_genes) == 0:
    print("Check passed ✅: No gene_ids in test are in train.")
else:
    print("Check failed ❌: Overlapping gene_ids found:", overlap_genes)

train_df = pd.read_csv("task1/train_set.csv")
test_df = pd.read_csv("task1/test_set.csv")

train_df = train_df.groupby(['transcript_id','position','kmer','gene_id']).mean(['feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','label']).reset_index()
test_df = test_df.groupby(['transcript_id','position','kmer','gene_id']).mean(['feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','label']).reset_index()

full_df = pd.concat([train_df,test_df])
len(full_df[full_df['label'] == 0])/len(full_df[full_df['label'] == 1])

feature_cols = [f"feat{i}" for i in range(1, 10)]
X_train = train_df[feature_cols]
y_train = train_df["label"]
X_test = test_df[feature_cols]
y_test = test_df["label"]



warnings.filterwarnings("ignore")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss',  # base metric for training
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
    cv=5,                  # 5-fold cross-validation
    verbose=2,
    n_jobs=-1              # use all CPU cores
)
grid_search.fit(X_train, y_train)

print("Best AUCROC: {:.4f}".format(grid_search.best_score_))
print("Best Parameters:", grid_search.best_params_)

# Retrieve best model (Best Parameters: {'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 500})

full_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=21
)

feature_cols = [f"feat{i}" for i in range(1, 10)]
x_full = full_df[feature_cols]
y_full = full_df['label']
full_model.fit(x_full, y_full)

y_score = full_model.predict_proba(x_full)[:,1]
auc_score = roc_auc_score(y_full, y_score)
auc_score

precision, recall, thresholds = precision_recall_curve(y_full, y_score)
# Use AUC function to calculate the area under the curve of precision recall curve
auc_precision_recall = auc(recall, precision)
auc_precision_recall



# Save the model to a file
joblib.dump(full_model, "full_xgb_model.pkl")



def predict_new_data(input_csv):
    # Load the model (no encoder needed now)
    model = joblib.load("full_xgb_model.pkl")
    
    # Load and prepare data (no label, no gene_id, drop kmer)
    df = pd.read_csv(input_csv)
    
    # Pool features by mean, grouped by transcript_id and position
    feature_cols = [f"feat{i}" for i in range(1, 10)]
    agg_funcs = {col: "mean" for col in feature_cols}
    
    grouped = df.groupby(["transcript_id", "position"], as_index=False).agg(agg_funcs)
    
    # Features for prediction
    X_new = grouped[feature_cols]
    
    # Predict probabilities for positive class
    preds = model.predict_proba(X_new)[:, 1]
    
    # Format output
    result = pd.DataFrame({
        "transcript_id": grouped["transcript_id"],
        "transcript_position": grouped["position"],
        "score": preds
    })
    
    # Save output CSV
    os.makedirs("xgboost_predictions", exist_ok=True)
    out_path = f"xgboost_predictions/predictions_{os.path.basename(input_csv)}"
    result.to_csv(out_path, index=False)
    print(f"Predictions saved to: {out_path}")


# List of input files
files = ["task1/parsed_dataset2.csv"]

# Run predictions for each file
for f in files:
    print(f"🔹 Running predictions on {f}...")
    predict_new_data(f)

