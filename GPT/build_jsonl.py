# ============================================================
# Polymer Property Fine-tuning Data Preparation (Nc example)
# Enhanced version — reproducible, validated, and robust
# ============================================================

import os, json, warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
CSV_PATH = "Nc.csv"                     # raw file with "smiles" and "value"
OUT_DIR  = "./nc_finetune_ready"        # folder for output
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ============================================================
# 1️⃣ LOAD AND CLEAN DATA
# ============================================================
df = pd.read_csv(CSV_PATH)
print(f"🔹 Raw entries: {len(df)}")

# Drop rows with missing or invalid entries
df = df.dropna(subset=["smiles", "value"]).reset_index(drop=True)

# Validate SMILES (ensures chemical validity)
def is_valid_smiles(s):
    try:
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return False

df = df[df["smiles"].apply(is_valid_smiles)].reset_index(drop=True)
print(f"✅ Valid SMILES retained: {len(df)}")

# Remove potential duplicates
df = df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
print(f"✅ Unique SMILES retained: {len(df)}")

# ============================================================
# 2️⃣ NORMALIZE PROPERTY VALUES TO [0, 1]
# ============================================================
scaler = MinMaxScaler()
df["value_scaled"] = scaler.fit_transform(df[["value"]])
vmin, vmax = float(scaler.data_min_[0]), float(scaler.data_max_[0])
print(f"✅ Normalized Nc between {vmin:.4f} and {vmax:.4f}")

# Save cleaned numeric version
clean_path = f"{OUT_DIR}/Nc_clean.csv"
df.to_csv(clean_path, index=False)
print(f"💾 Saved cleaned dataset → {clean_path}")

# ============================================================
# 3️⃣ TRAIN / VALIDATION SPLIT (80 / 20) WITH STRATIFICATION
# ============================================================
# Stratify to preserve property value distribution
df["bins"] = pd.qcut(df["value_scaled"], q=min(10, len(df)//5), duplicates="drop", labels=False)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["bins"])
train_df, val_df = train_df.drop(columns="bins"), val_df.drop(columns="bins")
print(f"✅ Split complete → Train: {len(train_df)} | Validation: {len(val_df)}")

# Save split CSVs before JSONL conversion
train_csv = f"{OUT_DIR}/Nc_train.csv"
val_csv = f"{OUT_DIR}/Nc_val.csv"
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
print(f"💾 Saved train split → {train_csv}")
print(f"💾 Saved validation split → {val_csv}")

# ============================================================
# 4️⃣ BUILD JSONL FILES FOR OPENAI SUPERVISED FINE-TUNING
# ============================================================
def make_jsonl(dataframe, path, property_name="Nc"):
    """Create Chat-style JSONL file for OpenAI fine-tuning."""
    with open(path, "w") as f:
        for _, row in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc=f"Building {os.path.basename(path)}"
        ):
            prompt = (
                f"Predict the refractive index ({property_name}) for the polymer "
                f"with SMILES: {row.smiles}."
            )
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"You are a chemistry assistant that predicts polymer refractive "
                            f"index ({property_name}) values numerically between 0 and 1 (normalized)."
                        )
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": str(round(float(row.value_scaled), 6))}
                ]
            }
            f.write(json.dumps(record) + "\n")
    print(f"✅ Saved: {path}")

# Build both JSONL files
train_json = f"{OUT_DIR}/train_Nc.jsonl"
val_json = f"{OUT_DIR}/val_Nc.jsonl"
make_jsonl(train_df, train_json, "Nc")
make_jsonl(val_df,   val_json,   "Nc")

# ============================================================
# 5️⃣ SAVE NORMALIZATION RANGE
# ============================================================
scaler_info = {"min": vmin, "max": vmax, "samples": len(df)}
with open(f"{OUT_DIR}/scaler_info.json", "w") as f:
    json.dump(scaler_info, f, indent=2)
print(f"📊 Normalization info saved → {OUT_DIR}/scaler_info.json")

# ============================================================
# ✅ SUMMARY
# ============================================================
print("\n🎯 All files ready for OpenAI fine-tuning:")
print(f"   • Training JSONL   → {train_json}")
print(f"   • Validation JSONL → {val_json}")
print(f"   • Training CSV     → {train_csv}")
print(f"   • Validation CSV   → {val_csv}")
print(f"   • Clean CSV        → {clean_path}")
print(f"   • Scaler info      → {OUT_DIR}/scaler_info.json")
print("\n✨ Ready for upload to OpenAI fine-tuning dashboard!")