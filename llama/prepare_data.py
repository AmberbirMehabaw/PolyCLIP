# prepare_data.py
# ============================================================
# LLaMA Regression Baseline (Step 1): Prepare data
# - Load CSV (smiles,value)
# - Canonicalize + deduplicate
# - Train/val split (seeded) + hash logging
# - Train-only Z-score normalization
# - Save: train.csv, val.csv, norm_info.json, split_hashes.json
# ============================================================

import os, math, json, hashlib, warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import MolToSmiles

warnings.filterwarnings("ignore")

# ===================== USER CONFIG =====================
PROPERTY = "Eea"                 # "Nc", "Eea", "Ei"
CSV_PATH = "./data/Eea.csv"      # "./data/Nc.csv", "./data/Eea.csv", "./data/Ei.csv"
OUT_DIR  = f"./llama_reghead_{PROPERTY.lower()}"

SEED = 42
TEST_SIZE = 0.20
# =======================================================

# -------------------- Offline / caches --------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- Repro --------------------
np.random.seed(SEED)

def sha1_of_series(vals) -> str:
    h = hashlib.sha1()
    for v in vals:
        h.update(str(v).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def canonicalize_smiles(s: str):
    try:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            return None
        return MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def train_val_split(df: pd.DataFrame, test_size: float, seed: int):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(len(df) * test_size)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

# ============================================================
# 1) LOAD + CLEAN
# ============================================================
df = pd.read_csv(CSV_PATH).dropna(subset=["smiles", "value"]).reset_index(drop=True)
df["smiles"] = df["smiles"].astype(str).apply(canonicalize_smiles)
df = df.dropna(subset=["smiles"]).drop_duplicates("smiles").reset_index(drop=True)
df["value"] = df["value"].astype(float)

print("================================================")
print("✅ Property:", PROPERTY)
print(f"✅ Clean samples: {len(df)}")
print(f"✅ Value range: [{df['value'].min():.6f}, {df['value'].max():.6f}]")
print("================================================")

train_df, val_df = train_val_split(df, TEST_SIZE, SEED)
train_hash = sha1_of_series(train_df["smiles"].tolist())
val_hash   = sha1_of_series(val_df["smiles"].tolist())

print(f"🔎 train hash: {train_hash}")
print(f"🔎 val   hash: {val_hash}")

# -------------------- Train-only Z-score --------------------
mu = float(train_df["value"].mean())
sigma = float(train_df["value"].std()) + 1e-8
train_df["y"] = (train_df["value"] - mu) / sigma
val_df["y"]   = (val_df["value"] - mu) / sigma

# Save normalization + split metadata
json.dump(
    {"mu": mu, "sigma": sigma, "seed": SEED, "test_size": TEST_SIZE, "property": PROPERTY, "csv_path": CSV_PATH},
    open(f"{OUT_DIR}/norm_info.json", "w"),
    indent=2
)
json.dump(
    {"train_hash": train_hash, "val_hash": val_hash, "n_train": len(train_df), "n_val": len(val_df)},
    open(f"{OUT_DIR}/split_hashes.json", "w"),
    indent=2
)

print(f"✅ Z-score: mu={mu:.6f}, sigma={sigma:.6f}")

# Save prepared splits
train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
print(f"✅ Saved → {OUT_DIR}/train.csv")
print(f"✅ Saved → {OUT_DIR}/val.csv")
print(f"✅ Saved → {OUT_DIR}/norm_info.json")
print(f"✅ Saved → {OUT_DIR}/split_hashes.json")
