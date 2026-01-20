# ============================================================
# Reproducible Evaluation — GPT-3.5 Fine-tuned Polymer Model (Nc)
# ============================================================

import os, json, time, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from openai import OpenAI
from datetime import datetime

# ---------- CONFIG ----------
os.environ["OPENAI_API_KEY"] = ""  # 🔒 replace with your valid key

MODEL_ID = "ft:gpt-3.5-turbo-0125:personal:"  # your fine-tuned model ID
VAL_PATH = "./nc_finetune_ready/Nc_val.csv"            # unseen validation data
SCALER_INFO = "./nc_finetune_ready/scaler_info.json"   # normalization range
OUT_DIR = "./nc_gpt_eval_reproducible"
os.makedirs(OUT_DIR, exist_ok=True)

TEMP = 0.3
SEED = 42
np.random.seed(SEED)
client = OpenAI()

# ---------- 1️⃣ LOAD DATA ----------
print("📂 Loading validation data...")
df = pd.read_csv(VAL_PATH)
df = df.dropna(subset=["smiles", "value_scaled"]).reset_index(drop=True)
with open(SCALER_INFO) as f: s = json.load(f)
vmin, vmax = s["min"], s["max"]
print(f"✅ Loaded {len(df)} samples | range = [{vmin:.3f}, {vmax:.3f}]")

# ---------- 2️⃣ QUERY MODEL ----------
preds, trues, smiles_list = [], [], []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating model"):
    smiles = row.smiles.strip()
    true_val = float(row.value_scaled)
    prompt = f"Predict the Refractive index (Nc) for the polymer with SMILES: {smiles}."
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            seed=SEED,
            temperature=TEMP,
            max_tokens=10,
            messages=[
                {"role": "system",
                 "content": "You are a chemistry assistant that predicts Refractive index (Nc) numerically between 0 and 1 (normalized)."},
                {"role": "user", "content": prompt}
            ],
        )
        val = float(resp.choices[0].message.content.strip().split()[0])
    except Exception as e:
        print(f"⚠️ {smiles[:20]}... error: {e}")
        val = np.nan
    preds.append(val); trues.append(true_val); smiles_list.append(smiles)
    time.sleep(0.25)

# ---------- 3️⃣ METRICS ----------
mask = ~np.isnan(preds)
preds, trues, smiles_list = np.array(preds)[mask], np.array(trues)[mask], np.array(smiles_list)[mask]
rmse_norm = np.sqrt(mean_squared_error(trues, preds))
r2_norm = r2_score(trues, preds)
preds_real = preds * (vmax - vmin) + vmin
trues_real = trues * (vmax - vmin) + vmin
rmse_real = np.sqrt(mean_squared_error(trues_real, preds_real))
r2_real = r2_score(trues_real, preds_real)

# ---------- 4️⃣ SAVE REPRODUCIBLE RESULTS ----------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
meta = {
    "model_id": MODEL_ID,
    "temperature": TEMP,
    "seed": SEED,
    "val_samples": len(df),
    "timestamp": timestamp,
    "rmse_norm": rmse_norm,
    "r2_norm": r2_norm,
    "rmse_real": rmse_real,
    "r2_real": r2_real,
}

pd.DataFrame({
    "smiles": smiles_list,
    "true_norm": trues,
    "pred_norm": preds,
    "true_real": trues_real,
    "pred_real": preds_real
}).to_csv(f"{OUT_DIR}/Nc_predictions_{timestamp}.csv", index=False)
json.dump(meta, open(f"{OUT_DIR}/Nc_meta_{timestamp}.json", "w"), indent=2)

print("\n📊 Reproducible Evaluation Results")
print("──────────────────────────────────────────────")
print(f"Normalized RMSE = {rmse_norm:.4f}")
print(f"Normalized R²   = {r2_norm:.4f}")
print(f"Denormalized RMSE = {rmse_real:.4f}")
print(f"Denormalized R²   = {r2_real:.4f}")
print("──────────────────────────────────────────────")
print(f"✅ Saved artifacts → {OUT_DIR}/Nc_predictions_{timestamp}.csv")