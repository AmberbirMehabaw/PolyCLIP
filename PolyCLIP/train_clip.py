# finetune_clip.py
# ============================================================
# CLIP Baseline (Step 2): Leak-free K-fold fine-tuning + eval
# - Load cached dataset {PROPERTY}_with_images.csv
# - Per-fold normalization (StandardScaler fit on train fold only)
# - Fine-tune last transformer layers + regression head
# - Report fold metrics + final mean±std
# ============================================================

import os, random, hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from transformers import CLIPProcessor, CLIPModel, get_linear_schedule_with_warmup

# ----------------- CONFIG -----------------
DATA_PATH = "./data"
PROPERTY = "Nc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

EPOCHS = 30
KFOLDS = 5
BATCH_SIZE = 8
PATIENCE = 7

LR_CLIP = 1e-5
LR_HEAD = 1e-4
WARMUP_RATIO = 0.1

CLIP_DIR = "./models/clip-vit-base-patch32"
# ------------------------------------------

clip_processor = CLIPProcessor.from_pretrained(CLIP_DIR)

# ============================================================
# Dataset + Collate
# ============================================================
class CLIPDataset(Dataset):
    def __init__(self, smiles, image_paths, values):
        self.smiles = smiles
        self.image_paths = image_paths
        self.values = values

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.tensor(self.values[idx], dtype=torch.float32)

        enc = clip_processor(
            text=[smi],
            images=[img],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return enc, label.unsqueeze(-1)

def collate_fn(batch):
    texts = [enc["input_ids"] for enc, _ in batch]
    masks = [enc["attention_mask"] for enc, _ in batch]
    images = [enc["pixel_values"] for enc, _ in batch]
    labels = torch.stack([lbl for _, lbl in batch])

    tokenizer = clip_processor.tokenizer
    text_inputs = tokenizer.pad(
        {"input_ids": texts, "attention_mask": masks},
        padding=True,
        return_tensors="pt"
    )
    pixel_values = torch.stack(images)
    batch_inputs = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": pixel_values
    }
    return batch_inputs, labels

# ============================================================
# Model
# ============================================================
class CLIPRegressor(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.regressor = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        self.mse = nn.MSELoss()

    def forward(self, inputs, labels=None):
        text_feat = self.clip.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        img_feat = self.clip.get_image_features(pixel_values=inputs["pixel_values"])
        fused = torch.cat([text_feat, img_feat], dim=1)
        preds = self.regressor(fused)
        if labels is not None:
            loss = self.mse(preds, labels)
            return loss, preds
        return preds

# ============================================================
# Main training loop
# ============================================================
def train_clip_model(property_name):
    img_cache_path = os.path.join(DATA_PATH, f"{property_name}_with_images.csv")
    if not os.path.exists(img_cache_path):
        raise FileNotFoundError(
            f"Missing cached dataset: {img_cache_path}\n"
            f"Run: python prepare_clip_data.py first."
        )

    print(f"📂 Using cached dataset: {img_cache_path}")
    df = pd.read_csv(img_cache_path).reset_index(drop=True)

    clip_model = CLIPModel.from_pretrained(CLIP_DIR)

    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
    fold_r2s, fold_rmses = [], []

    for fold, (tr, va) in enumerate(kf.split(df)):
        print(f"\n===== Fold {fold+1}/{KFOLDS} =====")
        scaler = StandardScaler()

        df_train, df_val = df.iloc[tr].copy(), df.iloc[va].copy()
        df_train["value_norm"] = scaler.fit_transform(df_train[["value"]])
        df_val["value_norm"] = scaler.transform(df_val[["value"]])

        train_ds = CLIPDataset(df_train["smiles"].tolist(), df_train["image_path"].tolist(), df_train["value_norm"].tolist())
        val_ds   = CLIPDataset(df_val["smiles"].tolist(),   df_val["image_path"].tolist(),   df_val["value_norm"].tolist())
        tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
        va_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = CLIPRegressor(clip_model).to(DEVICE)

        # freeze everything, then unfreeze last layers (your logic preserved)
        for p in model.clip.parameters(): p.requires_grad = False
        for p in model.clip.text_model.encoder.layers[-1].parameters(): p.requires_grad = True
        for p in model.clip.vision_model.encoder.layers[-1].parameters(): p.requires_grad = True

        opt = AdamW([
            {"params": model.clip.parameters(), "lr": LR_CLIP},
            {"params": model.regressor.parameters(), "lr": LR_HEAD},
        ])

        total_steps = EPOCHS * len(tr_loader)
        sched = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=int(WARMUP_RATIO * total_steps),
            num_training_steps=total_steps
        )

        best_r2, bad_epochs = -np.inf, 0
        last_val_rmse = None

        for ep in range(EPOCHS):
            model.train()
            preds, trues = [], []

            for inputs, labels in tr_loader:
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = labels.to(DEVICE)

                loss, yhat = model(inputs, labels)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                preds.append(yhat.detach().cpu().numpy())
                trues.append(labels.cpu().numpy())

            tr_y = scaler.inverse_transform(np.vstack(trues))
            tr_p = scaler.inverse_transform(np.vstack(preds))
            tr_r2 = r2_score(tr_y, tr_p)
            tr_rmse = np.sqrt(mean_squared_error(tr_y, tr_p))

            # --- Validation ---
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for inputs, labels in va_loader:
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    labels = labels.to(DEVICE)
                    _, yhat = model(inputs, labels)
                    preds.append(yhat.cpu().numpy())
                    trues.append(labels.cpu().numpy())

            y_true = scaler.inverse_transform(np.vstack(trues))
            y_pred = scaler.inverse_transform(np.vstack(preds))
            val_r2 = r2_score(y_true, y_pred)
            val_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            last_val_rmse = val_rmse

            print(f"Ep {ep+1:02d} | Train R²={tr_r2:.3f} | Val R²={val_r2:.3f} | "
                  f"Train RMSE={tr_rmse:.3f} | Val RMSE={val_rmse:.3f}")

            if val_r2 > best_r2:
                best_r2, bad_epochs = val_r2, 0
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    print("⏹️ Early stopping triggered.")
                    break

        fold_r2s.append(best_r2)
        fold_rmses.append(last_val_rmse)

    print(f"\n===== {property_name} Summary =====")
    print(f"R² = {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")
    print(f"RMSE = {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")


if __name__ == "__main__":
    train_clip_model(PROPERTY)
