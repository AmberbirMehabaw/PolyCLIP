# finetune_reghead.py
# ============================================================
# LLaMA Regression Baseline (Step 2): Fine-tune + evaluate
# - Load prepared train/val splits (from prepare_data.py)
# - Tokenize regression-friendly prompt
# - LLaMA-3.1-8B-Instruct (4-bit) + QLoRA + Regression head
# - Trainer + early stopping
# - Evaluate on val (invert Z-score) and save predictions
# ============================================================

import os, math, json, hashlib, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore")

# ===================== USER CONFIG =====================
PROPERTY = "Eea"                 # "Nc", "Eea", "Ei"
OUT_DIR  = f"./llama_reghead_{PROPERTY.lower()}"

MODEL_DIR = "/gpfs/home/ama25d/hf_models/llama3.1-8b-instruct"

SEED = 42
MAX_LEN = 256

# Training
EPOCHS = 30
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
SCHEDULER = "cosine"

# LoRA
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
# =======================================================

# -------------------- Force single GPU (IMPORTANT) ------
# Do this BEFORE importing / initializing CUDA contexts in practice.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -------------------- Repro --------------------
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------- Offline / caches --------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.makedirs(OUT_DIR, exist_ok=True)

def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

# ============================================================
# 0) LOAD PREPARED SPLITS + NORM INFO
# ============================================================
train_df = pd.read_csv(f"{OUT_DIR}/train.csv")
val_df   = pd.read_csv(f"{OUT_DIR}/val.csv")

norm = json.load(open(f"{OUT_DIR}/norm_info.json", "r"))
mu = float(norm["mu"])
sigma = float(norm["sigma"])

print("================================================")
print("✅ Property:", PROPERTY)
print(f"✅ Train samples: {len(train_df)}")
print(f"✅ Val samples:   {len(val_df)}")
print(f"✅ Z-score: mu={mu:.6f}, sigma={sigma:.6f}")
print("================================================")

# ============================================================
# 1) TOKENIZE (regression-friendly)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def make_input(smiles: str):
    return f"SMILES: {smiles}\n{PROPERTY}:"

def encode_batch(batch):
    texts = [make_input(s) for s in batch["smiles"]]
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    enc["labels_reg"] = batch["y"]
    return enc

train_ds = Dataset.from_pandas(train_df[["smiles", "y"]])
val_ds   = Dataset.from_pandas(val_df[["smiles", "y"]])

train_tok = train_ds.map(encode_batch, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(encode_batch, batched=True, remove_columns=val_ds.column_names)

# ============================================================
# 2) MODEL: LLaMA + QLoRA + Regression Head
# ============================================================
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ✅ For 4-bit: do NOT call .to("cuda")
# ✅ Pin whole model to GPU 0
base = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    quantization_config=bnb_cfg,
    device_map={"": 0},   # ✅ everything on cuda:0
)

base.config.use_cache = False
base.config.output_hidden_states = True
base = prepare_model_for_kbit_training(base)

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
base = get_peft_model(base, lora_cfg)

class LlamaRegressor(nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        hidden = llm.config.hidden_size
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, input_ids=None, attention_mask=None, labels_reg=None):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels_reg is not None:
            labels_reg = labels_reg.to(device)

        out = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        h = out.hidden_states[-1]  # [B, T, H]

        idx = attention_mask.sum(dim=1) - 1
        idx = idx.clamp(min=0)
        pooled = h[torch.arange(h.size(0), device=h.device), idx]  # [B, H]

        pred = self.reg_head(pooled).squeeze(-1)  # [B]

        loss = None
        if labels_reg is not None:
            loss = F.mse_loss(pred.float(), labels_reg.float())

        return {"loss": loss, "pred": pred}

model = LlamaRegressor(base)

# ============================================================
# 3) TRAIN (early stopping)
# ============================================================
def collate_fn(features):
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels_reg": torch.tensor([f["labels_reg"] for f in features], dtype=torch.float32),
    }

class RegTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels_reg=inputs["labels_reg"],
        )
        loss = out["loss"]
        return (loss, out) if return_outputs else loss

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=SCHEDULER,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    seed=SEED,
    report_to="none",
)

trainer = RegTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)

trainer.train()

# Save
torch.save(model.state_dict(), f"{OUT_DIR}/reg_head_model.pt")
tokenizer.save_pretrained(OUT_DIR)
print(f"✅ Saved → {OUT_DIR}/reg_head_model.pt")

# ============================================================
# 4) EVALUATION (invert Z-score)
# ============================================================
model.eval()
device = next(model.parameters()).device

preds, trues = [], []

with torch.no_grad():
    for i in tqdm(range(len(val_tok)), desc="Evaluating"):
        ex = val_tok[i]
        inp = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
        att = torch.tensor([ex["attention_mask"]], dtype=torch.long, device=device)

        out = model(input_ids=inp, attention_mask=att, labels_reg=None)
        yhat_norm = float(out["pred"].item())

        preds.append(yhat_norm * sigma + mu)          # invert Z-score
        trues.append(float(val_df.loc[i, "value"]))

preds = np.array(preds, dtype=float)
trues = np.array(trues, dtype=float)

print("\n📊 FINAL RESULTS (REAL)")
print("────────────────────────────────────")
print("✅ Property:", PROPERTY)
print(f"RMSE = {rmse_np(trues, preds):.4f}")
print(f"R²   = {r2_np(trues, preds):.4f}")
print(f"N evaluated = {len(preds)} / {len(val_df)}")
print("────────────────────────────────────")
print(f"true min/max: {trues.min():.4f} {trues.max():.4f}")
print(f"pred min/max: {preds.min():.4f} {preds.max():.4f}")

out_df = pd.DataFrame({
    "smiles": val_df["smiles"].values,
    "true": trues,
    "pred": preds,
})
out_df.to_csv(f"{OUT_DIR}/predictions.csv", index=False)
print(f"✅ Saved predictions → {OUT_DIR}/predictions.csv")
