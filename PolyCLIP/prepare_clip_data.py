# prepare_clip_data.py
# ============================================================
# CLIP Baseline (Step 1): Prepare data + image cache
# - Load property CSV (smiles,value)
# - Canonicalize column names
# - Drop NaNs
# - Generate or reuse cached images (unique smiles->filename)
# - Save cached dataset: {PROPERTY}_with_images.csv
# ============================================================

import os, random, hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw

# ----------------- CONFIG -----------------
DATA_PATH = "./data"          # folder with Nc.csv, Egc.csv, etc.
IMAGE_DIR = "./images"        # image folder (can reuse across properties safely)
PROPERTY = "Nc"               # property file name (eg. Nc.csv)
IMG_MODE = "structure"        # "structure" or "fingerprint" (fingerprint code kept in finetune file)
SEED = 42
random.seed(SEED); np.random.seed(SEED)
# ------------------------------------------


# ============================================================
# Image Generation
# ============================================================
def generate_structure_image(smiles, out_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Draw.MolToFile(mol, out_path, size=(224, 224))
    return out_path

def smiles_to_filename(smiles):
    """Generate a unique, consistent filename for each SMILES string."""
    return hashlib.md5(smiles.encode()).hexdigest() + ".png"

def generate_images_from_smiles(df, smiles_col="smiles", out_dir="./images", mode="structure"):
    os.makedirs(out_dir, exist_ok=True)
    image_paths = []
    print(f"🧪 Generating {mode} images for {len(df)} molecules...")

    for smi in tqdm(df[smiles_col]):
        img_name = smiles_to_filename(smi)
        img_path = os.path.join(out_dir, img_name)

        if not os.path.exists(img_path):
            try:
                if mode == "structure":
                    generate_structure_image(smi, img_path)
                else:
                    # Keeping the option open; fingerprint generator lives in finetune file
                    generate_structure_image(smi, img_path)
            except Exception:
                img_path = None
        image_paths.append(img_path)

    df["image_path"] = image_paths
    df = df.dropna(subset=["image_path"])
    print(f"✅ Generated {len(df)} images and saved in {out_dir}")
    return df.reset_index(drop=True)


# ============================================================
# Main
# ============================================================
def prepare(property_name):
    base_path = os.path.join(DATA_PATH, f"{property_name}.csv")
    img_cache_path = os.path.join(DATA_PATH, f"{property_name}_with_images.csv")

    if os.path.exists(img_cache_path):
        print(f"📂 Already cached: {img_cache_path}")
        return

    print(f"⚙️ Preparing dataset + images for {property_name}")
    df = pd.read_csv(base_path)

    for c in ["smiles", "SMILES", "SMILES descriptor 1"]:
        if c in df.columns: df = df.rename(columns={c: "smiles"})
    for c in ["value", "Value", "logCond60"]:
        if c in df.columns: df = df.rename(columns={c: "value"})

    df = df.dropna(subset=["smiles", "value"]).reset_index(drop=True)
    df = generate_images_from_smiles(df, "smiles", IMAGE_DIR, IMG_MODE)
    df.to_csv(img_cache_path, index=False)

    print(f"✅ Saved cached dataset → {img_cache_path}")


if __name__ == "__main__":
    prepare(PROPERTY)
