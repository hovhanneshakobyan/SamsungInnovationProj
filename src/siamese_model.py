"""
siamese_model.py
Siamese Transformer Network for semantic resume ↔ job-description matching.
Architecture based on Resume2Vec (Electronics 2025) and sentence-transformers.

Training pipeline:
  1. Load resume–JD pairs from data/processed/pairs.csv
  2. Fine-tune a bi-encoder (shared weights) with cosine-similarity loss
  3. Save checkpoint to models/siamese_checkpoint.pt
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
from loguru import logger

# ── Config ───────────────────────────────────────────────────────────────────
BASE_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"   # lightweight BERT variant
CHECKPOINT   = os.path.join("models", "siamese_checkpoint.pt")
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-5
MARGIN       = 0.5    # for contrastive loss


# ── Mean-pooling helper ───────────────────────────────────────────────────────
def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state          # (B, T, H)
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


# ── Siamese encoder ───────────────────────────────────────────────────────────
class SiameseTransformer(nn.Module):
    """
    Single shared encoder; both resume and JD pass through the same weights.
    Outputs cosine similarity score in [0, 1].
    """
    def __init__(self, model_name: str = BASE_MODEL):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),   # compact embedding space
        )

    def encode(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(output, attention_mask)
        return F.normalize(self.projection(pooled), dim=-1)

    def forward(self, r_ids, r_mask, j_ids, j_mask):
        r_emb = self.encode(r_ids, r_mask)
        j_emb = self.encode(j_ids, j_mask)
        similarity = F.cosine_similarity(r_emb, j_emb)   # (B,)
        return similarity, r_emb, j_emb


# ── Dataset ───────────────────────────────────────────────────────────────────
class ResumePairDataset(Dataset):
    """
    Expects a CSV with columns: resume_text, jd_text, label
    label = 1  (good match) | 0 (mismatch)

    Example rows come from:
      - Kaggle Resume Dataset
      - Kaggle Job Description Dataset
      - Tech Job Posting Dataset (see 3rd_checkpoint.docx)
    """
    def __init__(self, csv_path: str, tokenizer, max_len: int = MAX_LEN):
        self.df        = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.df)

    def _tokenize(self, text: str):
        return self.tokenizer(
            str(text),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        r_enc = self._tokenize(row["resume_text"])
        j_enc = self._tokenize(row["jd_text"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return (
            r_enc["input_ids"].squeeze(0),
            r_enc["attention_mask"].squeeze(0),
            j_enc["input_ids"].squeeze(0),
            j_enc["attention_mask"].squeeze(0),
            label,
        )


# ── Contrastive loss ──────────────────────────────────────────────────────────
class ContrastiveLoss(nn.Module):
    """
    Pulls similar pairs together, pushes dissimilar pairs apart.
    loss = y * (1 - cos)^2  +  (1-y) * max(0, cos - margin)^2
    """
    def __init__(self, margin: float = MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, similarity, labels):
        pos_loss = labels       * (1 - similarity) ** 2
        neg_loss = (1 - labels) * torch.clamp(similarity - self.margin, min=0) ** 2
        return (pos_loss + neg_loss).mean()


# ── Training loop ─────────────────────────────────────────────────────────────
def train(csv_path: str = "data/processed/pairs.csv",
          epochs: int = EPOCHS,
          save_path: str = CHECKPOINT):

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset   = ResumePairDataset(csv_path, tokenizer)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = SiameseTransformer().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for r_ids, r_mask, j_ids, j_mask, labels in tqdm(loader, desc=f"Epoch {epoch}"):
            r_ids, r_mask = r_ids.to(device), r_mask.to(device)
            j_ids, j_mask = j_ids.to(device), j_mask.to(device)
            labels        = labels.to(device)

            optimizer.zero_grad()
            similarity, _, _ = model(r_ids, r_mask, j_ids, j_mask)
            loss = criterion(similarity, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        logger.info(f"Epoch {epoch}/{epochs}  loss={avg:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "base_model": BASE_MODEL}, save_path)
    logger.success(f"Checkpoint saved → {save_path}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def load_model(save_path: str = CHECKPOINT) -> tuple:
    """Load trained model and tokenizer. Returns (model, tokenizer)."""
    ckpt      = torch.load(save_path, map_location="cpu")
    model     = SiameseTransformer(ckpt["base_model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt["base_model"])
    return model, tokenizer


@torch.no_grad()
def get_match_score(resume_text: str, jd_text: str,
                    model=None, tokenizer=None,
                    save_path: str = CHECKPOINT) -> float:
    """
    Returns cosine similarity score in [0, 1].
    Higher = better match.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model(save_path)

    device = next(model.parameters()).device

    def enc(text):
        t = tokenizer(text, max_length=MAX_LEN, padding="max_length",
                      truncation=True, return_tensors="pt")
        return t["input_ids"].to(device), t["attention_mask"].to(device)

    r_ids, r_mask = enc(resume_text)
    j_ids, j_mask = enc(jd_text)
    score, _, _   = model(r_ids, r_mask, j_ids, j_mask)
    return round(float(score.item()), 4)


if __name__ == "__main__":
    # Example: train from scratch (requires data/processed/pairs.csv)
    train()
