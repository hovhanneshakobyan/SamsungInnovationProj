"""
resume_optimizer.py
Generative AI module:
  - train_t5()           → fine-tune T5 for resume OPTIMIZATION  (resume+jd → improved resume)
  - train_t5_generator() → fine-tune T5 for resume GENERATION    (prompt    → full resume)
"""

import os
import re
import numpy as np
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset
import pandas as pd
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
T5_BASE       = "t5-base"
T5_CKPT       = os.path.join("models", "optimizer_t5")
T5_GEN_CKPT   = os.path.join("models", "generator_t5")
MAX_INPUT     = 512
MAX_TARGET    = 512
BATCH_SIZE    = 4
EPOCHS        = 3
LR            = 3e-4

_STOPWORDS = {
    "a","an","the","and","or","in","on","at","to","for","of","with","is","are",
    "was","were","be","been","have","has","had","do","does","did","will","would",
    "can","could","should","may","might","we","you","they","he","she","it","this",
    "that","which","who","from","by","as","our","your","their","all","also","not",
    "but","if","then","so","its","into","over","after","under","while","about",
    "through","during","before","between","each","both","few","more","most",
    "other","some","such","no","only","same","than","too","very","just","now",
    "looking","hiring","seeking","required","must","please","able","work","team",
    "strong","good","excellent","responsibilities","requirements","candidate",
}


def _t5_is_ready(path: str = T5_CKPT) -> bool:
    if not os.path.isdir(path):
        return False
    has_model  = any(f in os.listdir(path)
                     for f in ["model.safetensors", "pytorch_model.bin", "tf_model.h5"])
    has_config = "config.json" in os.listdir(path)
    return has_model and has_config

def _t5_gen_is_ready() -> bool:
    return _t5_is_ready(T5_GEN_CKPT)


# ── Generic seq2seq dataset ───────────────────────────────────────────────────
class Seq2SeqDataset(Dataset):
    """
    Generic dataset for any T5 seq2seq task.

    FIX: __getitem__ returns plain Python lists (no padding, no return_tensors).
    DataCollatorForSeq2Seq handles dynamic padding and tensor conversion itself,
    which avoids the "Creating a tensor from a list of numpy.ndarrays is slow" warning.
    """
    def __init__(self, csv_path, tokenizer,
                 input_col="input", target_col="target",
                 max_in=MAX_INPUT, max_tgt=MAX_TARGET):
        self.df         = pd.read_csv(csv_path)
        self.tokenizer  = tokenizer
        self.input_col  = input_col
        self.target_col = target_col
        self.max_in     = max_in
        self.max_tgt    = max_tgt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        source = str(row[self.input_col])
        target = str(row[self.target_col])

        # ── Tokenize WITHOUT padding and WITHOUT return_tensors ───────────────
        # DataCollatorForSeq2Seq will pad and convert to tensors in the collate step.
        # This avoids: "Creating a tensor from a list of numpy.ndarrays is slow"
        enc = self.tokenizer(
            source,
            max_length=self.max_in,
            truncation=True,
            padding=False,          # no padding here
        )
        tgt = self.tokenizer(
            target,
            max_length=self.max_tgt,
            truncation=True,
            padding=False,          # no padding here
        )

        # Convert to numpy arrays first so torch.tensor() gets a single ndarray
        input_ids      = np.array(enc["input_ids"],      dtype=np.int64)
        attention_mask = np.array(enc["attention_mask"], dtype=np.int64)
        labels         = np.array(tgt["input_ids"],      dtype=np.int64)

        # Mask padding tokens in labels so they are ignored in loss
        # (DataCollatorForSeq2Seq will pad labels with -100 automatically
        #  when label_pad_token_id=-100 is set — which is the default)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Shared T5 trainer ─────────────────────────────────────────────────────────
def _train_t5(csv_path, output_dir, input_col, target_col, epochs=EPOCHS):
    tokenizer = T5Tokenizer.from_pretrained(T5_BASE)
    model     = T5ForConditionalGeneration.from_pretrained(T5_BASE)
    dataset   = Seq2SeqDataset(csv_path, tokenizer, input_col, target_col)

    # label_pad_token_id=-100 ensures padding in labels is ignored in loss
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,     # small speedup on modern hardware
    )

    args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = BATCH_SIZE,
        learning_rate               = LR,
        save_strategy               = "epoch",
        logging_steps               = 50,
        fp16                        = torch.cuda.is_available(),
        report_to                   = "none",
        dataloader_pin_memory       = False,   # suppress pin_memory warning on CPU
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=dataset, data_collator=collator)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ── Public training functions ─────────────────────────────────────────────────
def train_t5(csv_path: str = "data/processed/optimized_pairs.csv",
             output_dir: str = T5_CKPT):
    """Fine-tune T5 for resume OPTIMIZATION (resume+jd → optimized resume)."""
    logger.info("Building optimizer CSV with input/target columns …")
    df = pd.read_csv(csv_path)
    df["input"]  = ("optimize resume: " + df["resume_text"].astype(str)
                    + " </s> job: "     + df["jd_text"].astype(str))
    df["target"] = df["optimized_resume"].astype(str)
    tmp_path = csv_path.replace(".csv", "_t5.csv")
    df[["input", "target"]].to_csv(tmp_path, index=False)

    _train_t5(tmp_path, output_dir, "input", "target")
    logger.success(f"T5 optimizer saved → {output_dir}")


def train_t5_generator(
        csv_path:   str = "data/processed/generation_pairs.csv",
        output_dir: str = T5_GEN_CKPT):
    """Fine-tune T5 for resume GENERATION (structured prompt → full resume)."""
    _train_t5(csv_path, output_dir, "prompt", "target")
    logger.success(f"T5 generator saved → {output_dir}")


# ── Rule-based fallback helpers ───────────────────────────────────────────────
def _extract_jd_keywords(jd_text: str, top_n: int = 20) -> list:
    tokens = re.findall(r'\b[A-Za-z][A-Za-z0-9+#.\-]{1,}\b', jd_text)
    seen, result = set(), []
    for t in tokens:
        tl = t.lower()
        if tl not in _STOPWORDS and len(tl) > 2 and tl not in seen:
            seen.add(tl); result.append(t)
        if len(result) >= top_n:
            break
    return result


def _inject_keywords(resume_text: str, keywords: list) -> str:
    resume_lower = resume_text.lower()
    missing      = [kw for kw in keywords if kw.lower() not in resume_lower]
    if not missing:
        return resume_text
    missing_str  = ", ".join(missing)

    skills_pat = re.compile(r'(SKILLS?[:\s]*\n?)(.*?)(\n\n|\Z)',
                             re.IGNORECASE | re.DOTALL)
    m = skills_pat.search(resume_text)
    if m:
        old = m.group(2).strip()
        new = old + (", " if old else "") + missing_str
        resume_text = resume_text[:m.start(2)] + new + resume_text[m.end(2):]
    else:
        resume_text = resume_text.rstrip() + f"\n\nSKILLS\n{missing_str}\n"

    top3 = missing[:3]
    sum_pat = re.compile(r'(SUMMARY\s*\n?)(.*?)(\n\n|\Z)',
                          re.IGNORECASE | re.DOTALL)
    sm = sum_pat.search(resume_text)
    if sm and top3:
        old = sm.group(2).strip()
        add = f" Proficient in {', '.join(top3)}."
        if add.lower() not in old.lower():
            resume_text = (resume_text[:sm.start(2)]
                           + old.rstrip('.') + add
                           + resume_text[sm.end(2):])
    return resume_text


# ── ResumeOptimizer ───────────────────────────────────────────────────────────
class ResumeOptimizer:
    """Optimizes an existing resume to match a job description."""

    def __init__(self, use_t5: bool = None):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_t5 = (_t5_is_ready() if use_t5 is None else
                        (use_t5 and _t5_is_ready()))
        if self._use_t5:
            logger.info("Loading fine-tuned T5 optimizer …")
            self.tokenizer = T5Tokenizer.from_pretrained(T5_CKPT)
            self.model     = T5ForConditionalGeneration.from_pretrained(T5_CKPT).to(self.device)
        else:
            logger.info("Using rule-based keyword optimizer (T5 not ready yet).")
            self.tokenizer = None
            self.model     = None

    @torch.no_grad()
    def optimize(self, resume_text: str, jd_text: str,
                 num_beams: int = 4, max_new_tokens: int = 400) -> str:
        if self._use_t5:
            prompt = f"optimize resume: {resume_text[:400]} </s> job: {jd_text[:400]}"
            ids    = self.tokenizer(prompt, return_tensors="pt",
                                    max_length=MAX_INPUT, truncation=True
                                    ).input_ids.to(self.device)
            out = self.model.generate(ids, num_beams=num_beams,
                                       max_new_tokens=max_new_tokens, early_stopping=True)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            return _inject_keywords(resume_text,
                                    _extract_jd_keywords(jd_text, top_n=20))
