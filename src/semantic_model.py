from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import torch
from loguru import logger
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.config import (
    BASE_EMBED_MODEL,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEMANTIC_MODEL_DIR,
    SEMANTIC_PAIRS_CSV,
)


class SemanticResumeMatcher:
    def __init__(self, model_dir: str | None = None):
        self.model_dir = str(model_dir or SEMANTIC_MODEL_DIR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Semantic matcher device: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        if Path(self.model_dir).exists():
            self.model = SentenceTransformer(self.model_dir, device=self.device)
        else:
            self.model = SentenceTransformer(BASE_EMBED_MODEL, device=self.device)

    def train(self, csv_path: str | None = None) -> None:
        csv_path = csv_path or str(SEMANTIC_PAIRS_CSV)
        df = pd.read_csv(csv_path)

        examples: List[InputExample] = []
        for _, row in df.iterrows():
            examples.append(
                InputExample(
                    texts=[str(row["text_a"]), str(row["text_b"])],
                    label=float(row["label"]),
                )
            )

        train_loader = DataLoader(
            examples,
            shuffle=True,
            batch_size=BATCH_SIZE,
            pin_memory=(self.device == "cuda"),
        )

        train_loss = losses.CosineSimilarityLoss(self.model)

        logger.info(f"Training semantic matcher on {len(examples)} pairs...")
        logger.info(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")

        self.model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=EPOCHS,
            warmup_steps=max(10, len(train_loader) // 10),
            optimizer_params={"lr": LEARNING_RATE},
            output_path=self.model_dir,
            show_progress_bar=True,
        )

        logger.success(f"Saved fine-tuned semantic model to {self.model_dir}")

    def encode(self, texts: List[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


if __name__ == "__main__":
    SemanticResumeMatcher().train()