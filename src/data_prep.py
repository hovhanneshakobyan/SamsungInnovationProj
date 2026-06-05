from future import annotations

import random
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from src.config import PROCESSED_DIR, RANDOM_SEED, SEMANTIC_PAIRS_CSV
from src.parser import ResumeParser
from src.utils import clean_text_for_matching, set_seed


class ResumeDatasetPreparer:
    def __init__(self):
        set_seed(RANDOM_SEED)
        self.parser = ResumeParser()

    def load_docx_resumes(self, folder: str = "data/raw") -> List[str]:
        folder_path = Path(folder)

        files = list(folder_path.glob("*.docx"))

        if not files:
            raise ValueError("No DOCX files found in data/raw/")

        texts = []

        for file in files:
            try:
                text = self.parser.parse_docx(str(file))
                text = clean_text_for_matching(text)

                if len(text) > 200:
                    texts.append(text)

            except Exception as e:
                logger.warning(f"Failed to parse {file}: {e}")

        logger.info(f"Loaded {len(texts)} resumes")
        return texts

    def split_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []

        chunk_size = 80
        stride = 40

        for i in range(0, len(words), stride):
            chunk = words[i:i + chunk_size]
            if len(chunk) >= 30:
                chunks.append(" ".join(chunk))

        return chunks

    def build_pairs(self, texts: List[str]) -> pd.DataFrame:
        rows = []

        # POSITIVE: same resume chunks
        for text in texts:
            chunks = self.split_into_chunks(text)

            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    if random.random() < 0.3:
                        rows.append({
                            "text_a": chunks[i],
                            "text_b": chunks[j],
                            "label": 1.0,
                        })

        # NEGATIVE: different resumes
        for _ in range(len(rows)):
            t1 = random.choice(texts)
            t2 = random.choice(texts)

            if t1 == t2:
                continue

            c1 = random.choice(self.split_into_chunks(t1))
            c2 = random.choice(self.split_into_chunks(t2))

            rows.append({
                "text_a": c1,
                "text_b": c2,
                "label": 0.0,
            })

        df = pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)
        return df

    def run(self):
        logger.info("Loading DOCX resumes...")
        texts = self.load_docx_resumes()

        logger.info("Building semantic training pairs...")
        pairs = self.build_pairs(texts)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(SEMANTIC_PAIRS_CSV, index=False)

        logger.success(f"Saved {len(pairs)} training pairs")