from __future__ import annotations

from loguru import logger

from src.data_prep import ResumeDatasetPreparer
from src.semantic_model import SemanticResumeMatcher


def main() -> None:
    logger.info("Step 1: Preparing data from Kaggle resume dataset...")
    ResumeDatasetPreparer().run()

    logger.info("Step 2: Fine-tuning semantic resume matcher...")
    SemanticResumeMatcher().train()

    logger.success("Training complete.")


if __name__ == "__main__":
    main()
