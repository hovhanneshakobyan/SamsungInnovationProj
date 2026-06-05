"""
train_all.py
One-command training — runs all three models sequentially.

  Step 0: Generate synthetic data (if not present)
  Step 1: Siamese Transformer  (semantic matching)
  Step 2: T5 Optimizer         (resume improvement: resume+jd → optimized)
  Step 3: T5 Generator         (resume generation:  prompt    → full resume)

Usage:
    python train_all.py                          # train everything
    python train_all.py --synthetic              # force synthetic data
    python train_all.py --skip-siamese           # skip siamese
    python train_all.py --skip-optimizer         # skip T5 optimizer
    python train_all.py --skip-generator         # skip T5 generator
    python train_all.py --only-generator         # train generator only
"""

import argparse
import os
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Train Resume2Vec models")
    parser.add_argument("--skip-siamese",    action="store_true")
    parser.add_argument("--skip-optimizer",  action="store_true")
    parser.add_argument("--skip-generator",  action="store_true")
    parser.add_argument("--only-generator",  action="store_true",
                        help="Train T5 generator only (fastest option after Siamese is done)")
    parser.add_argument("--synthetic",       action="store_true")
    parser.add_argument("--siamese-data",    default="data/processed/pairs.csv")
    parser.add_argument("--optimizer-data",  default="data/processed/optimized_pairs.csv")
    parser.add_argument("--generator-data",  default="data/processed/generation_pairs.csv")
    parser.add_argument("--epochs-siamese",  type=int, default=5)
    parser.add_argument("--epochs-t5",       type=int, default=3)
    args = parser.parse_args()

    # --only-generator shortcut
    if args.only_generator:
        args.skip_siamese  = True
        args.skip_optimizer = True

    logger.info("=" * 60)
    logger.info("  RESUME2VEC — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Step 0: Data ─────────────────────────────────────────────────────────
    needs_data = (
        args.synthetic
        or not os.path.exists(args.siamese_data)
        or not os.path.exists(args.generator_data)
    )
    if needs_data:
        logger.info("Generating synthetic training data …")
        from src.generate_synthetic_data import generate
        generate(n_per_domain=500)
    else:
        logger.info("Using existing processed data.")

    # ── Step 1: Siamese ───────────────────────────────────────────────────────
    if not args.skip_siamese:
        logger.info("=" * 60)
        logger.info("STEP 1 — Siamese Transformer Network")
        logger.info("=" * 60)
        from src.siamese_model import train as train_siamese
        train_siamese(csv_path=args.siamese_data, epochs=args.epochs_siamese)
    else:
        logger.info("Skipping Siamese training.")

    # ── Step 2: T5 Optimizer ──────────────────────────────────────────────────
    if not args.skip_optimizer:
        logger.info("=" * 60)
        logger.info("STEP 2 — T5 Resume Optimizer  (resume+jd → optimized)")
        logger.info("=" * 60)
        from src.resume_optimizer import train_t5
        train_t5(csv_path=args.optimizer_data)
    else:
        logger.info("Skipping T5 optimizer training.")

    # ── Step 3: T5 Generator ──────────────────────────────────────────────────
    if not args.skip_generator:
        logger.info("=" * 60)
        logger.info("STEP 3 — T5 Resume Generator  (prompt → full resume)")
        logger.info("=" * 60)
        from src.resume_optimizer import train_t5_generator
        train_t5_generator(csv_path=args.generator_data)
    else:
        logger.info("Skipping T5 generator training.")

    logger.success("=" * 60)
    logger.success("  ALL TRAINING COMPLETE!")
    logger.success("  Evaluate : python evaluate.py")
    logger.success("  Launch   : streamlit run app.py")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
