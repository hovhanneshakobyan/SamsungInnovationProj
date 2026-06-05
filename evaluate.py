"""
evaluate.py
Evaluation script — measures model performance after training.

Metrics
-------
Siamese model  : Accuracy, AUC-ROC, F1 on held-out pairs
ATS Checker    : Mean ATS score improvement before vs after optimization
T5 Optimizer   : ROUGE-L score on optimized resume vs gold label
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from loguru import logger

try:
    from rouge_score import rouge_scorer
    _ROUGE = True
except ImportError:
    _ROUGE = False
    logger.warning("rouge-score not installed — skipping ROUGE evaluation")

from src.siamese_model   import load_model, get_match_score
from src.ats_checker     import ats_score
from src.resume_optimizer import ResumeOptimizer


# ── Siamese evaluation ─────────────────────────────────────────────────────────
def evaluate_siamese(test_csv: str = "data/processed/pairs.csv",
                     ckpt: str     = "models/siamese_checkpoint.pt",
                     threshold: float = 0.5,
                     n_samples: int   = 400) -> dict:
    """Evaluate similarity model on held-out portion of pairs.csv."""

    if not os.path.exists(ckpt):
        logger.error(f"Checkpoint not found: {ckpt}  — train first with train_all.py")
        return {}

    logger.info("Loading Siamese model …")
    model, tokenizer = load_model(ckpt)

    df = pd.read_csv(test_csv).tail(n_samples)   # use last N rows as test split
    logger.info(f"Evaluating on {len(df)} pairs …")

    scores = []
    labels = []
    for _, row in df.iterrows():
        score = get_match_score(str(row["resume_text"]), str(row["jd_text"]),
                                model=model, tokenizer=tokenizer)
        scores.append(score)
        labels.append(int(row["label"]))

    scores_np = np.array(scores)
    labels_np = np.array(labels)
    preds     = (scores_np >= threshold).astype(int)

    results = {
        "accuracy": round(accuracy_score(labels_np, preds), 4),
        "auc_roc":  round(roc_auc_score(labels_np, scores_np), 4),
        "f1":       round(f1_score(labels_np, preds), 4),
        "n_samples": len(df),
        "threshold": threshold,
    }

    logger.success("─── Siamese Model Results ───────────────────────")
    for k, v in results.items():
        logger.success(f"  {k:<15} {v}")
    print("\nClassification Report:")
    print(classification_report(labels_np, preds, target_names=["Mismatch", "Match"]))

    return results


# ── ATS improvement evaluation ─────────────────────────────────────────────────
def evaluate_ats_improvement(test_csv: str = "data/processed/optimized_pairs.csv",
                              n_samples: int = 100) -> dict:
    """
    Measures mean ATS score before and after optimization.
    Reports average improvement.
    """
    if not os.path.exists(test_csv):
        logger.error(f"File not found: {test_csv}")
        return {}

    logger.info("Evaluating ATS improvement …")
    optimizer = ResumeOptimizer()
    df        = pd.read_csv(test_csv).head(n_samples)

    before_scores = []
    after_scores  = []

    for i, row in df.iterrows():
        resume    = str(row["resume_text"])
        jd        = str(row["jd_text"])

        before    = ats_score(resume, jd)["overall_score"]

        optimized = optimizer.optimize(resume, jd)
        after     = ats_score(optimized, jd)["overall_score"]

        before_scores.append(before)
        after_scores.append(after)

        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i+1}/{n_samples}")

    results = {
        "mean_before":     round(float(np.mean(before_scores)), 2),
        "mean_after":      round(float(np.mean(after_scores)),  2),
        "mean_improvement": round(float(np.mean(after_scores)) - float(np.mean(before_scores)), 2),
        "n_samples":       n_samples,
    }

    logger.success("─── ATS Improvement Results ─────────────────────")
    logger.success(f"  Mean ATS before : {results['mean_before']}")
    logger.success(f"  Mean ATS after  : {results['mean_after']}")
    logger.success(f"  Improvement     : +{results['mean_improvement']}")

    return results


# ── ROUGE evaluation (T5 quality) ──────────────────────────────────────────────
def evaluate_rouge(test_csv: str = "data/processed/optimized_pairs.csv",
                   n_samples: int = 100) -> dict:
    """ROUGE-L score: optimized output vs gold label."""
    if not _ROUGE:
        return {"error": "rouge-score not installed"}
    if not os.path.exists(test_csv):
        return {"error": f"File not found: {test_csv}"}

    logger.info("Evaluating ROUGE-L …")
    optimizer = ResumeOptimizer()
    scorer    = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    df        = pd.read_csv(test_csv).head(n_samples)

    rouge_scores = []
    for _, row in df.iterrows():
        pred  = optimizer.optimize(str(row["resume_text"]), str(row["jd_text"]))
        ref   = str(row["optimized_resume"])
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        rouge_scores.append(score)

    result = {"rougeL_mean": round(float(np.mean(rouge_scores)), 4)}
    logger.success(f"─── ROUGE-L : {result['rougeL_mean']} ───────────────────────")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RESUME2VEC — MODEL EVALUATION")
    print("="*60 + "\n")

    siamese_results = evaluate_siamese()
    ats_results     = evaluate_ats_improvement(n_samples=50)
    rouge_results   = evaluate_rouge(n_samples=50)

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    if siamese_results:
        print(f"  Siamese Accuracy : {siamese_results.get('accuracy')}")
        print(f"  Siamese AUC-ROC  : {siamese_results.get('auc_roc')}")
        print(f"  Siamese F1       : {siamese_results.get('f1')}")
    if ats_results:
        print(f"  ATS Improvement  : +{ats_results.get('mean_improvement')} pts")
    if rouge_results and "rougeL_mean" in rouge_results:
        print(f"  ROUGE-L          : {rouge_results.get('rougeL_mean')}")
    print("="*60)
