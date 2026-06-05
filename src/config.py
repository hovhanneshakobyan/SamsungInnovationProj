from __future__ import annotations
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESUME_CSV = RAW_DIR / "resumes.csv"
SEMANTIC_PAIRS_CSV = PROCESSED_DIR / "semantic_pairs.csv"
LABEL_MAP_JSON = PROCESSED_DIR / "label_map.json"
SEMANTIC_MODEL_DIR = MODELS_DIR / "semantic_resume_matcher"
BASE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TRAIN_SAMPLES_PER_CLASS = 120
MAX_PAIRS_PER_CLASS = 400
NEGATIVE_MULTIPLIER = 2
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
RANDOM_SEED = 42