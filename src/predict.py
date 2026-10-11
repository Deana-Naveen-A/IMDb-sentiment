"""Prediction helpers shared by the demo app and evaluation scripts."""
import os

import joblib
import numpy as np

from src.preprocess import clean_text

MODEL_DIR = "models"
BERT_DIR = os.path.join(MODEL_DIR, "distilbert")


def load_classical(name: str = "best_classical"):
    return joblib.load(os.path.join(MODEL_DIR, f"{name}.joblib"))


def predict_classical(model, text: str):
    """Return (label, confidence in [0.5, 1])."""
    cleaned = clean_text(text)
    if hasattr(model, "predict_proba"):
        p_pos = float(model.predict_proba([cleaned])[0][1])
    else:  # LinearSVC: squash the margin through a sigmoid (not calibrated)
        p_pos = float(1 / (1 + np.exp(-model.decision_function([cleaned])[0])))
    label = "Positive" if p_pos >= 0.5 else "Negative"
    return label, max(p_pos, 1 - p_pos)


def bert_available() -> bool:
    return os.path.isdir(BERT_DIR)


def load_bert():
    from transformers import pipeline

    return pipeline("text-classification", model=BERT_DIR, tokenizer=BERT_DIR)


def predict_bert(pipe, text: str):
    out = pipe(text, truncation=True, max_length=256)[0]
    label = "Positive" if out["label"].endswith("1") else "Negative"
    return label, float(out["score"])
