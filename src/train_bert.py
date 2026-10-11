"""Fine-tune DistilBERT on IMDb.

Run (GPU strongly recommended, e.g. Google Colab / Kaggle):
    python -m src.train_bert
Low on time or only have a CPU? Subsample:
    python -m src.train_bert --train_size 4000 --test_size 2000 --epochs 1
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.preprocess import load_imdb

MODEL_NAME = "distilbert-base-uncased"
HTML_RE = re.compile(r"<[^>]+>")


def light_clean(t):
    # Transformers need nearly raw text; only strip HTML tags
    return HTML_RE.sub(" ", str(t)).strip()


def subsample(df, n, seed=42):
    if n >= len(df):
        return df
    return train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_size", type=int, default=25000)
    ap.add_argument("--test_size", type=int, default=25000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("GPU available:", torch.cuda.is_available())

    train_df, test_df = load_imdb()
    train_df = subsample(train_df, args.train_size)
    test_df = subsample(test_df, args.test_size)
    for d in (train_df, test_df):
        d["text"] = d["text"].map(light_clean)
    print(f"Training on {len(train_df)} reviews, testing on {len(test_df)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    cols = ["text", "label"]
    train_ds = Dataset.from_pandas(train_df[cols].reset_index(drop=True)).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[cols].reset_index(drop=True)).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    targs = TrainingArguments(
        output_dir="distilbert_out",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()

    # ---- Evaluate on test set ----
    out = trainer.predict(test_ds)
    y_pred = np.argmax(out.predictions, axis=-1)
    y_true = test_df["label"].to_numpy()
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    row = {"model": "DistilBERT", "best_param": f"lr={args.lr}, epochs={args.epochs}",
           "val_acc": np.nan, "accuracy": accuracy_score(y_true, y_pred),
           "precision": p, "recall": r, "f1": f, "train_time_s": np.nan}
    print("\nDistilBERT test results:", {k: row[k] for k in ("accuracy", "precision", "recall", "f1")})

    # Add/replace the DistilBERT row in the shared results table
    path = "results/results.csv"
    res = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if "model" in res.columns:
        res = res[res["model"] != "DistilBERT"]
    pd.concat([res, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)

    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                           display_labels=["Negative", "Positive"]).plot(cmap="Blues")
    import matplotlib.pyplot as plt
    plt.title("DistilBERT (test)")
    plt.tight_layout()
    plt.savefig("results/confusion_distilbert.png", dpi=150)

    model.save_pretrained("models/distilbert")
    tok.save_pretrained("models/distilbert")
    print("Saved model to models/distilbert")


if __name__ == "__main__":
    main()
