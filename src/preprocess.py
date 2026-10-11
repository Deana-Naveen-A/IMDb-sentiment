"""Data loading and text cleaning for the IMDb sentiment project."""
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"http\S+|www\.\S+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation. Negations are KEPT on purpose."""
    text = str(text).lower()
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    # Turn contractions into explicit negations so "don't" -> "do not"
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = text.replace("n't", " not")
    text = NON_ALNUM_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def load_imdb(data_dir: str = "data"):
    """Return (train_df, test_df) with columns: text, label (1 = positive).

    Option A: if data/'IMDB Dataset.csv' (Kaggle, 50k reviews) exists, use it
              with a stratified 50/50 split.
    Option B: otherwise download the standard 25k/25k split from Hugging Face.
    """
    csv_path = os.path.join(data_dir, "IMDB Dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).rename(columns={"review": "text", "sentiment": "label"})
        df["label"] = (df["label"] == "positive").astype(int)
        train_df, test_df = train_test_split(
            df, test_size=0.5, stratify=df["label"], random_state=42
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    return ds["train"].to_pandas(), ds["test"].to_pandas()
