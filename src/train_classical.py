"""Train + evaluate classical TF-IDF models. Run: python -m src.train_classical"""
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import clean_text, load_imdb

SEED = 42
RESULTS_DIR = "results"
MODEL_DIR = "models"


def compute_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f}


def make_vectorizer(ngram_range=(1, 2), stop_words=None):
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
        stop_words=stop_words,
    )


def slug(name):
    return name.lower().replace(" ", "_")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---------- Data ----------
    train_df, test_df = load_imdb()
    train_df["clean"] = train_df["text"].map(clean_text)
    test_df["clean"] = test_df["text"].map(clean_text)
    tr, val = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label"], random_state=SEED
    )
    print(f"train={len(tr)} val={len(val)} test={len(test_df)}")

    # ---------- Features (fit on TRAIN only: no leakage) ----------
    vec = make_vectorizer().fit(tr["clean"])
    X_tr, X_val, X_te = (vec.transform(d["clean"]) for d in (tr, val, test_df))
    y_tr, y_val, y_te = tr["label"], val["label"], test_df["label"]
    print(f"TF-IDF features: {X_tr.shape[1]}")

    # ---------- Models + small hyperparameter search on validation set ----------
    grids = {
        "Logistic Regression": (lambda v: LogisticRegression(C=v, max_iter=1000), [0.5, 1, 5, 10, 20]),
        "Naive Bayes": (lambda v: MultinomialNB(alpha=v), [0.1, 0.3, 0.5, 1.0]),
        "Linear SVM": (lambda v: LinearSVC(C=v, max_iter=5000), [0.05, 0.1, 0.5, 1.0]),
    }

    rows = []
    for name, (builder, values) in grids.items():
        best_val, best_param = -1, None
        for v in values:
            acc = accuracy_score(y_val, builder(v).fit(X_tr, y_tr).predict(X_val))
            print(f"  {name} param={v}: val acc={acc:.4f}")
            if acc > best_val:
                best_val, best_param = acc, v

        start = time.time()
        clf = builder(best_param).fit(X_tr, y_tr)
        train_time = time.time() - start
        y_pred = clf.predict(X_te)
        m = compute_metrics(y_te, y_pred)
        rows.append({"model": name, "best_param": best_param, "val_acc": best_val, **m,
                     "train_time_s": round(train_time, 2)})

        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{slug(name)}.joblib"))

        cm = confusion_matrix(y_te, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(cmap="Blues")
        plt.title(f"{name} (test)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_{slug(name)}.png"), dpi=150)
        plt.close()

    results = pd.DataFrame(rows)
    # Best model chosen by VALIDATION accuracy (not test) to stay honest
    best_name = results.sort_values("val_acc", ascending=False).iloc[0]["model"]
    joblib.dump(joblib.load(os.path.join(MODEL_DIR, f"{slug(best_name)}.joblib")),
                os.path.join(MODEL_DIR, "best_classical.joblib"))
    results.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print("\n=== Test results ===")
    print(results.round(4).to_string(index=False))
    print(f"\nBest classical model (by val acc): {best_name}")

    # ---------- Ablation (Logistic Regression, C=5) ----------
    configs = {
        "Unigrams": dict(ngram_range=(1, 1), stop_words=None),
        "Uni+Bigrams": dict(ngram_range=(1, 2), stop_words=None),
        "Uni+Bigrams + stopword removal": dict(ngram_range=(1, 2), stop_words="english"),
    }
    abl = []
    for label, cfg in configs.items():
        v = make_vectorizer(**cfg).fit(tr["clean"])
        clf = LogisticRegression(C=5, max_iter=1000).fit(v.transform(tr["clean"]), y_tr)
        acc = accuracy_score(y_te, clf.predict(v.transform(test_df["clean"])))
        abl.append({"features": label, "n_features": len(v.vocabulary_), "test_accuracy": acc})
    abl = pd.DataFrame(abl)
    abl.to_csv(os.path.join(RESULTS_DIR, "ablation.csv"), index=False)
    print("\n=== Ablation (Logistic Regression) ===")
    print(abl.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
