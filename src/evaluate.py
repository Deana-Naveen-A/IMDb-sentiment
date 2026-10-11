"""Error analysis for the best classical model. Run: python -m src.evaluate"""
import os

import joblib
import numpy as np
import pandas as pd

from src.preprocess import clean_text, load_imdb

RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _, test_df = load_imdb()
    test_df["clean"] = test_df["text"].map(clean_text)
    model = joblib.load("models/best_classical.joblib")
    test_df["pred"] = model.predict(test_df["clean"])
    test_df["words"] = test_df["text"].str.split().str.len()

    wrong = test_df[test_df["pred"] != test_df["label"]]
    fp = ((test_df["pred"] == 1) & (test_df["label"] == 0)).sum()
    fn = ((test_df["pred"] == 0) & (test_df["label"] == 1)).sum()
    print(f"Misclassified: {len(wrong)} / {len(test_df)}  (false positives={fp}, false negatives={fn})")
    wrong[["text", "label", "pred", "words"]].to_csv(
        os.path.join(RESULTS_DIR, "misclassified.csv"), index=False
    )

    print("\n--- 15 random misclassified reviews (read these for your write-up!) ---")
    for _, r in wrong.sample(min(15, len(wrong)), random_state=7).iterrows():
        true, pred = ("pos", "neg") if r["label"] == 1 else ("neg", "pos")
        print(f"[true={true} pred={pred}, {r['words']} words] {r['text'][:300]}...\n")

    # Accuracy by review length
    bins = [0, 100, 200, 300, 500, 100000]
    labels = ["<=100", "101-200", "201-300", "301-500", ">500"]
    test_df["length_bucket"] = pd.cut(test_df["words"], bins=bins, labels=labels)
    by_len = (
        test_df.assign(correct=test_df["pred"] == test_df["label"])
        .groupby("length_bucket", observed=True)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy"})
    )
    print("--- Accuracy by review length (words) ---")
    print(by_len.round(4))
    by_len.to_csv(os.path.join(RESULTS_DIR, "accuracy_by_length.csv"))

    # Most informative words (Logistic Regression coefficients)
    lr = joblib.load("models/logistic_regression.joblib")
    names = np.array(lr.named_steps["tfidf"].get_feature_names_out())
    coefs = lr.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    top_neg, top_pos = names[order[:20]], names[order[-20:][::-1]]
    print("\n--- Top 20 NEGATIVE features ---\n", ", ".join(top_neg))
    print("\n--- Top 20 POSITIVE features ---\n", ", ".join(top_pos))
    pd.DataFrame({"negative": top_neg, "positive": top_pos}).to_csv(
        os.path.join(RESULTS_DIR, "top_features.csv"), index=False
    )


if __name__ == "__main__":
    main()
