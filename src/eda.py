"""Exploratory data analysis. Run: python -m src.eda"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocess import load_imdb

RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_df, test_df = load_imdb()
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    print("Train label counts:\n", train_df["label"].value_counts().rename({0: "negative", 1: "positive"}))
    print("Test label counts:\n", test_df["label"].value_counts().rename({0: "negative", 1: "positive"}))

    train_df["words"] = train_df["text"].str.split().str.len()
    print("\nReview length (words):")
    print(train_df["words"].describe().round(1))

    # Plot 1: class balance
    counts = train_df["label"].value_counts().sort_index()
    plt.figure(figsize=(4, 3))
    plt.bar(["Negative", "Positive"], counts.values, color=["#d9534f", "#5cb85c"])
    plt.title("Training set class balance")
    plt.ylabel("Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "eda_class_balance.png"), dpi=150)
    plt.close()

    # Plot 2: review length distribution
    plt.figure(figsize=(5, 3))
    plt.hist(train_df["words"].clip(upper=1000), bins=50, color="#337ab7")
    plt.title("Review length (words, clipped at 1000)")
    plt.xlabel("Words")
    plt.ylabel("Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "eda_length_hist.png"), dpi=150)
    plt.close()

    print("\nSample reviews:")
    for _, row in train_df.sample(3, random_state=1).iterrows():
        print(f"[{'pos' if row['label'] else 'neg'}] {row['text'][:250]}...\n")
    print(f"Plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
