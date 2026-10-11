# IMDb Review Sentiment Classification

UE24CS352A - Machine Learning mini-project (Problem #10)
Team: `<Name 1 (SRN)>`, `<Name 2 (SRN)>`

Binary sentiment classification (positive / negative) of IMDb movie reviews
using classical ML (TF-IDF + Logistic Regression / Naive Bayes / Linear SVM)
and a fine-tuned transformer (DistilBERT). Reference: Wu & Shin,
*Machine Learning based classification for sentimental analysis of IMDb reviews*,
Stanford CS229 project report (2020).

## Dataset
Large Movie Review Dataset (Maas et al., 2011): 50,000 labelled reviews,
25,000 train / 25,000 test, perfectly balanced.
Loaded automatically from Hugging Face (`imdb`). Alternatively place the Kaggle
file `IMDB Dataset.csv` in `data/` (it is then split 50/50, stratified).

## Project structure
```
app.py                  Streamlit demo
src/preprocess.py       text cleaning + data loading
src/eda.py              exploratory plots/statistics
src/train_classical.py  TF-IDF + LR / NB / SVM, tuning, ablation
src/train_bert.py       DistilBERT fine-tuning
src/evaluate.py         error analysis (misclassified reviews, top words)
src/predict.py          prediction helpers used by the app
results/                metrics tables and figures (generated)
models/                 saved models (generated, not committed)
docs/                   write-up template and slide outline
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run (from the repo root, in this order)
```bash
python -m src.eda               # 1. data exploration, plots -> results/
python -m src.train_classical   # 2. train LR/NB/SVM, results.csv, ablation.csv
python -m src.evaluate          # 3. error analysis
python -m src.train_bert        # 4. (optional) DistilBERT, use a GPU (Colab/Kaggle)
streamlit run app.py            # 5. live demo at http://localhost:8501
```
CPU only? `python -m src.train_bert --train_size 4000 --test_size 2000 --epochs 1`

## Results
Paste your final table from `results/results.csv` here.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Naive Bayes | | | | |
| Linear SVM | | | | |
| DistilBERT | | | | |

## Contributions
- `<Name 1>`: ...
- `<Name 2>`: ...
