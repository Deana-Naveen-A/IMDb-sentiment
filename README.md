# IMDb Review Sentiment Classification

UE24CS352A - Machine Learning mini-project (Problem #10)
Team: `<Deana Naveen A -PES1UG24AM077>`, `<Gonu Guntla Hema Kowshik -PES1UG24AM106>`

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

<img width="977" height="122" alt="Screenshot 2026-10-11 175102" src="https://github.com/user-attachments/assets/1fe49333-750e-474f-81bf-470d663c2479" />


## Contributions
**Deana Naveen (PES1UG24AM077)**
- Data loading and text preprocessing (`src/preprocess.py`) and exploratory analysis (`src/eda.py`)
- TF-IDF feature extraction and the classical models: Logistic Regression, Naive Bayes and Linear SVM, with hyperparameter tuning and the feature ablation (`src/train_classical.py`)
- Error analysis of misclassified reviews and top features (`src/evaluate.py`)
- Dataset, preprocessing and classical-model sections of the report and slides

**Gonu Guntla Hema Kowshik (PES1UG24AM106)**
- DistilBERT fine-tuning and evaluation on Google Colab (`src/train_bert.py`)
- Prediction helpers and the Streamlit demo app (`src/predict.py`, `app.py`)
- DistilBERT, implementation and demo sections of the report and slides
