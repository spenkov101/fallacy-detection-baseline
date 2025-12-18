# Fallacy Detection Baseline

A **public-safe research baseline** for fallacy detection using classical NLP
and lightweight semantic retrieval.

The goal of this repository is to provide a **minimal, reproducible starting point**
for experimenting with fallacy-related text classification and similarity-based
retrieval — without relying on proprietary data or systems.

---

## 🧠 Methods Included

- **TF–IDF + Logistic Regression** (scikit-learn)
- **Sentence-transformer embeddings** for semantic similarity
- **Cosine similarity–based retrieval**
- Structured, modular pipeline design

All examples use **synthetic or public-safe data only**.

---

## 📂 Repository Structure

fallacy-detection-baseline/
├── data/
│ └── sample_examples.jsonl # Synthetic example data
├── src/
│ ├── data_prep.py # Dataset loading utilities
│ ├── tfidf_baseline.py # TF–IDF + logistic regression model
│ └── retrieval.py # Embedding-based retrieval
├── notebooks/
│ └── demo.ipynb # Minimal usage demonstration
├── README.md
└── requirements.txt


---

## 🚀 Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

📓 Demo Notebook

A minimal Jupyter notebook is provided to demonstrate basic usage:

jupyter notebook notebooks/demo.ipynb
The notebook shows how to:

load the synthetic dataset

train a TF–IDF baseline classifier

run simple embedding-based retrieval

This is intended as a lightweight exploratory demo, not a production system.

📌 Scope & Notes

This repository is retrieval- and classification-focused

No proprietary datasets, models, or taxonomies are included

Designed for clarity, reproducibility, and experimentation

📜 License

MIT License
