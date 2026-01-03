# Fallacy Detection Baseline

A public-safe research baseline for fallacy detection using classical NLP and lightweight semantic retrieval.

The goal of this repository is to provide a minimal, reproducible starting point for experimenting with fallacy-related text classification and similarity-based retrieval — without relying on proprietary data or systems.

---

## Methods Included

- TF–IDF + Logistic Regression (scikit-learn)
- Sentence-transformer embeddings for semantic similarity
- Cosine similarity–based retrieval
- Structured, modular pipeline design

All examples use synthetic or public-safe data only.

---

## Repository Structure

fallacy-detection-baseline/
├── data/
│   └── sample_examples.jsonl        # Synthetic example data
├── examples/
│   └── example_output.json          # Representative output format
├── src/
│   ├── data_prep.py                 # Dataset loading utilities
│   ├── tfidf_baseline.py            # TF–IDF + logistic regression model
│   └── retrieval.py                 # Embedding-based retrieval
├── notebooks/
│   └── demo.ipynb                   # Minimal usage demonstration
├── requirements.txt
└── README.md

---

## Quick Start

Install dependencies:

pip install -r requirements.txt

Run the demo notebook:

jupyter notebook notebooks/demo.ipynb

The notebook shows how to:

- load the synthetic dataset
- train a TF–IDF baseline classifier
- run simple embedding-based retrieval

This is intended as a lightweight exploratory demo, not a production system.

---

## Example Output

A representative example of the system’s output format is available in:

examples/example_output.json

---

## Scope & Notes

- Retrieval- and classification-focused baseline
- No proprietary datasets, models, or taxonomies included
- Designed for clarity, reproducibility, and experimentation

---

## License

MIT License
