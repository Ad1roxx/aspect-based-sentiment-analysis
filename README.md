# Aspect-Based Sentiment Analysis — Restaurant Reviews

Predicts sentiment (positive / negative / neutral) for a fixed set of aspect categories
— **food, service, ambience, price, misc** — in restaurant reviews, served through a
FastAPI backend and a React frontend.

Trained on [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/) restaurant data.

> **Status:** in active development. Full architecture diagram, setup instructions and API
> docs land once the pipeline is complete.

## Planned architecture

```
React page  →  FastAPI (/predict, /health, /model-info)  →  DistilBERT ABSA model
                                                              ↑
                                                       MLflow (params, metrics,
                                                       confusion matrix, artifact)
```

A single DistilBERT encoder with five 4-way classification heads — one per aspect, each
predicting *not mentioned / positive / negative / neutral*. One forward pass produces all
five predictions, and each head's softmax gives the per-aspect confidence the UI displays.

## Repo layout

```
├── ml/
│   ├── src/                 # data loading, training, evaluation, explainability
│   └── models/              # trained artifact (gitignored)
├── api/                     # FastAPI service
├── frontend/                # React prediction page
├── tests/                   # pytest: model inference + API
├── scripts/
│   └── download_data.py     # fetch + checksum-verify the SemEval-2014 data
├── data/                    # raw XML (gitignored, fetched by script)
├── explanations/            # per-sprint engineering notes
└── .github/workflows/       # CI
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python scripts/download_data.py
```

`requirements.txt` pins the CUDA 12.6 build of PyTorch for GPU training. Serving and CI run
CPU-only and will use a separate dependency profile.

## Licence / data note

The SemEval-2014 Task 4 dataset is third-party licensed and is **not** redistributed here.
`scripts/download_data.py` fetches it from pinned upstream commits and verifies each file
against a known SHA-256 digest. See [data/README.md](data/README.md).
