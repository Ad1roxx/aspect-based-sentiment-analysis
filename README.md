# Aspect-Based Sentiment Analysis — Restaurant Reviews

Predicts sentiment (positive / negative / neutral) for a fixed set of aspect categories
— **food, service, ambiance, price, misc** — in restaurant reviews, served through a
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
├── explanations/            # per-sprint engineering notes
└── .github/workflows/       # CI
```

## Licence / data note

The SemEval-2014 Task 4 dataset is third-party licensed and is **not** redistributed in this
repository.
