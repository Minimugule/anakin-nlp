"""Fine-tune DistilBERT on the same task, under the same by-character protocol.

The question this module answers is narrow: does a model with contextual semantics
generalise to *unseen speakers* better than TF-IDF does? On the folds it was run on,
it does not clearly do so (scripts/04_analyze_arc.py compares them on the same folds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from . import config
from .models import _scores

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 96


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}


def _train_fold(texts_tr, y_tr, texts_te, tokenizer, dev, epochs: int, batch_size: int,
                lr: float) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(config.RANDOM_STATE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(dev)
    loader = DataLoader(LineDataset(texts_tr, y_tr, tokenizer), batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    # The light camp outnumbers the dark one ~4.5:1; without the weight the model
    # collapses to predicting "light" for everything and reports a flattering accuracy.
    counts = np.bincount(y_tr, minlength=2)
    weight = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimiser.zero_grad()
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            optimiser.step()

    model.eval()
    probs = []
    eval_loader = DataLoader(LineDataset(texts_te, np.zeros(len(texts_te), dtype=int), tokenizer),
                             batch_size=64)
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            batch.pop("labels")
            probs.append(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def cross_validate(frame: pd.DataFrame, grouped: bool = True, n_splits: int = 5,
                   epochs: int = 3, batch_size: int = 16, lr: float = 3e-5,
                   verbose: bool = True) -> tuple[dict[str, float], pd.DataFrame]:
    """Same protocol as ``models.cross_validate``, pooled out-of-fold predictions."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dev = device()
    X = frame["text"].to_numpy()
    y = (frame["faction"] == "dark").astype(int).to_numpy()
    groups = frame["character"].to_numpy()

    splitter = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
                if grouped else
                StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE))
    split_args = (X, y, groups) if grouped else (X, y)

    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(splitter.split(*split_args)):
        if verbose:
            held = sorted(set(groups[te])) if grouped else "random lines"
            print(f"  fold {k + 1}/{n_splits} on {dev.type} - held out: {held}", flush=True)
        oof[te] = _train_fold(X[tr], y[tr], X[te], tokenizer, dev, epochs, batch_size, lr)

    return _scores(y, (oof >= 0.5).astype(int), oof), frame.assign(p_dark=oof)


def fit_full(frame: pd.DataFrame, epochs: int = 3, batch_size: int = 16, lr: float = 3e-5):
    """Train on every labelled line and return (model, tokenizer) for scoring Anakin."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dev = device()
    X = frame["text"].to_numpy()
    y = (frame["faction"] == "dark").astype(int).to_numpy()

    torch.manual_seed(config.RANDOM_STATE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(dev)
    loader = DataLoader(LineDataset(X, y, tokenizer), batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    counts = np.bincount(y, minlength=2)
    weight = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimiser.zero_grad()
            loss_fn(model(**batch).logits, labels).backward()
            optimiser.step()
    return model, tokenizer


def predict(model, tokenizer, texts, batch_size: int = 64) -> np.ndarray:
    dev = device()
    model.eval()
    loader = DataLoader(LineDataset(texts, np.zeros(len(texts), dtype=int), tokenizer),
                        batch_size=batch_size)
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            batch.pop("labels")
            out.append(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)
