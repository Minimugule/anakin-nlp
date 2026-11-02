"""Faction classifier, and the validation protocol that makes it mean anything.

The central problem with classifying "light side vs dark side" from film dialogue is
that the label is attached to a *speaker*, not to a sentence. A model trained on a
random split of lines sees Yoda in training and Yoda in test, so it can score well
by learning who talks like Yoda rather than anything about the two camps' rhetoric.
That skill is worth nothing on a character it has never seen.

So every honest number here comes from ``StratifiedGroupKFold`` grouped **by
character**: a character's lines are entirely in training or entirely in test,
never both. ``grouped=False`` reproduces the leaky random split on purpose, so the
report can show the size of the gap rather than assert it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion

from . import config


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Labelled lines long enough to carry rhetoric."""
    return df[df["faction"].notna() & (df["n_words"] >= config.MIN_WORDS)].reset_index(drop=True)


def subject_frame(df: pd.DataFrame, character: str) -> pd.DataFrame:
    return df[(df["character"] == character) & (df["n_words"] >= config.MIN_WORDS)].reset_index(drop=True)


def build_pipeline() -> Pipeline:
    """TF-IDF over words *and* character n-grams, then a linear model.

    Character n-grams add morphology and register on top of whole words.
    ``class_weight='balanced'`` is not optional here: the light camp outnumbers the
    dark one roughly five to one.
    """
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                     strip_accents="unicode", lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     sublinear_tf=True, lowercase=True)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                   random_state=config.RANDOM_STATE)),
    ])


def _scores(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "dark_f1": f1_score(y, pred, pos_label=1),
        "roc_auc": roc_auc_score(y, proba),
    }


def cross_validate(frame: pd.DataFrame, grouped: bool = True, n_splits: int = 5,
                   seed: int | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    """Return (metrics, per-line out-of-fold predictions).

    Metrics are computed once over the **pooled** out-of-fold predictions rather than
    averaged per fold. With only seven dark-side characters, a grouped fold can
    legitimately contain none of them, which makes per-fold ROC-AUC undefined and
    per-fold F1 wildly unstable. Pooling sidesteps that; ``repeated_cv`` supplies the
    uncertainty instead, by re-running the whole split under different seeds.

    ``grouped=True`` holds each character out whole; ``grouped=False`` is the naive
    random split, kept so the report can quantify the leak rather than assert it.
    """
    X = frame["text"].to_numpy()
    y = (frame["faction"] == "dark").astype(int).to_numpy()
    groups = frame["character"].to_numpy()
    seed = config.RANDOM_STATE if seed is None else seed

    splitter = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                if grouped else
                StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    split_args = (X, y, groups) if grouped else (X, y)

    oof = np.zeros(len(y))
    fold_id = np.zeros(len(y), dtype=int)
    for k, (tr, te) in enumerate(splitter.split(*split_args)):
        model = build_pipeline().fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
        fold_id[te] = k

    metrics = _scores(y, (oof >= 0.5).astype(int), oof)
    return metrics, frame.assign(p_dark=oof, fold=fold_id)


def repeated_cv(frame: pd.DataFrame, grouped: bool = True, n_splits: int = 5,
                n_repeats: int = 5) -> tuple[dict[str, float], pd.DataFrame]:
    """Run ``cross_validate`` under several seeds; report mean and spread.

    The spread across seeds is the honest error bar for a dataset this small: the
    grouped ROC-AUC runs from 0.50 to 0.68 depending on which characters share a fold.
    """
    runs, last = [], None
    for r in range(n_repeats):
        m, last = cross_validate(frame, grouped=grouped, n_splits=n_splits,
                                 seed=config.RANDOM_STATE + r)
        runs.append(m)
    summary = {k: float(np.mean([run[k] for run in runs])) for k in runs[0]}
    summary |= {f"{k}_std": float(np.std([run[k] for run in runs])) for k in runs[0]}
    return summary, last


def fit_full(frame: pd.DataFrame) -> Pipeline:
    """Fit on every labelled line - used to score Anakin, who is never labelled."""
    X = frame["text"].to_numpy()
    y = (frame["faction"] == "dark").astype(int).to_numpy()
    return build_pipeline().fit(X, y)


def top_terms(model: Pipeline, n: int = 20) -> pd.DataFrame:
    """The word features pulling hardest toward each camp - the model's own reading."""
    word_vec = model.named_steps["features"].transformer_list[0][1]
    names = word_vec.get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0][:len(names)]
    order = np.argsort(coef)
    return pd.DataFrame({
        "dark_term": names[order[::-1][:n]], "dark_weight": coef[order[::-1][:n]],
        "light_term": names[order[:n]], "light_weight": coef[order[:n]],
    })
