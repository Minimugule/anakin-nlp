"""Train and validate the faction classifiers, and score every character.

Writes:
  reports/faction_metrics.json   - the random-vs-grouped comparison for both models
  data/processed/scored.parquet  - a P(dark) per line, from both models
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anakin_nlp import config, models, transformer
from anakin_nlp.characters import SUBJECT

CONTROLS = ["OBI-WAN", "PADME", "PALPATINE"]


def main(skip_transformer: bool = False) -> None:
    df = pd.read_parquet(config.DIALOGUE_PARQUET)
    frame = models.modelling_frame(df)
    metrics_path = config.REPORTS / "faction_metrics.json"

    # A TF-IDF-only run must not erase the transformer results it did not recompute.
    keep_existing = skip_transformer and metrics_path.exists() and config.SCORED_PARQUET.exists()
    results: dict[str, dict] = json.loads(metrics_path.read_text()) if keep_existing else {}

    print(f"{len(frame)} labelled lines, {frame.character.nunique()} characters "
          f"({frame[frame.faction=='dark'].character.nunique()} dark)\n")

    # --- TF-IDF baseline, both protocols -------------------------------------
    print("TF-IDF + logistic regression")
    for grouped, key in [(False, "tfidf_naive"), (True, "tfidf_grouped")]:
        t0 = time.time()
        metrics, oof = models.repeated_cv(frame, grouped=grouped)
        results[key] = metrics | {"seconds": round(time.time() - t0, 1)}
        print(f"  {'grouped' if grouped else 'naive  '} -> ROC-AUC {metrics['roc_auc']:.3f} "
              f"±{metrics['roc_auc_std']:.3f} | macro-F1 {metrics['macro_f1']:.3f}")
        if grouped:
            oof_tfidf = oof

    # --- DistilBERT, both protocols ------------------------------------------
    if not skip_transformer:
        print("\nDistilBERT fine-tuning")
        for grouped, key in [(True, "distilbert_grouped"), (False, "distilbert_naive")]:
            t0 = time.time()
            print(f" protocol: {'grouped by character' if grouped else 'naive random split'}")
            metrics, oof = transformer.cross_validate(frame, grouped=grouped)
            results[key] = metrics | {"seconds": round(time.time() - t0, 1)}
            print(f"  -> ROC-AUC {metrics['roc_auc']:.3f} | macro-F1 {metrics['macro_f1']:.3f} "
                  f"({results[key]['seconds']:.0f}s)")
            if grouped:
                oof_bert = oof

    # --- score every character ------------------------------------------------
    # Labelled characters get their out-of-fold score (the model never saw them in
    # training). Anakin is never labelled, so a model fitted on all labelled lines
    # has no access to him either. Both are leak-free by construction.
    print("\nscoring characters")
    scored = (pd.read_parquet(config.SCORED_PARQUET) if keep_existing else df).set_index("global_order")
    scored["p_dark_tfidf"] = oof_tfidf.set_index("global_order")["p_dark"]

    # Labelled characters already carry an out-of-fold score. Anakin is never
    # labelled, so a model fitted on every labelled line has not seen him either.
    full = models.fit_full(frame)
    sub = models.subject_frame(df, SUBJECT)
    scored.loc[sub["global_order"], "p_dark_tfidf"] = full.predict_proba(sub["text"])[:, 1]

    if not skip_transformer:
        scored["p_dark_bert"] = oof_bert.set_index("global_order")["p_dark"]
        model, tok = transformer.fit_full(frame)
        sub = models.subject_frame(df, SUBJECT)
        scored.loc[sub["global_order"], "p_dark_bert"] = transformer.predict(model, tok, sub["text"].tolist())

    scored = scored.reset_index()
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(config.SCORED_PARQUET, index=False)
    metrics_path.write_text(json.dumps(results, indent=2))

    print(f"\n-> {config.SCORED_PARQUET.relative_to(config.ROOT)}")
    print(f"-> {(config.REPORTS / 'faction_metrics.json').relative_to(config.ROOT)}")
    top = models.top_terms(full, n=12)
    print("\nstrongest word features")
    print(top.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


if __name__ == "__main__":
    main(skip_transformer="--skip-transformer" in sys.argv)
