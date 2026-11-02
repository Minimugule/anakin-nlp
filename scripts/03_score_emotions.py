"""Attach 7-way emotion probabilities to every line of dialogue."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anakin_nlp import config
from anakin_nlp.emotions import EMOTIONS, dark_affect, score_emotions


def main() -> None:
    source = config.SCORED_PARQUET if config.SCORED_PARQUET.exists() else config.DIALOGUE_PARQUET
    df = pd.read_parquet(source)
    print(f"scoring {len(df):,} lines with {Path(source).name} as the base")

    emotions = score_emotions(df["text"].tolist())
    for col in EMOTIONS:
        df[col] = emotions[col].to_numpy()
    df["dark_affect"] = dark_affect(emotions).to_numpy()

    df.to_parquet(config.SCORED_PARQUET, index=False)
    print(f"-> {config.SCORED_PARQUET.relative_to(config.ROOT)}")

    focus = df[df["character"].isin(["ANAKIN", "OBI-WAN", "PADME", "PALPATINE"])]
    print("\nmean emotion by character")
    print(focus.groupby("character")[EMOTIONS + ["dark_affect"]].mean().round(3).to_string())
    print("\nAnakin, mean emotion by episode")
    print(df[df.character == "ANAKIN"].groupby("episode")[EMOTIONS + ["dark_affect"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
