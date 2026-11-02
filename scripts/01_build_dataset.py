"""Parse the three screenplays into data/processed/dialogue.parquet."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anakin_nlp import config
from anakin_nlp.characters import SUBJECT, UNALIGNED, faction
from anakin_nlp.parse import build_dataset


def main() -> None:
    df = build_dataset(config.script_dir(), config.SCRIPT_FILES)

    df["faction"] = df["character"].map(faction)
    df["is_subject"] = df["character"] == SUBJECT
    df.loc[df["is_subject"], "faction"] = None

    # A label only survives if its character speaks enough to be a meaningful
    # held-out group.
    counts = df["character"].value_counts()
    thin = counts[counts < config.MIN_LINES_FOR_LABEL].index
    df.loc[df["character"].isin(thin), "faction"] = None

    df["role"] = "unaligned"
    df.loc[df["faction"].notna(), "role"] = df["faction"]
    df.loc[df["is_subject"], "role"] = "subject"

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DIALOGUE_PARQUET, index=False)

    print(f"{len(df):,} lines from {df['character'].nunique()} characters "
          f"-> {config.DIALOGUE_PARQUET.relative_to(config.ROOT)}")
    print(df.groupby("role").agg(lines=("text", "size"), characters=("character", "nunique")))
    print(f"\nlabelled characters: light {sorted(df[df.faction=='light'].character.unique())}")
    print(f"                     dark  {sorted(df[df.faction=='dark'].character.unique())}")
    print(f"\ndeliberately unlabelled: {len(UNALIGNED)} characters "
          f"({df[df.role=='unaligned'].shape[0]} lines)")


if __name__ == "__main__":
    main()
