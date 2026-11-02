"""Paths and shared constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"

DIALOGUE_PARQUET = DATA_PROCESSED / "dialogue.parquet"
SCORED_PARQUET = DATA_PROCESSED / "scored.parquet"

EPISODES = {1: "The Phantom Menace", 2: "Attack of the Clones", 3: "Revenge of the Sith"}
SCRIPT_FILES = {1: "StarWars1.txt", 2: "StarWars2.txt", 3: "StarWars3.txt"}

#: A character needs this many lines before it can carry a faction label. Below it,
#: a held-out fold would be a handful of sentences and its score pure noise.
MIN_LINES_FOR_LABEL = 8

#: Lines shorter than this are dropped from modelling ("Yes.", "Master?"). They carry
#: no rhetorical signal and would dominate the metrics.
MIN_WORDS = 3

RANDOM_STATE = 42


def script_dir() -> Path:
    """data/raw/ once fetched, or the local Script/ folder the project started from."""
    return DATA_RAW if (DATA_RAW / SCRIPT_FILES[1]).exists() else ROOT / "Script"
