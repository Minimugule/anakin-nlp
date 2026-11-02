"""Download the three screenplays into data/raw/ (git-ignored)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anakin_nlp.scrape import fetch_all

if __name__ == "__main__":
    fetch_all()
