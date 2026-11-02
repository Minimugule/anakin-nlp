"""Fetch the three screenplays into data/raw/.

The scripts are third-party copyrighted material, so this repository ships the
fetcher rather than the texts: ``data/raw/`` is git-ignored and every user
regenerates it locally. Responses are cached on disk, so a re-run costs no request.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from . import config

#: url, output filename, and the CSS selector that isolates the script body.
SOURCES: dict[int, tuple[str, str, str]] = {
    1: ("https://imsdb.com/scripts/Star-Wars-The-Phantom-Menace.html",
        "StarWars1.txt", "td.scrtext"),
    2: ("http://sellascript.com/Source/resources/screenplays/attackoftheclones.htm",
        "StarWars2.txt", "pre"),
    3: ("https://imsdb.com/scripts/Star-Wars-Revenge-of-the-Sith.html",
        "StarWars3.txt", "td.scrtext"),
}

HEADERS = {"User-Agent": "anakin-nlp/1.0 (research; contact via repository)"}
MIN_CHARS = 50_000  # a real screenplay; anything shorter means the selector missed


def fetch(episode: int, out_dir: Path | None = None, pause: float = 1.0) -> Path:
    """Download one screenplay. Returns the path written."""
    url, filename, selector = SOURCES[episode]
    out_dir = out_dir or config.DATA_RAW
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename

    if target.exists() and target.stat().st_size > MIN_CHARS:
        print(f"episode {episode}: cached ({target.stat().st_size:,} bytes)")
        return target

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    blocks = BeautifulSoup(response.text, "html.parser").select(selector)
    text = "\n".join(block.get_text() for block in blocks) + "\n"

    if len(text) < MIN_CHARS:
        raise RuntimeError(
            f"episode {episode}: selector {selector!r} returned {len(text):,} chars "
            f"(expected > {MIN_CHARS:,}). The page layout has probably changed.")

    target.write_text(text, encoding="utf-8")
    print(f"episode {episode}: {len(text):,} chars -> {target}")
    time.sleep(pause)
    return target


def fetch_all(out_dir: Path | None = None) -> list[Path]:
    return [fetch(ep, out_dir) for ep in sorted(SOURCES)]
