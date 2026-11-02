r"""Turn three differently-formatted screenplays into one tidy dialogue table.

The three sources do not share a layout: episode I uses ``NAME : line`` with a space
before the colon, episodes II and III use ``NAME: line``, scene headings are
``INT./EXT.`` in I and II and numbered in III, and dialogue wraps freely across
lines in all three.

Unlike the original pipeline, punctuation and casing are **preserved** here. The
downstream sentiment and transformer models are trained on ordinary English text;
stripping apostrophes and full stops with ``re.sub(r"[^\w\s:]", "", ...)`` degrades
them for no gain. Cleaning that a specific model needs belongs to that model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from .characters import canonical

#: ``NAME:`` or ``NAME :`` at the start of a line. The name is uppercase (accents
#: included, for PADMÉ and CORDÉ), at most a few words, and must contain a letter.
CUE = re.compile(r"^\s*([A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9'\-\. ]{1,28}?)\s*:\s*(.*)$")

#: Scene headings: ``INT. X``, ``EXTERIOR: Y``, or episode III's ``12 EXT. Z``.
HEADING = re.compile(r"^\s*(?:\d+\s+)?(?:INT|EXT|INTERIOR|EXTERIOR)\b", re.I)

#: Stage directions inside a line: ``(V.O.)``, ``(cont'd)``, ``(off screen voice)``.
PARENTHETICAL = re.compile(r"\([^)]*\)")

#: Collapsed whitespace and stray screenplay artefacts.
WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Line:
    episode: int
    scene: int
    order: int          # position of the line within its episode
    character: str      # canonical name
    cue: str            # the raw cue, kept for auditing the alias table
    text: str


def _clean(text: str) -> str:
    text = PARENTHETICAL.sub(" ", text)
    return WS.sub(" ", text).strip(" -")


def parse_script(path: Path, episode: int) -> list[Line]:
    """Extract every attributable line of dialogue from one screenplay."""
    raw = path.read_text(encoding="utf-8").splitlines()
    lines: list[Line] = []
    scene, order, i = 0, 0, 0

    while i < len(raw):
        line = raw[i]
        if HEADING.match(line):
            scene += 1
            i += 1
            continue

        match = CUE.match(line)
        if not match:
            i += 1
            continue

        cue, first = match.group(1).strip(), match.group(2)
        character = canonical(cue)
        if character is None:          # scene heading, crowd cue or untracked role
            i += 1
            continue

        # Absorb wrapped continuation lines: stop at a blank line, the next cue,
        # or a scene heading.
        parts, j = [first], i + 1
        while j < len(raw):
            nxt = raw[j]
            if not nxt.strip() or CUE.match(nxt) or HEADING.match(nxt):
                break
            parts.append(nxt)
            j += 1

        text = _clean(" ".join(parts))
        if len(text) > 1:
            order += 1
            lines.append(Line(episode, scene, order, character, cue, text))
        i = j

    return lines


def scene_headings(path: Path) -> dict[int, str]:
    """Scene number -> heading, numbered exactly as ``parse_script`` numbers scenes."""
    headings, scene = {}, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if HEADING.match(line):
            scene += 1
            headings[scene] = line.strip()
    return headings


def build_dataset(script_dir: Path, files: dict[int, str]) -> pd.DataFrame:
    """Parse every episode and return one ordered dialogue table.

    ``position`` is the narrative clock the whole analysis runs on: the line's rank
    across the trilogy, rescaled to [0, 1]. It is the only time axis available from
    the scripts alone, and it is monotone by construction, so any trend measured
    against it is a trend in story time.
    """
    frames = []
    for episode, filename in sorted(files.items()):
        rows = parse_script(script_dir / filename, episode)
        if not rows:
            raise ValueError(f"no dialogue parsed from {filename}")
        frames.append(pd.DataFrame([asdict(r) for r in rows]))

    df = pd.concat(frames, ignore_index=True)
    df["n_words"] = df["text"].str.split().str.len()
    df = df.sort_values(["episode", "order"], ignore_index=True)
    df["global_order"] = range(1, len(df) + 1)
    df["position"] = (df["global_order"] - 1) / (len(df) - 1)
    return df
