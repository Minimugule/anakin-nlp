"""Story events, located from the screenplay rather than from the scores."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Event:
    episode: int
    scene: int   # numbered as in parse.py: the count of scene headings in the episode
    label: str


#: Anakin says nothing during the massacre of the Tusken camp (scenes 117-119 of the
#: Episode II screenplay). Scene 125, the homestead garage, is where he first speaks
#: afterwards and tells Padme what he did. So in dialogue, this is the first line at
#: which the massacre can show.
TUSKEN_MASSACRE = Event(episode=2, scene=125, label="Tusken massacre")


def split_index(lines: pd.DataFrame, event: Event) -> int:
    """Rank of the first line at or after the event, in lines sorted by story order."""
    after = (lines["episode"] > event.episode) | (
        (lines["episode"] == event.episode) & (lines["scene"] >= event.scene))
    return int(after.to_numpy().argmax())
