"""Measuring the arc: trend, breaks, and whether either is real.

Everything here works on *levels*, never on cumulative sums, and every claim comes
with a null model.

The null model is a **rotation**: the series is circularly shifted against story
time, for every possible shift. Shuffling lines one by one would be the obvious
choice and the wrong one, because neighbouring lines of a scene are correlated, and
destroying that correlation makes the null too narrow and the p-values too small.
Rotation keeps the correlation intact and only breaks the link with story time. It
also makes the test exact: a series of n lines has n - 1 rotations, so the smallest
reachable p-value is 1/n.

Confidence intervals resample **scenes** rather than lines, for the same reason.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)


def _rotations(values: np.ndarray, statistic: Callable[[np.ndarray], float]) -> np.ndarray:
    """The statistic under every circular shift of the series."""
    return np.array([statistic(np.roll(values, s)) for s in range(1, len(values))])


def _p_value(observed: float, null: np.ndarray) -> float:
    """Share of the null at least as extreme, counting the observed series itself."""
    return float((1 + (null >= observed).sum()) / (len(null) + 1))


def rolling(values: np.ndarray, window: int = 60, n_boot: int = 500,
            rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Centred rolling mean with a 90% band. Descriptive, not a test: the band
    resamples lines within the window and ignores their correlation."""
    rng = rng or RNG
    s = pd.Series(values)
    mean = s.rolling(window, center=True, min_periods=max(5, window // 4)).mean()

    lo = np.full(len(values), np.nan)
    hi = np.full(len(values), np.nan)
    half = window // 2
    for i in range(len(values)):
        chunk = values[max(0, i - half):min(len(values), i + half + 1)]
        if len(chunk) < max(5, window // 4):
            continue
        boot = rng.choice(chunk, size=(n_boot, len(chunk)), replace=True).mean(axis=1)
        lo[i], hi[i] = np.percentile(boot, [5, 95])
    return pd.DataFrame({"mean": mean.to_numpy(), "lo": lo, "hi": hi})


def trend(position: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Spearman rho against story time, with a two-sided rotation p-value."""
    rank_x = stats.rankdata(position)
    rank_y = stats.rankdata(values)
    rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
    null = _rotations(rank_y, lambda r: abs(np.corrcoef(rank_x, r)[0, 1]))
    return {"rho": rho, "p_value": _p_value(abs(rho), null), "n": int(len(values))}


def scan(values: np.ndarray, min_size: int) -> tuple[np.ndarray, np.ndarray]:
    """For every admissible split, the variance a before/after step would explain."""
    n = len(values)
    csum = np.concatenate([[0.0], np.cumsum(values)])
    csq = np.concatenate([[0.0], np.cumsum(values ** 2)])
    total = csq[n] - csum[n] ** 2 / n
    idx = np.arange(min_size, n - min_size + 1)
    left = csq[idx] - csum[idx] ** 2 / idx
    right = (csq[n] - csq[idx]) - (csum[n] - csum[idx]) ** 2 / (n - idx)
    return idx, total - (left + right)


def change_point(values: np.ndarray, min_size: int) -> dict[str, float]:
    """The best single split, found without being told where to look, and tested.

    Any series has a best split, so the gain only means something against the best
    gain reachable once the series is rotated away from story time.
    """
    idx, gain = scan(values, min_size)
    best = int(idx[gain.argmax()])
    null = _rotations(values, lambda v: scan(v, min_size)[1].max())
    return {
        "index": best,
        "gain": float(gain.max()),
        "p_value": _p_value(float(gain.max()), null),
        "threshold_95": float(np.percentile(null, 95)),
        "before": float(values[:best].mean()),
        "after": float(values[best:].mean()),
    }


def step_at(values: np.ndarray, k: int) -> dict[str, float]:
    """Mean after line k minus mean before it, against the same split under rotation.

    One-sided: the hypothesis is a rise, stated by the story, not fitted to the data.
    """
    statistic = lambda v: float(v[k:].mean() - v[:k].mean())
    observed = statistic(values)
    null = _rotations(values, statistic)
    return {
        "index": int(k),
        "before": float(values[:k].mean()),
        "after": float(values[k:].mean()),
        "difference": observed,
        "p_value": _p_value(observed, null),
        "n_before": int(k),
        "n_after": int(len(values) - k),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values, for reading several tests as one family."""
    ordered = sorted(p_values, key=p_values.get)
    m, running, adjusted = len(ordered), 0.0, {}
    for i, key in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p_values[key]))
        adjusted[key] = running
    return adjusted


def _scene_groups(frame: pd.DataFrame) -> list[pd.DataFrame]:
    return [group for _, group in frame.groupby(["episode", "scene"], sort=False)]


def _resample_scenes(groups: list[pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    picks = rng.integers(0, len(groups), len(groups))
    return pd.concat([groups[i] for i in picks], ignore_index=True)


def slope_contrast(subject: pd.DataFrame, control: pd.DataFrame, value_col: str,
                   n_boot: int = 2_000, rng: np.random.Generator | None = None) -> dict[str, float]:
    """Is the subject's trend steeper than a control's? Interval resamples scenes."""
    rng = rng or RNG

    def slope(frame: pd.DataFrame) -> float:
        return float(np.polyfit(frame["position"], frame[value_col], 1)[0])

    subject_scenes, control_scenes = _scene_groups(subject), _scene_groups(control)
    diffs = np.array([slope(_resample_scenes(subject_scenes, rng))
                      - slope(_resample_scenes(control_scenes, rng)) for _ in range(n_boot)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "subject_slope": slope(subject),
        "control_slope": slope(control),
        "difference": slope(subject) - slope(control),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def difference_in_differences(subject: pd.DataFrame, control: pd.DataFrame, value_col: str,
                              position: float, n_boot: int = 2_000,
                              rng: np.random.Generator | None = None) -> dict[str, float]:
    """The subject's before/after jump minus a control's, at one narrative position."""
    rng = rng or RNG

    def jump(frame: pd.DataFrame) -> float:
        after = frame["position"] >= position
        if after.all() or not after.any():
            return np.nan
        return float(frame.loc[after, value_col].mean() - frame.loc[~after, value_col].mean())

    subject_scenes, control_scenes = _scene_groups(subject), _scene_groups(control)
    diffs = np.array([jump(_resample_scenes(subject_scenes, rng))
                      - jump(_resample_scenes(control_scenes, rng)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {
        "subject_jump": jump(subject),
        "control_jump": jump(control),
        "difference": jump(subject) - jump(control),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def locate_break(frame: pd.DataFrame, value_col: str, min_size: int, n_boot: int = 2_000,
                 rng: np.random.Generator | None = None) -> pd.Series:
    """How firmly the data pin the break down: the scene it lands on when the scenes
    themselves are resampled. Returns the scene number per usable resample."""
    rng = rng or RNG
    groups = _scene_groups(frame)
    landed = []
    for _ in range(n_boot):
        sample = _resample_scenes(groups, rng).sort_values("global_order")
        values = sample[value_col].to_numpy()
        if len(values) < 3 * min_size:
            continue
        idx, gain = scan(values, min_size)
        landed.append(int(sample["scene"].iloc[idx[gain.argmax()]]))
    return pd.Series(landed, dtype=int)
