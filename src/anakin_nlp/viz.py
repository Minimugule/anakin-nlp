"""Figure style and the report's charts.

House rules, applied everywhere: one measure per axis (never a second y-scale),
2px lines and thin marks, a recessive hairline grid, every series both in the legend
and directly labelled at its end so identity is never carried by colour alone, and
colours drawn only from the validated palette in ``palette.py``.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .palette import CHARACTER_COLORS, DARK_SIDE, INK, LIGHT_SIDE

EPISODE_LABEL = {1: "I · The Phantom Menace", 2: "II · Attack of the Clones",
                 3: "III · Revenge of the Sith"}


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"], "savefig.dpi": 200, "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "text.color": INK["primary"], "axes.labelcolor": INK["secondary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def _title(ax, title: str, subtitle: str | None = None, gap: float = 0.055) -> None:
    """Title and subtitle stacked above the axes, never overlapping each other."""
    ax.text(0, 1.0 + (gap * 2 if subtitle else gap), title, transform=ax.transAxes,
            fontsize=13, fontweight="600", color=INK["primary"], va="bottom", ha="left")
    if subtitle:
        ax.text(0, 1.0 + gap * 0.55, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=INK["secondary"], va="bottom", ha="left")


def _spread_labels(entries, min_gap):
    """Nudge end-of-line labels apart so two series never write over each other."""
    entries = sorted(entries, key=lambda e: e[1])
    for i in range(1, len(entries)):
        if entries[i][1] - entries[i - 1][1] < min_gap:
            entries[i] = (entries[i][0], entries[i - 1][1] + min_gap, *entries[i][2:])
    return entries


def _episode_bands(ax, df: pd.DataFrame, label_y: float = 0.985) -> None:
    """Mark where each episode starts on the shared narrative-time axis.

    Labels sit inside the plot area: above it they would compete with the subtitle.
    """
    for ep, chunk in df.groupby("episode"):
        start = chunk["position"].min()
        ax.axvline(start, color=INK["axis"], lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.text(start + 0.006, label_y, EPISODE_LABEL[ep], transform=ax.get_xaxis_transform(),
                fontsize=8.5, color=INK["muted"], va="top", ha="left")


def leakage_figure(metrics: dict, path, subtitle: str) -> None:
    """The drop between the two protocols, as a dumbbell: the point is the size of the gap."""
    use_style()
    rows = [("TF-IDF + logistic regression", metrics["tfidf_naive"]["roc_auc"],
             metrics["tfidf_grouped"]["roc_auc"])]
    if "distilbert_naive" in metrics:
        rows.append(("DistilBERT (fine-tuned)", metrics["distilbert_naive"]["roc_auc"],
                     metrics["distilbert_grouped"]["roc_auc"]))

    fig, ax = plt.subplots(figsize=(8.4, 1.35 + 0.9 * len(rows)))
    for i, (name, naive, grouped) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([grouped, naive], [y, y], color=INK["axis"], lw=2, zorder=1,
                solid_capstyle="round")
        ax.scatter([naive], [y], s=110, color=DARK_SIDE, zorder=3,
                   edgecolor=INK["surface"], linewidth=2)
        ax.scatter([grouped], [y], s=110, color=LIGHT_SIDE, zorder=3,
                   edgecolor=INK["surface"], linewidth=2)
        ax.text(naive, y + 0.24, f"{naive:.2f}", ha="center", fontsize=9.5,
                color=INK["primary"], fontweight="600")
        ax.text(grouped, y + 0.24, f"{grouped:.2f}", ha="center", fontsize=9.5,
                color=INK["primary"], fontweight="600")
        ax.text(0.457, y - 0.26, name, ha="left", va="center", fontsize=9.5,
                color=INK["secondary"])

    ax.axvline(0.5, color=INK["muted"], lw=1, ls=(0, (2, 3)), zorder=0)
    ax.text(0.5, len(rows) - 0.42, "chance", fontsize=8.5, color=INK["muted"], ha="center")
    ax.set_xlim(0.45, 0.92)
    ax.set_ylim(-0.55, len(rows) - 0.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ROC-AUC, dark side vs light side")
    ax.grid(axis="y", visible=False)
    _title(ax, "The score falls as soon as the test characters are unseen", subtitle)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=DARK_SIDE,
                          label="Random split of lines (a character appears in train and test)"),
               plt.Line2D([], [], marker="o", ls="", ms=9, color=LIGHT_SIDE,
                          label="Split by character (every test character is unseen)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.22), fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _mark_events(ax, events, y: float = 0.02) -> None:
    """Dotted line per story event, labelled beside it on a surface-coloured plate."""
    for x, label in events or []:
        ax.axvline(x, color=INK["primary"], lw=1.1, ls=(0, (1, 2)), zorder=4)
        ax.annotate(label, (x, y), xycoords=ax.get_xaxis_transform(), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8.5,
                    color=INK["primary"], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.3", fc=INK["surface"], ec="none", alpha=0.85))


def arc_figure(series: dict[str, pd.DataFrame], df: pd.DataFrame, value_label: str,
               title: str, subtitle: str, path, events=None) -> None:
    """Rolling level over narrative time: the subject against control characters."""
    use_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    _episode_bands(ax, df)

    ends = []
    for name, frame in series.items():
        colour = CHARACTER_COLORS.get(name, INK["muted"])
        is_subject = name == "ANAKIN"
        if is_subject:
            ax.fill_between(frame["position"], frame["lo"], frame["hi"],
                            color=colour, alpha=0.11, linewidth=0, zorder=1)
        ax.plot(frame["position"], frame["mean"], color=colour,
                lw=2.4 if is_subject else 1.5, alpha=1.0 if is_subject else 0.7,
                zorder=3 if is_subject else 2)
        valid = frame.dropna(subset=["mean"])
        if len(valid):
            ends.append((name, float(valid["mean"].iloc[-1]), float(valid["position"].iloc[-1]), colour))

    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for name, y, x, colour in _spread_labels(ends, min_gap=span * 0.055):
        ax.annotate(name.title(), (x, y), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=colour,
                    fontweight="600" if name == "ANAKIN" else "normal")

    _mark_events(ax, events)
    ax.set_xlim(-0.005, 1.12)
    ax.set_xlabel("Narrative time  ·  position of the line across the three screenplays")
    ax.set_ylabel(value_label)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Line2D([], [], color=CHARACTER_COLORS.get(n, INK["muted"]),
                                  lw=2.4 if n == "ANAKIN" else 1.5, label=n.title())
                       for n in series],
              loc="upper left", ncol=len(series), fontsize=9, bbox_to_anchor=(0, -0.13))
    _title(ax, title, subtitle)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def episode_figure(df: pd.DataFrame, characters: list[str], value_col: str,
                   value_label: str, title: str, subtitle: str, path,
                   n_boot: int = 2000, seed: int = 42) -> None:
    """Per-episode means with bootstrap 95% intervals.

    The rolling curve shows shape; this shows whether the shape survives its own
    error bars. Intervals resample each character's scenes within each episode, so a
    character with few scenes gets a visibly wider one.
    """
    use_style()
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    offsets = np.linspace(-0.16, 0.16, len(characters))
    ends = []

    for name, dx in zip(characters, offsets):
        colour = CHARACTER_COLORS.get(name, INK["muted"])
        xs, ys, los, his = [], [], [], []
        for ep in sorted(df["episode"].unique()):
            sub = df[(df["character"] == name) & (df["episode"] == ep)].dropna(subset=[value_col])
            if len(sub) < 10:
                continue
            scenes = [g[value_col].to_numpy() for _, g in sub.groupby("scene")]
            boot = [np.concatenate([scenes[i] for i in rng.integers(0, len(scenes), len(scenes))]).mean()
                    for _ in range(n_boot)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            xs.append(ep + dx); ys.append(sub[value_col].mean()); los.append(lo); his.append(hi)
        if not xs:
            continue
        is_subject = name == "ANAKIN"
        ax.plot(xs, ys, color=colour, lw=2.2 if is_subject else 1.4,
                alpha=1.0 if is_subject else 0.65, zorder=2, ls="-" if is_subject else (0, (4, 2)))
        ax.vlines(xs, los, his, color=colour, lw=2 if is_subject else 1.4,
                  alpha=0.9 if is_subject else 0.55, zorder=3)
        ax.scatter(xs, ys, s=52 if is_subject else 34, color=colour, zorder=4,
                   edgecolor=INK["surface"], linewidth=1.6)
        ends.append((name, ys[-1], xs[-1], colour))

    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for name, y, x, colour in _spread_labels(ends, min_gap=span * 0.062):
        ax.annotate(name.title(), (x, y), xytext=(11, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=colour,
                    fontweight="600" if name == "ANAKIN" else "normal")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([EPISODE_LABEL[e] for e in (1, 2, 3)], fontsize=9)
    ax.set_xlim(0.7, 3.55)
    ax.set_ylabel(value_label)
    ax.grid(axis="x", visible=False)
    _title(ax, title, subtitle)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def emotion_panels(df: pd.DataFrame, characters: list[str], emotions: list[str],
                   window: int, path, rolling_fn, event_position: float | None = None) -> None:
    """Small multiples: one emotion per panel, one line per character, shared scale."""
    use_style()
    fig, axes = plt.subplots(1, len(emotions), figsize=(3.3 * len(emotions), 3.5), sharey=True)
    for ax, emotion in zip(np.atleast_1d(axes), emotions):
        for name in characters:
            sub = df[df["character"] == name].sort_values("global_order")
            if len(sub) < window // 2:
                continue
            smooth = rolling_fn(sub[emotion].to_numpy(), window=window, n_boot=1)
            colour = CHARACTER_COLORS.get(name, INK["muted"])
            ax.plot(sub["position"].to_numpy(), smooth["mean"], color=colour,
                    lw=2.2 if name == "ANAKIN" else 1.4,
                    alpha=1.0 if name == "ANAKIN" else 0.7)
        if event_position is not None:
            ax.axvline(event_position, color=INK["primary"], lw=1.0, ls=(0, (1, 2)), zorder=4)
        ax.set_title(emotion.title(), loc="left", fontsize=10.5, fontweight="600",
                     color=INK["primary"], pad=6)
        ax.set_xlabel("narrative time")
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("rolling mean probability")
    fig.legend(handles=[plt.Line2D([], [], color=CHARACTER_COLORS.get(n, INK["muted"]),
                                   lw=2.2 if n == "ANAKIN" else 1.4, label=n.title())
                        for n in characters],
               loc="lower center", ncol=len(characters), fontsize=9,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Anger carries the rise; fear barely moves",
                 x=0.005, y=1.14, ha="left", fontsize=13, fontweight="600",
                 color=INK["primary"])
    fig.text(0.005, 1.055, "Same emotion model, same rolling window, shared vertical scale · "
             "dotted line = Tusken massacre",
             ha="left", fontsize=9.5, color=INK["secondary"])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def terms_figure(top: pd.DataFrame, path, n: int = 12) -> None:
    """Diverging bars: the vocabulary each camp is recognised by."""
    use_style()
    fig, ax = plt.subplots(figsize=(8.2, 0.235 * 2 * n + 1.1))
    dark = top.head(n)[["dark_term", "dark_weight"]].to_numpy()
    light = top.head(n)[["light_term", "light_weight"]].to_numpy()
    items = [(t, w, DARK_SIDE) for t, w in dark][::-1] + [(t, w, LIGHT_SIDE) for t, w in light]

    ys = np.arange(len(items))
    for y, (term, weight, colour) in zip(ys, items):
        ax.barh(y, weight, color=colour, height=0.62, zorder=2)
        offset = 4 if weight > 0 else -4
        ax.text(weight + offset * 0.01, y, term, va="center",
                ha="left" if weight > 0 else "right", fontsize=9, color=INK["primary"])
    ax.axvline(0, color=INK["axis"], lw=1)
    ax.set_yticks([])
    ax.set_ylim(-0.8, len(items) - 0.2)
    ax.set_xlabel("logistic-regression weight  ·  negative = light side, positive = dark side")
    ax.grid(axis="y", visible=False)
    _title(ax, "What the classifier actually keys on",
           "Strongest word features, model fitted on all labelled characters", gap=0.022)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def massacre_figure(ep2: pd.DataFrame, k: int, blind: int, scans: dict[str, tuple], step: dict, path) -> None:
    """Anakin in Episode II: what the lines do, and where the data alone would split them.

    The lower panel is the statistical half of the claim. It is computed without being
    told where the massacre is, so its peak can land anywhere in the episode, and it
    shows the measure fixed in advance next to anger alone, the one that clears the bar.
    """
    use_style()
    red = CHARACTER_COLORS["ANAKIN"]
    fig, (top, low) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, height_ratios=[2.3, 1],
                                   gridspec_kw={"hspace": 0.1})
    x = np.arange(len(ep2))
    values = ep2["dark_affect"].to_numpy()
    smooth = pd.Series(values).rolling(15, center=True, min_periods=8).mean().to_numpy()

    top.scatter(x, values, s=13, color=INK["muted"], alpha=0.4, linewidth=0, zorder=2)
    top.plot(x, smooth, color=red, lw=2.4, zorder=3)
    for start, stop, level, ha, tx in [(0, k, step["before"], "left", 46),
                                       (k, len(x) - 1, step["after"], "left", k + 2)]:
        top.plot([start, stop], [level, level], color=INK["primary"], lw=1.4, ls=(0, (4, 2)), zorder=4)
        top.text(tx, level + 0.02, f"mean {level:.2f}", ha=ha, va="bottom",
                 fontsize=10, fontweight="600", color=INK["primary"], zorder=5,
                 bbox=dict(boxstyle="round,pad=0.2", fc=INK["surface"], ec="none", alpha=0.85))
    for ax in (top, low):
        ax.axvline(k - 0.5, color=INK["primary"], lw=1.2, ls=(0, (1, 2)), zorder=4)
        ax.grid(axis="x", visible=False)
    top.annotate("Tusken massacre:\nhis first words afterwards", (k - 0.5, 0.97),
                 xycoords=top.get_xaxis_transform(), xytext=(-7, 0), textcoords="offset points",
                 ha="right", va="top", fontsize=9.5, color=INK["primary"], linespacing=1.35)
    top.set_ylabel("P(anger) + P(fear), per line")
    top.set_ylim(-0.02, 1.02)

    styles = {"anger + fear": "-", "anger alone": (0, (4, 2))}
    peak = 0.0
    for name, (idx, gain, threshold, p_value) in scans.items():
        if styles[name] == "-":
            low.fill_between(idx, 0, gain, color=red, alpha=0.14, linewidth=0, zorder=1)
        low.plot(idx, gain, color=red, lw=2 if styles[name] == "-" else 1.6, ls=styles[name], zorder=3)
        low.plot([idx[0], idx[-1]], [threshold, threshold], color=INK["secondary"], lw=0.9,
                 ls=styles[name], zorder=2)
        best = idx[gain.argmax()]
        peak = max(peak, float(gain.max()))
        low.scatter([best], [gain.max()], s=44, color=red, zorder=5, edgecolor=INK["surface"], linewidth=1.5)
        right = styles[name] != "-"
        low.annotate(f"{name}, p = {p_value:.3f}", (best, gain.max()), xytext=(9 if right else -30, 2 if right else -7),
                     textcoords="offset points", ha="left" if right else "right", va="center", fontsize=9,
                     color=INK["primary"], zorder=6,
                     bbox=dict(boxstyle="round,pad=0.15", fc=INK["surface"], ec="none", alpha=0.85))
    low.text(16, min(v[2] for v in scans.values()) * 1.03, "thin lines: 95% of rotated series stay below",
             va="bottom", ha="left", fontsize=8.5, color=INK["secondary"])
    low.set_ylabel("Variance explained\nby a split here")
    low.set_ylim(0, peak * 1.3)
    low.set_xlabel(f"Anakin's lines in Episode II, in story order  ·  the best split is found without being "
                   f"told the scene, {abs(blind - k)} lines from the massacre")
    low.set_xlim(-2, len(x) + 1)
    _title(top, "In Episode II, the strongest break in Anakin's language lands on the Tusken massacre",
           "Every line he speaks in Attack of the Clones, scored by an emotion model that knows "
           "nothing of the story", gap=0.04)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def dark_score_figure(frames: dict[str, pd.DataFrame], columns: dict[str, str], df: pd.DataFrame,
                      rolling_fn, events, path, window: int = 60) -> None:
    """The original question, asked directly: Anakin's dark-side score over the trilogy."""
    use_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    _episode_bands(ax, df)
    colour = CHARACTER_COLORS["ANAKIN"]
    ends = []
    for (name, frame), style in zip(frames.items(), ("-", (0, (4, 2)))):
        smooth = rolling_fn(frame[columns[name]].to_numpy(), window=window, n_boot=1)["mean"]
        ax.plot(frame["position"], smooth, color=colour, lw=2.2, ls=style, zorder=3)
        valid = smooth.notna().to_numpy()
        ends.append((name, float(smooth[valid].iloc[-1]), float(frame["position"][valid].iloc[-1]), colour))
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for name, y, x, c in _spread_labels(ends, min_gap=span * 0.07):
        ax.annotate(name, (x, y), xytext=(9, 0), textcoords="offset points", va="center",
                    fontsize=9.5, color=INK["primary"])
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + (high - low) * 0.1)
    _mark_events(ax, events)
    ax.set_xlim(-0.005, 1.12)
    ax.set_xlabel("Narrative time  ·  position of the line across the three screenplays")
    ax.set_ylabel("P(dark side),  rolling mean over 60 lines")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Line2D([], [], color=colour, lw=2.2, ls=st, label=n)
                       for n, st in zip(frames, ("-", (0, (4, 2))))],
              loc="upper left", ncol=2, fontsize=9, bbox_to_anchor=(0, -0.14))
    _title(ax, "Anakin's dark-side score rises around the massacre, falls back, then climbs in Episode III",
           "Classifiers trained on other characters only · probabilities are not calibrated, "
           "so read the shape, not the level")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
