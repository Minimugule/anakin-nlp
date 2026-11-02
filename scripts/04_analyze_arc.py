"""Measure Anakin's arc, test the Tusken massacre as a break, draw the figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from anakin_nlp import arc, config, models, viz
from anakin_nlp.emotions import EMOTIONS
from anakin_nlp.events import TUSKEN_MASSACRE as EVENT, split_index
from anakin_nlp.parse import scene_headings

SUBJECT = "ANAKIN"
CONTROLS = ["OBI-WAN", "PADME", "PALPATINE"]
WINDOW = 60
MEASURES = {"dark_affect": "P(anger) + P(fear)", "anger": "P(anger)",
            "p_dark_bert": "P(dark side), DistilBERT", "p_dark_tfidf": "P(dark side), TF-IDF"}
GARAGE, GUNSHIP = 125, 164      # the two angriest scenes after the massacre
EP2_MIN_SIZE = 15


def lines_for(df: pd.DataFrame, name: str, col: str, episode: int | None = None) -> pd.DataFrame:
    keep = (df["character"] == name) & df[col].notna()
    if episode is not None:
        keep &= df["episode"] == episode
    return df[keep].sort_values("global_order").reset_index(drop=True)


def series_for(df: pd.DataFrame, name: str, col: str, window: int) -> pd.DataFrame:
    sub = lines_for(df, name, col)
    return arc.rolling(sub[col].to_numpy(), window=window).assign(position=sub["position"].to_numpy())


def located(frame: pd.DataFrame, i: int, headings: dict) -> dict:
    row = frame.iloc[i]
    return {"episode": int(row["episode"]), "scene": int(row["scene"]),
            "position": float(row["position"]),
            "heading": headings[int(row["episode"])].get(int(row["scene"]), "")}


def analyse_arc(df: pd.DataFrame, col: str, label: str, headings: dict) -> dict:
    """Trend, best single break at several minimum sizes, and contrasts with controls."""
    subject = lines_for(df, SUBJECT, col)
    values = subject[col].to_numpy()
    out: dict = {"measure": label, "trend": arc.trend(subject["position"].to_numpy(), values)}

    # Reported at several sizes because the answer moves, which is a result in itself.
    out["best_break_by_min_size"] = {}
    for size in (15, 30, 50):
        cp = arc.change_point(values, size)
        out["best_break_by_min_size"][str(size)] = cp | located(subject, cp["index"], headings)

    pooled = df[(df["faction"] == "light") & df[col].notna()].sort_values("global_order")
    out["pooled_light_control"] = {
        "n_lines": int(len(pooled)), "n_characters": int(pooled["character"].nunique()),
        "contrast": arc.slope_contrast(subject, pooled, col),
    }
    out["controls"] = {name: arc.slope_contrast(subject, lines_for(df, name, col), col)
                       for name in CONTROLS if len(lines_for(df, name, col)) >= 40}
    return out


def analyse_massacre(df: pd.DataFrame, headings: dict) -> dict:
    """Does the dialogue break where the story says Anakin does?"""
    out: dict = {"event": {"episode": EVENT.episode, "scene": EVENT.scene,
                           "heading": headings[EVENT.episode][EVENT.scene]},
                 "measures": {}}

    for col, label in MEASURES.items():
        full = lines_for(df, SUBJECT, col)
        ep2 = lines_for(df, SUBJECT, col, EVENT.episode)
        ep3 = lines_for(df, SUBJECT, col, 3)
        k_full, k = split_index(full, EVENT), split_index(ep2, EVENT)
        values = ep2[col].to_numpy()

        blind = arc.change_point(values, EP2_MIN_SIZE)
        idx, gain = arc.scan(values, EP2_MIN_SIZE)
        by_size = {}
        for size in (5, 10, 15, 20, 30):
            i, g = arc.scan(values, size)
            best = int(i[g.argmax()])
            by_size[str(size)] = {"scene": int(ep2["scene"].iloc[best]), "lines_from_event": abs(best - k)}

        without = ep2[ep2["scene"] != GARAGE].reset_index(drop=True)
        cp_without = arc.change_point(without[col].to_numpy(), EP2_MIN_SIZE)
        before, after = ep2.iloc[:k], ep2.iloc[k:]

        out["measures"][col] = {
            "label": label,
            "whole_trilogy": arc.step_at(full[col].to_numpy(), k_full),
            "episode_2": arc.step_at(values, k),
            "blind_break": blind | {"scene": int(ep2["scene"].iloc[blind["index"]]),
                                    "lines_from_event": abs(blind["index"] - k),
                                    "event_gain": float(gain[idx == k][0]),
                                    "event_gain_rank": int((gain > gain[idx == k][0]).sum() + 1),
                                    "n_splits": int(len(idx))},
            "blind_break_by_min_size": by_size,
            "without_garage": {"blind_scene": int(without["scene"].iloc[cp_without["index"]]),
                               "after_mean": float(after[after["scene"] != GARAGE][col].mean())},
            "without_garage_and_gunship": {
                "after_mean": float(after[~after["scene"].isin([GARAGE, GUNSHIP])][col].mean()),
                "n_after": int((~after["scene"].isin([GARAGE, GUNSHIP])).sum())},
            "levels": {"before_massacre": float(before[col].mean()),
                       "after_massacre_episode_2": float(after[col].mean()),
                       "episode_3_first_half": float(ep3[col].iloc[:len(ep3) // 2].mean()),
                       "episode_3_second_half": float(ep3[col].iloc[len(ep3) // 2:].mean())},
        }

    # How firmly is the break located? Resample the scenes of Episode II.
    ep2 = lines_for(df, SUBJECT, "dark_affect", EVENT.episode)
    landed = arc.locate_break(ep2, "dark_affect", EP2_MIN_SIZE)
    out["location_uncertainty"] = {
        "n_resamples": int(len(landed)),
        "median_scene": float(landed.median()),
        "interval_90": [float(v) for v in np.percentile(landed, [5, 95])],
        "share_on_event_scene": float((landed == EVENT.scene).mean()),
        "share_at_or_after_camp": float((landed >= 117).mean()),
        "most_frequent": {int(k): int(v) for k, v in landed.value_counts().head(5).items()},
    }

    # Is it him, or the film? Same split, same episode, other characters.
    position = float(ep2["position"].iloc[split_index(ep2, EVENT)])
    out["position"] = position
    out["controls"] = {}
    for name in ("OBI-WAN", "PADME"):
        entry = {}
        for col in ("dark_affect", "anger"):
            control = lines_for(df, name, col, EVENT.episode)
            k = int((control["position"] >= position).to_numpy().argmax())
            entry[col] = {"step": arc.step_at(control[col].to_numpy(), k),
                          "vs_anakin": arc.difference_in_differences(
                              lines_for(df, SUBJECT, col, EVENT.episode), control, col, position)}
        out["controls"][name] = entry
    return out


def compare_models(df: pd.DataFrame) -> dict:
    """TF-IDF and DistilBERT on identical folds, since seeds move the score so much."""
    frame = models.modelling_frame(pd.read_parquet(config.DIALOGUE_PARQUET))
    seeds = {str(config.RANDOM_STATE + i): models.cross_validate(frame, seed=config.RANDOM_STATE + i)[0]["roc_auc"]
             for i in range(5)}
    _, oof = models.cross_validate(frame, seed=config.RANDOM_STATE)   # the DistilBERT folds
    oof = oof.merge(df[["global_order", "p_dark_bert"]], on="global_order")
    y = (oof["faction"] == "dark").astype(int)
    rest = oof["character"] != "PALPATINE"
    return {
        "tfidf_grouped_auc_by_seed": seeds,
        "same_folds": {"tfidf": roc_auc_score(y, oof["p_dark"]),
                       "distilbert": roc_auc_score(y, oof["p_dark_bert"])},
        "same_folds_without_palpatine": {"tfidf": roc_auc_score(y[rest], oof["p_dark"][rest]),
                                         "distilbert": roc_auc_score(y[rest], oof["p_dark_bert"][rest])},
    }


def analyse_emotions(df: pd.DataFrame) -> dict:
    subject = lines_for(df, SUBJECT, "joy")
    pos = subject["position"].to_numpy()
    trends = {e: arc.trend(pos, subject[e].to_numpy()) for e in EMOTIONS}
    adjusted = arc.holm({e: t["p_value"] for e, t in trends.items()})
    for e in EMOTIONS:
        trends[e]["p_holm"] = adjusted[e]
        trends[e]["within_episode"] = {
            str(ep): arc.trend(g["position"].to_numpy(), g[e].to_numpy())
            for ep, g in subject.groupby("episode")}

    # The emotion model reads a question mark as surprise, so check that reading.
    is_question = subject["text"].str.contains(r"\?")
    plain = subject[~is_question]
    return {
        "trends": trends,
        "question_marks": {
            "surprise_with": float(subject.loc[is_question, "surprise"].mean()),
            "surprise_without": float(plain["surprise"].mean()),
            "share_by_episode": {str(ep): float(g["text"].str.contains(r"\?").mean())
                                 for ep, g in subject.groupby("episode")},
            "surprise_trend_without_questions": arc.trend(plain["position"].to_numpy(),
                                                          plain["surprise"].to_numpy()),
        },
    }


def main() -> None:
    df = pd.read_parquet(config.SCORED_PARQUET)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((config.REPORTS / "faction_metrics.json").read_text())
    headings = {ep: scene_headings(config.script_dir() / name) for ep, name in config.SCRIPT_FILES.items()}

    results = {"model_comparison": compare_models(df),
               "tusken_massacre": analyse_massacre(df, headings),
               "emotions": analyse_emotions(df)}
    for col, label in [("dark_affect", MEASURES["dark_affect"]),
                       ("p_dark_tfidf", MEASURES["p_dark_tfidf"]),
                       ("p_dark_bert", MEASURES["p_dark_bert"])]:
        results[col] = analyse_arc(df, col, label, headings)

    table = (df[df.character.isin([SUBJECT] + CONTROLS)]
             .groupby(["character", "episode"])[EMOTIONS + ["dark_affect"]].mean().round(4))
    table.to_csv(config.REPORTS / "emotion_by_episode.csv")
    (config.REPORTS / "arc_results.json").write_text(json.dumps(results, indent=2, default=float))

    # --- figures -------------------------------------------------------------
    F = config.FIGURES
    seeds = list(results["model_comparison"]["tfidf_grouped_auc_by_seed"].values())
    viz.leakage_figure(metrics, F / "01_validation_gap.png",
                       f"TF-IDF: mean of five seeds, from {min(seeds):.2f} to {max(seeds):.2f}. "
                       "DistilBERT: a single run.")

    ep2 = lines_for(df, SUBJECT, "dark_affect", EVENT.episode)
    massacre = results["tusken_massacre"]
    primary = massacre["measures"]["dark_affect"]
    scans = {}
    for name, col in (("anger + fear", "dark_affect"), ("anger alone", "anger")):
        blind = massacre["measures"][col]["blind_break"]
        scans[name] = (*arc.scan(ep2[col].to_numpy(), EP2_MIN_SIZE), blind["threshold_95"], blind["p_value"])
    viz.massacre_figure(ep2, primary["episode_2"]["index"], primary["blind_break"]["index"], scans,
                        primary["episode_2"], F / "00_tusken_break.png")

    marker = [(massacre["position"], EVENT.label)]
    viz.arc_figure({name: series_for(df, name, "dark_affect", WINDOW) for name in [SUBJECT] + CONTROLS},
                   df, "P(anger) + P(fear),  rolling mean over 60 lines",
                   "Anger and fear across the trilogy",
                   "Emotion model trained outside this project, blind to the faction labels · "
                   "band = 90% interval on Anakin, descriptive",
                   F / "02_emotional_arc.png", events=marker)
    viz.emotion_panels(df, [SUBJECT, "OBI-WAN", "PADME"], ["anger", "fear", "sadness", "joy"],
                       WINDOW, F / "03_emotion_panels.png", arc.rolling, massacre["position"])
    viz.terms_figure(models.top_terms(models.fit_full(models.modelling_frame(df)), n=12), F / "04_top_terms.png")
    viz.episode_figure(df, [SUBJECT] + CONTROLS, "dark_affect", "P(anger) + P(fear),  episode mean",
                       "Episode means, with their uncertainty attached",
                       "95% intervals from resampling each character's scenes within each episode",
                       F / "05_by_episode.png")
    bert_break = results["p_dark_bert"]["best_break_by_min_size"]["50"]  # same line at every size
    viz.dark_score_figure(
        {"DistilBERT": lines_for(df, SUBJECT, "p_dark_bert"), "TF-IDF": lines_for(df, SUBJECT, "p_dark_tfidf")},
        {"DistilBERT": "p_dark_bert", "TF-IDF": "p_dark_tfidf"}, df, arc.rolling,
        marker + [(bert_break["position"], "Largest break in both scores")], F / "06_dark_score.png")

    print(json.dumps({k: results[k] for k in ("model_comparison",)}, indent=1, default=float))
    print(json.dumps(massacre, indent=1, default=float))
    print(json.dumps(results["emotions"]["question_marks"], indent=1, default=float))
    for e, t in sorted(results["emotions"]["trends"].items(), key=lambda kv: -abs(kv[1]["rho"])):
        print(f"{e:<9} rho {t['rho']:+.3f} p {t['p_value']:.4f} holm {t['p_holm']:.4f} within "
              f"{ {k: (round(v['rho'], 3), round(v['p_value'], 3)) for k, v in t['within_episode'].items()} }")
    for col in ("dark_affect", "p_dark_tfidf", "p_dark_bert"):
        r = results[col]
        print(f"\n{col} trend {r['trend']}")
        for size, cp in r["best_break_by_min_size"].items():
            print(f"  min {size}: pos {cp['position']:.3f} ep{cp['episode']} sc{cp['scene']} "
                  f"{cp['before']:.3f}->{cp['after']:.3f} p={cp['p_value']:.4f} | {cp['heading'][:48]}")
        print("  pooled", {k: round(v, 3) if isinstance(v, float) else v for k, v in r["pooled_light_control"]["contrast"].items()})
        for name, c in r["controls"].items():
            print(f"  vs {name}: {c['difference']:+.3f} [{c['ci_low']:+.3f},{c['ci_high']:+.3f}]")


if __name__ == "__main__":
    main()
