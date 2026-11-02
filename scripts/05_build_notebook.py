"""Generate notebooks/anakin_fall.ipynb from the computed artefacts.

The notebook reads the files produced by scripts 01-04 rather than retraining anything,
so it executes in seconds and shows exactly the numbers in the report. It never prints
raw dialogue: the screenplays are third-party copyrighted text.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nbformat as nbf

from anakin_nlp import config

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell


def build() -> nbf.NotebookNode:
    cells = [
        MD("""# Did Anakin Skywalker's language actually change?

Measuring a character's turn to the dark side from screenplay dialogue alone, rebuilt
from a master's project. This notebook loads the results computed by the pipeline
(`make dataset models emotions arc`) and retrains nothing. The full write-up is in
`docs/project_report.pdf`."""),
        CODE("""import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, str(Path.cwd().parent / "src"))
from anakin_nlp import config
from anakin_nlp.characters import UNALIGNED

dialogue = pd.read_parquet(config.DIALOGUE_PARQUET)
metrics = json.loads((config.REPORTS / "faction_metrics.json").read_text())
results = json.loads((config.REPORTS / "arc_results.json").read_text())
massacre = results["tusken_massacre"]

def figure(name):
    display(Image(filename=str(config.FIGURES / name), width=900))

print(f"{len(dialogue):,} lines of dialogue, {dialogue.character.nunique()} characters")"""),
        MD("""## 1. The dataset

Three screenplays in three layouts go through one parser. Names are corrected from an
explicit table, and punctuation is kept."""),
        CODE("""summary = (dialogue.groupby("episode")
           .agg(lines=("text", "size"), characters=("character", "nunique"),
                mean_words=("n_words", "mean")).round(1))
summary.index = [config.EPISODES[i] for i in summary.index]
display(summary)
display(dialogue.groupby("role").agg(lines=("text", "size"), characters=("character", "nunique")))
print(f"{len(UNALIGNED)} characters deliberately left unlabelled:")
for name, reason in sorted(UNALIGNED.items()):
    print(f"  {name:<14} {reason}")"""),
        MD("""## 2. The validation problem

The label belongs to a character, not to a sentence. A random split puts a character's
lines in both training and test, so a model can score by recognising the speaker.
Holding each character out whole gives the honest figure."""),
        CODE("""figure("01_validation_gap.png")
compare = results["model_comparison"]
print("TF-IDF ROC-AUC by seed, split by character:",
      {k: round(v, 3) for k, v in compare["tfidf_grouped_auc_by_seed"].items()})
print("Same folds:", {k: round(v, 3) for k, v in compare["same_folds"].items()})
print("Same folds, without Palpatine:",
      {k: round(v, 3) for k, v in compare["same_folds_without_palpatine"].items()})"""),
        MD("""TF-IDF falls from 0.80 to 0.58 on average. DistilBERT shows no demonstrated gain: on
identical folds the two models score about the same, and without Palpatine TF-IDF scores
higher."""),
        CODE("""figure("04_top_terms.png")"""),
        MD("""The weights mix factional vocabulary (`lord`, `senator`) with `mesa`, which is almost
entirely Jar Jar's, and `annie`, the nickname other characters use for
Anakin. Neither says anything about the rhetoric of a camp."""),
        MD("""## 3. The Tusken massacre

Anakin says nothing during the massacre. His first words afterwards are in the homestead
garage (Episode II, scene 125). The lower panel below is computed without being told where
that scene is."""),
        CODE("""figure("00_tusken_break.png")
rows = []
for col, m in massacre["measures"].items():
    rows.append({"measure": m["label"], "before": m["episode_2"]["before"],
                 "after": m["episode_2"]["after"], "step p": m["episode_2"]["p_value"],
                 "blind split: lines from massacre": m["blind_break"]["lines_from_event"],
                 "blind split: scene": m["blind_break"]["scene"],
                 "blind split p": m["blind_break"]["p_value"]})
display(pd.DataFrame(rows).set_index("measure").round(3))
print("Location when scenes are resampled:", massacre["location_uncertainty"])"""),
        MD("""All four measures rise at the massacre, and all four blind searches land within 2 lines
of it. Only anger clears the bar in the blind search, and the location is a best estimate:
when scenes are resampled the break lands on scene 125 in 28% of runs."""),
        CODE("""for name, entry in massacre["controls"].items():
    for col, res in entry.items():
        v = res["vs_anakin"]
        print(f"{name:<8} {col:<12} jump {res['step']['difference']:+.3f} | Anakin's extra jump "
              f"{v['difference']:+.3f}, 95% CI [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")"""),
        MD("""Padmé rises at the same point too. Anakin's jump is larger, but the difference from
Padmé's is not statistically clear."""),
        MD("""## 4. Across the whole trilogy"""),
        CODE("""figure("06_dark_score.png")
display(pd.DataFrame({m["label"]: m["levels"] for m in massacre["measures"].values()}).T.round(3))
rows = []
for col in ("dark_affect", "p_dark_tfidf", "p_dark_bert"):
    for size, cp in results[col]["best_break_by_min_size"].items():
        rows.append({"measure": results[col]["measure"], "min lines per side": int(size),
                     "episode": cp["episode"], "scene": cp["scene"],
                     "position": round(cp["position"], 3), "before": round(cp["before"], 3),
                     "after": round(cp["after"], 3), "p": round(cp["p_value"], 4)})
display(pd.DataFrame(rows).set_index(["measure", "min lines per side"]))"""),
        MD("""The massacre break does not last: both measures return to about their earlier level in
the first half of Episode III. Across the trilogy, both classifiers place the largest break
at the same line in Episode III, just before the Chancellor's office scenes. The single best
break in anger and fear depends on the minimum size allowed."""),
        CODE("""figure("02_emotional_arc.png")
figure("05_by_episode.png")
emo = results["dark_affect"]
c = emo["pooled_light_control"]["contrast"]
print(f"vs pooled light side: {c['difference']:+.3f}  95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")
for name, c in emo["controls"].items():
    print(f"vs {name:<10} {c['difference']:+.3f}  95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")"""),
        MD("""Anger and fear rise overall, but no comparison with a control excludes zero, so this
overall rise is not clearly his alone."""),
        MD("""## 5. Which emotions move"""),
        CODE("""figure("03_emotion_panels.png")
trends = pd.DataFrame(results["emotions"]["trends"]).T[["rho", "p_value", "p_holm"]].astype(float)
display(trends.sort_values("rho").round(3))
print(results["emotions"]["question_marks"])"""),
        MD("""Surprise and joy fall more clearly than anger rises. Most of that fall happens between
Episodes I and II, where the script source changes and the character goes from child to young
man."""),
        MD("""## Limitations

- Seven dark-side characters, and Palpatine has 52% of their lines.
- Many tests on one small dataset; only the emotion table is adjusted for multiple comparisons.
- The three scripts differ in source and stage, so comparisons within an episode are more
  reliable than comparisons between episodes.
- Story time is line rank, not screen time.
- The models are out of domain and uncalibrated, so trends and contrasts mean more than
  absolute levels."""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


if __name__ == "__main__":
    out = config.ROOT / "notebooks" / "anakin_fall.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build(), out)
    print(f"-> {out.relative_to(config.ROOT)}")
