"""Per-line emotion scores from a model that never saw this dataset.

This replaces the original project's ``TextBlob`` polarity, for two reasons.

TextBlob returns a single number on a positive/negative axis. Fear, anger and grief
all read as "negative", and they are the three the story actually distinguishes.

More importantly, this model is trained elsewhere, on unrelated corpora, with no
knowledge of Star Wars, of the faction labels, or of who is speaking. Whatever
trajectory it finds cannot be an artefact of the label design in this repository -
which is exactly the weakness of the faction classifier it is here to complement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def score_emotions(texts: list[str], batch_size: int = 64, verbose: bool = True) -> pd.DataFrame:
    """Return a (len(texts), 7) frame of emotion probabilities, one row per line."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()

    order = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=128,
                            return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
            if verbose and start % (batch_size * 10) == 0:
                print(f"  {start + len(batch)}/{len(texts)}", flush=True)

    return pd.DataFrame(np.vstack(out), columns=order)[EMOTIONS]


def dark_affect(emotions: pd.DataFrame) -> pd.Series:
    """Anger + fear: the two emotions the story names as the road to the dark side.

    Chosen from Yoda's stated causal chain before any line was scored, rather than by
    picking whichever emotions happened to trend. This was not a formal
    pre-registration.
    """
    return emotions["anger"] + emotions["fear"]
