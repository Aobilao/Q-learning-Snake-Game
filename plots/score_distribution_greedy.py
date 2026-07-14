import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import load_log

LOG_PATH = ROOT / "values" / "run_log.pkl"
FIG_PATH = ROOT / "figures" / "score_distribution_greedy.png"

if __name__ == "__main__":
    log = load_log(LOG_PATH)
    scores = np.asarray(log["score"])

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(scores.min(), scores.max() + 2)
    ax.hist(scores, bins=bins, edgecolor="white", linewidth=0.4)
    ax.axvline(scores.mean(), color="red", lw=1.5, label=f"mean = {scores.mean():.1f}")

    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300)
