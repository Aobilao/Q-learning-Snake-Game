import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import load_log

LOG_PATH = ROOT / "values" / "training_log.pkl"
FIG_PATH = ROOT / "figures" / "learning_curve.png"
WINDOW = 5000


def rolling_mean(x, w):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= w:
        return np.arange(n), x
    cs = np.concatenate(([0.0], np.cumsum(x)))
    out = (cs[w:] - cs[:-w]) / w
    idx = np.arange(w - 1, n) - (w - 1) // 2
    return idx, out


if __name__ == "__main__":
    log = load_log(LOG_PATH)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    ep, score = rolling_mean(log["score"], WINDOW)
    ax1.plot(ep, score)
    ax1.set_ylabel(f"Score (mean/{WINDOW})")
    ax1.set_title("Learning curve")
    ax1.grid(True, alpha=0.3)

    causes = np.array(["timeout" if c is None else c for c in log["death_cause"]])
    labels = ["wall", "body", "timeout"]
    fracs = [rolling_mean((causes == lab).astype(float), WINDOW)[1] for lab in labels]
    ax2.stackplot(ep, *fracs, labels=labels, alpha=0.85)
    ax2.set_ylabel(f"Death cause (frac/{WINDOW})")
    ax2.set_xlabel("Episode")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
