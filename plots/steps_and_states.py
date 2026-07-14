import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import load_log

LOG_PATH = ROOT / "values" / "augmented_training_log.pkl"
FIG_PATH = ROOT / "figures" / "steps_and_states.png"
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

    ep, steps = rolling_mean(log["steps"], WINDOW)
    ax1.plot(ep, steps)
    ax1.set_ylabel(f"Steps (mean/{WINDOW})")
    ax1.set_title("Steps and states explored")
    ax1.grid(True, alpha=0.3)

    states = np.asarray(log["states_visited"])
    ax2.plot(states)
    ax2.set_ylabel("States explored")
    ax2.set_xlabel("Episode")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300)
