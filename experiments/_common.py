"""Small helpers shared by experiment scripts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:      # allow `python experiments/xyz.py` as well as `python -m experiments.xyz`
    sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"
SAMPLE_DIR = ROOT / "outputs" / "samples"
for _d in (FIG_DIR, LOG_DIR, SAMPLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
