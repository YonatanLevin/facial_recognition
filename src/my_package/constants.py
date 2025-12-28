from pathlib import Path

# Points to the project root (where history.csv lives)
BASE_DIR = Path(__file__).resolve().parents[2]

# Specific file paths
HISTORY_PATH = BASE_DIR / "history.csv"
HISTORY_PLOTS_DIR = BASE_DIR / "history_plots"
DATA_DIR = BASE_DIR / "data"
