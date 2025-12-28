from os.path import isdir
from pathlib import Path

# Points to the project root (where history.csv lives)
BASE_DIR = Path(__file__).resolve().parents[2]

# Specific file paths
HISTORY_PATH = BASE_DIR / "history.csv"
HISTORY_PLOTS_DIR = BASE_DIR / "history_plots"
kaggle_data_dir = '/kaggle/input/face-recognition-data/data'
local_data_dir = BASE_DIR / "data"

DATA_DIR = kaggle_data_dir if isdir(kaggle_data_dir) else local_data_dir
