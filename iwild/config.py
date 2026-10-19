import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = Path(os.getenv("IWILD_DATA_DIR", str(PROJ_ROOT / "data")))
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
ZEROSHOT_DIR = MODELS_DIR / "zeroshot"
AUTOFT_DIR = MODELS_DIR / "autoft"

# Keep CLIP/open_clip weight downloads on the project drive (the workstation C: is full)
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "hub"))

# Fine-tuning CSVs (FLYP/AutoFT format) live under interim/, following the official auto-ft layout
CSV_DIR = INTERIM_DATA_DIR / "csv"

# iWildCam y -> species name mapping shipped with the package (from the official auto-ft repo)
IWILDCAM_LABELS_PATH = (
    Path(__file__).resolve().parent / "modeling" / "iwildcam_metadata" / "labels.csv"
)

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
