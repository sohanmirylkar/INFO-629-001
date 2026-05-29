from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
RAW_DATA=BASE_DIR/'data'/'raw'
PROCESSED_DATA=BASE_DIR/'data'/'processed'
MODELS_DIR=BASE_DIR/'models'
REPORTS_DIR=BASE_DIR/'reports'
THRESHOLD_PERCENTILE=95
RANDOM_STATE=42
