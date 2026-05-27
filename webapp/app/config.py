from pathlib import Path

APP_NAME = "Water Source Classifier"
APP_VERSION = "0.2.0"

MODEL_BACKEND = "yolo"
MODEL_VERSION = "yolo11m-cls-water-source-v1"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best-finetune.pt"

# Number of source-type predictions to show in the interface.
TOP_K = 3

# Local Streamlit port used by the desktop launcher.
STREAMLIT_PORT = 8501

# Mapping from source type to binary class.
# IMPORTANT: update these keys so they match the exact class names used by the YOLO model.
# The app normalizes class names by lowercasing and replacing spaces, hyphens, and slashes with underscores.
SOURCE_TYPE_TO_BINARY_CLASS = {
    # Improved water sources
    "piped_water": "improved",
    "borehole_tubewell": "improved",
    "protected_well": "improved",
    "protected_spring": "improved",
    "rainwater_harvesting": "improved",
    "sand_or_sub_surface_dam": "improved",
    
    # Non-improved water sources
    "unprotected_well": "non-improved",
    "delivered_water": "non-improved",
    "surface_water": "non-improved"
}
