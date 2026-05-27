import pandas as pd
import requests
from pathlib import Path
from urllib.parse import urlparse
import mimetypes
import time

# =========================
# Configuration
# =========================
CSV_PATH = "input/pre-processing/Rest_try_again.csv"

OUTPUT_SUCCESS_CSV = "output/pre-processing/raw_data_plus_paths.csv"
OUTPUT_FAILED_CSV = "output/pre-processing/failed_downloads_.csv"

IMAGE_DIR = Path("output/pre-processing/images")

ROW_ID_COL = "row_id"
IMAGE_URL_COL = "#photo_lnk"  

REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 0.05

# =========================
# Setup
# =========================
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# load data
df = pd.read_csv(CSV_PATH)


df = df.dropna(subset=[IMAGE_URL_COL]).copy()
df[IMAGE_URL_COL] = df[IMAGE_URL_COL].astype(str).str.strip()
df = df[df[IMAGE_URL_COL] != ""].copy()

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

successful_rows = []
failed_rows = []
image_paths = []

# =========================
# Helper functions
# =========================

def sanitize_row_id(x):
    return str(x).replace(",", "").replace(" ", "")

def guess_extension(response, url):
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    ext = mimetypes.guess_extension(content_type)

    if ext:
        return ext

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix

    if suffix:
        return suffix

    return ".jpg"


# =========================
# Download loop
# =========================

total = len(df)

for i, (_, row) in enumerate(df.iterrows(), start=1):

    row_id = sanitize_row_id(row[ROW_ID_COL])
    url = row[IMAGE_URL_COL]

    print(f"[{i}/{total}] Downloading {row_id}")

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "image" not in content_type:
            raise ValueError(f"Not an image ({content_type})")

        ext = guess_extension(response, url)
        image_path = IMAGE_DIR / f"{row_id}{ext}"

        with open(image_path, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)

        successful_rows.append(row)
        image_paths.append(str(image_path))

    except Exception as e:
        row_copy = row.copy()
        row_copy["download_error"] = str(e)
        failed_rows.append(row_copy)

        print(f"   Failed: {e}")

    time.sleep(SLEEP_BETWEEN_REQUESTS)

# =========================
# Build success dataframe
# =========================

df_success = pd.DataFrame(successful_rows).reset_index(drop=True)
df_success["image_path"] = image_paths

# =========================
# Build failed dataframe
# =========================

df_failed = pd.DataFrame(failed_rows)

# =========================
# Save outputs
# =========================

df_success.to_csv(OUTPUT_SUCCESS_CSV, index=False)
df_failed.to_csv(OUTPUT_FAILED_CSV, index=False)

print("\nDownload summary")
print("----------------")
print("Successful:", len(df_success)) #7229 + 3
print("Failed:", len(df_failed)) #5301 -3 
print("Images saved to:", IMAGE_DIR.resolve())
print("Clean dataset saved to:", OUTPUT_SUCCESS_CSV)
print("Failed rows saved to:", OUTPUT_FAILED_CSV)
