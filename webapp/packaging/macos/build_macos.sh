#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
# When the real YOLO model is ready, uncomment:
# pip install -r requirements-yolo.txt

pyinstaller --clean --noconfirm packaging/pyinstaller/water_source_classifier.spec

# Option A: Create a simple ZIP if you do not have create-dmg installed.
mkdir -p packaging/macos/dist-installer
 ditto -c -k --keepParent "dist/Water Source Classifier.app" "packaging/macos/dist-installer/WaterSourceClassifier-macOS.zip"

# Option B: Create a DMG if create-dmg is installed.
if command -v create-dmg >/dev/null 2>&1; then
  create-dmg \
    --volname "Water Source Classifier" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --app-drop-link 430 185 \
    "packaging/macos/dist-installer/WaterSourceClassifier.dmg" \
    "dist/Water Source Classifier.app" || true
fi

echo "macOS package created in packaging/macos/dist-installer"
