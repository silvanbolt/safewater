$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
# When the real YOLO model is ready, uncomment:
# pip install -r requirements-yolo.txt

pyinstaller --clean --noconfirm packaging\pyinstaller\water_source_classifier.spec

# Requires Inno Setup installed and ISCC.exe available on PATH.
iscc packaging\windows\installer.iss

Write-Host "Windows installer created in packaging\windows\dist-installer"
