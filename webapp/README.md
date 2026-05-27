# Water Source Classifier

A simple English-language desktop interface for classifying photos of water sources in Ethiopia.

The app predicts:

1. The top 3 likely water-source types.
2. A final binary label: **Improved** or **Non-improved**.

The binary label is derived from the top source-type prediction using the mapping in `app/config.py`.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/app.py
```

## Build using GitHub Actions

A workflow is included in:

```text
.github/workflows/build-all.yml
```

Run it from:

```text
GitHub → Actions → Build Desktop Apps → Run workflow
```

It builds macOS and Windows artifacts online using Python 3.11.

## Build Windows installer locally

Run this on Windows, not macOS/Linux:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\packaging\windows\build_windows.ps1
```

Requirements:

- Python 3.11 or 3.12
- Inno Setup installed
- `ISCC.exe` available on PATH

Output:

```text
packaging/windows/dist-installer/WaterSourceClassifierSetup.exe
```

## Build macOS package locally

Run this on macOS, not Windows/Linux:

```bash
./packaging/macos/build_macos.sh
```

Output:

```text
packaging/macos/dist-installer/WaterSourceClassifier-macOS.zip
packaging/macos/dist-installer/WaterSourceClassifier.dmg  # if create-dmg is installed
```

For public distribution on macOS, sign and notarize the app with an Apple Developer ID.

```
