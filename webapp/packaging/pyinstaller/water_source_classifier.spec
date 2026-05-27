# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None
ROOT = Path.cwd()


def safe_copy_metadata(package_name):
    try:
        return copy_metadata(package_name)
    except Exception:
        return []


streamlit_datas = collect_data_files("streamlit")
for package in [
    "streamlit",
    "altair",
    "pyarrow",
    "pandas",
    "numpy",
    "pillow",
    "packaging",
    "tenacity",
    "tornado",
    "watchdog",
]:
    streamlit_datas += safe_copy_metadata(package)

streamlit_hiddenimports = collect_submodules("streamlit")

app_datas = [
    (str(ROOT / "app" / "app.py"), "."),
    (str(ROOT / "app" / "config.py"), "."),
    (str(ROOT / "app" / "inference.py"), "."),
]

model_dir = ROOT / "app" / "model"
if model_dir.exists():
    app_datas.append((str(model_dir), "model"))

# Add Streamlit secrets if you later create .streamlit/secrets.toml.
secrets_dir = ROOT / ".streamlit"
if secrets_dir.exists():
    app_datas.append((str(secrets_dir), ".streamlit"))

# If you package the real YOLO model, install requirements-yolo.txt in the build
# environment first. PyInstaller may require extra hidden imports depending on
# the exact Ultralytics/PyTorch version.
# ultralytics_hiddenimports = collect_submodules("ultralytics")

hiddenimports = streamlit_hiddenimports


a = Analysis(
    [str(ROOT / "app" / "launcher.py")],
    pathex=[str(ROOT), str(ROOT / "app")],
    binaries=[],
    datas=streamlit_datas + app_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Water Source Classifier",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Water Source Classifier",
)

app = BUNDLE(
    coll,
    name="Water Source Classifier.app",
    icon=None,
    bundle_identifier="org.example.watersourceclassifier",
)
