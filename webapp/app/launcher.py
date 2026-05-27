"""Desktop launcher for the Streamlit app.

This launcher is compatible with PyInstaller. It does not spawn sys.executable,
because in a packaged app sys.executable points back to the app itself and would
create a recursive launch loop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from config import STREAMLIT_PORT


def resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parent / relative_path


def main() -> int:
    app_path = resource_path("app.py")

    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.headless=false",
        "--server.address=127.0.0.1",
        f"--server.port={STREAMLIT_PORT}",
        "--browser.gatherUsageStats=false",
    ]

    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
