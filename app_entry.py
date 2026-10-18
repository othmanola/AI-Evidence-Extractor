"""Direct executable entry point for the Windows desktop application."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    app_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    data_dir = Path(os.environ.get(
        "AEE_DATA_DIR",
        str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AI Evidence Extractor"),
    )).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AEE_APP_DIR"] = str(app_dir)
    os.environ["AEE_DATA_DIR"] = str(data_dir)
    try:
        if getattr(sys, "frozen", False):
            from desktop_entry import main as launch_source_app
            return launch_source_app()
        from desktop_launcher import main as launch_desktop
        launch_desktop()
        return 0
    except Exception:
        (data_dir / "desktop.log").open("a", encoding="utf-8").write(
            "".join(traceback.format_exc())
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
