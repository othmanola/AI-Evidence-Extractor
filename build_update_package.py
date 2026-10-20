"""Build the source-only archive used by the in-app updater."""

from __future__ import annotations

import re
import os
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
FILES = (
    "app.py",
    "account_client.py",
    "desktop_launcher.py",
    "desktop_entry.py",
    "update_client.py",
    "update_apply.py",
    "version.txt",
    "update_config.json",
    "assets/aee.ico",
)
DEFAULTS_PATH_ENV = "AEE_ACCOUNT_DEFAULTS_PATH"


def main() -> None:
    version = (ROOT / "version.txt").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit("version.txt must contain a three-part version number")
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing update files: " + ", ".join(missing))
    defaults_path = os.environ.get(DEFAULTS_PATH_ENV)
    if defaults_path and not Path(defaults_path).is_file():
        raise SystemExit(f"{DEFAULTS_PATH_ENV} does not exist")
    DIST.mkdir(exist_ok=True)
    archive = DIST / "AI_Evidence_Extractor_Update.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=8) as package:
        for name in FILES:
            package.write(ROOT / name, arcname=name)
        if defaults_path:
            package.write(defaults_path, arcname="account_defaults.json")
        else:
            package.writestr(
                "account_defaults.json",
                '{"project_url":"","anon_key":""}',
            )
    print(f"Update package created for v{version}: {archive}")


if __name__ == "__main__":
    main()
