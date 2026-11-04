"""Build the source-only archive used by the in-app updater."""

from __future__ import annotations

import json
import os
import re
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


def _update_config_for_package() -> str:
    config_path = ROOT / "update_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise SystemExit("update_config.json must contain a JSON object")
    defaults_path = Path(os.environ.get(
        "AEE_ACCOUNT_DEFAULTS_PATH",
        str(ROOT / "account_defaults.json"),
    ))
    if defaults_path.is_file():
        defaults = json.loads(defaults_path.read_text(encoding="utf-8-sig"))
        if not isinstance(defaults, dict):
            raise SystemExit("The account defaults file must contain a JSON object")
        project_url = defaults.get("project_url")
        anon_key = defaults.get("anon_key")
        if not isinstance(project_url, str) or not isinstance(anon_key, str):
            raise SystemExit("The account defaults file is invalid")
        config["default_project_url"] = project_url
        config["default_anon_key"] = anon_key
    return json.dumps(config, indent=2) + "\n"


def main() -> None:
    version = (ROOT / "version.txt").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit("version.txt must contain a three-part version number")
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing update files: " + ", ".join(missing))
    DIST.mkdir(exist_ok=True)
    archive = DIST / "AI_Evidence_Extractor_Update.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=8) as package:
        for name in FILES:
            if name == "update_config.json":
                package.writestr(name, _update_config_for_package())
            else:
                package.write(ROOT / name, arcname=name)
    print(f"Update package created for v{version}: {archive}")


if __name__ == "__main__":
    main()
