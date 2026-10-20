"""Apply a staged update in-process, with rollback if any replacement fails."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable


ALLOWED = {
    "app.py", "account_client.py", "desktop_launcher.py", "desktop_entry.py",
    "update_client.py", "update_apply.py", "version.txt", "update_config.json",
    "assets/aee.ico", "account_defaults.json",
}
REQUIRED = ALLOWED - {
    "account_client.py", "assets/aee.ico", "account_defaults.json",
}


def apply_staged_update(stage: Path | str, app_dir: Path | str, expected_version: str,
                        progress: Callable | None = None) -> None:
    """Replace allowlisted application files and restore the old set on failure."""
    stage = Path(stage).resolve()
    app_dir = Path(app_dir).resolve()
    data_dir = Path(os.environ.get(
        "AEE_DATA_DIR", str(Path.home() / "AppData" / "Local" / "AI Evidence Extractor")
    )).resolve()
    updates_dir = (data_dir / "updates").resolve()
    if stage.parent != updates_dir or not stage.is_dir():
        raise RuntimeError("The prepared update folder is missing or invalid.")
    if (stage / "version.txt").read_text(encoding="utf-8").strip() != expected_version:
        raise RuntimeError("The prepared update version does not match.")
    app_exe = app_dir / "AI Evidence Extractor.exe"
    if not app_exe.is_file():
        raise RuntimeError("The installed desktop application could not be found.")

    files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    if not REQUIRED.issubset(files) or any(name not in ALLOWED for name in files):
        raise RuntimeError("The update package contains missing or unexpected files.")
    if any(p.is_symlink() for p in stage.rglob("*")):
        raise RuntimeError("The update package contains an unsafe link.")
    # Commit the version marker last so it never advertises a partial update.
    files = [name for name in files if name != "version.txt"] + ["version.txt"]
    rollback = stage / ".rollback"
    changed: list[tuple[Path, Path | None]] = []
    total = len(files)
    try:
        for index, name in enumerate(files, start=1):
            source = stage.joinpath(*Path(name).parts)
            target = app_dir.joinpath(*Path(name).parts)
            if target.resolve().parent != (app_dir / Path(name).parent).resolve():
                raise RuntimeError("An update destination path is unsafe.")
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if target.exists():
                backup = rollback.joinpath(*Path(name).parts)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            temporary = target.with_name(target.name + ".aee-update-new")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
            changed.append((target, backup))
            if progress:
                progress(index / total, desc=f"Installing update: {name}")
    except Exception:
        for target, backup in reversed(changed):
            try:
                if backup and backup.is_file():
                    os.replace(backup, target)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    # The new version is active on next launch. Remove only updater-owned staging data.
    shutil.rmtree(stage, ignore_errors=True)
    package = updates_dir / f"AI_Evidence_Extractor_Update_{expected_version}.zip"
    package.unlink(missing_ok=True)
    if progress:
        progress(1, desc="Update installed. Restart the app to load it.")
