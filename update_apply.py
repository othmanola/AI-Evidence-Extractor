"""Apply a verified staged app package after the desktop app has exited."""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


UPDATE_FILES = {
    "app.py", "desktop_launcher.py", "desktop_entry.py", "update_client.py",
    "update_apply.py", "version.txt", "update_config.json", "assets/aee.ico",
}


def _log_file() -> Path:
    data_dir = Path(os.environ.get(
        "AEE_DATA_DIR", str(Path.home() / "AppData" / "Local" / "AI Evidence Extractor")
    ))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "update_apply.log"


def _wait_for_process(pid: int) -> None:
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.WaitForSingleObject.restype = ctypes.c_ulong
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_bool
        synchronize = 0x00100000
        handle = kernel.OpenProcess(synchronize, False, pid)
        if handle:
            try:
                kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
            finally:
                kernel.CloseHandle(handle)
        return
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.5)


def _safe_stage_files(stage: Path) -> list[str]:
    files: list[str] = []
    for path in stage.rglob("*"):
        relative = path.relative_to(stage).as_posix()
        if path.is_symlink():
            raise ValueError("Staged updates cannot contain symbolic links")
        if path.is_dir():
            if relative != "assets":
                raise ValueError(f"Unexpected update folder: {relative}")
            continue
        if relative not in UPDATE_FILES:
            raise ValueError(f"Unexpected update file: {relative}")
        files.append(relative)
    required = UPDATE_FILES - {"assets/aee.ico"}
    if not required.issubset(files):
        raise ValueError("The staged update is missing required app files")
    return sorted(files)


def _apply(stage: Path, app_dir: Path, expected_version: str) -> None:
    data_dir = Path(os.environ.get(
        "AEE_DATA_DIR", str(Path.home() / "AppData" / "Local" / "AI Evidence Extractor")
    )).resolve()
    expected_parent = (data_dir / "updates").resolve()
    if stage.resolve().parent != expected_parent:
        raise ValueError("The staged update is outside the app updates folder")
    if not (app_dir / "AI Evidence Extractor.exe").is_file():
        raise ValueError("The app executable was not found")
    staged_version = (stage / "version.txt").read_text(encoding="utf-8").strip()
    if staged_version != expected_version:
        raise ValueError("The staged version did not match")

    files = _safe_stage_files(stage)
    backup_dir = stage / ".rollback"
    applied: list[tuple[Path, Path | None]] = []
    try:
        for relative in files:
            source = stage / Path(*relative.split("/"))
            target = app_dir / Path(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            backup: Path | None = None
            if target.exists():
                backup = backup_dir / Path(*relative.split("/"))
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            replacement = target.with_name(target.name + ".aee-update-new")
            replacement.unlink(missing_ok=True)
            shutil.copy2(source, replacement)
            os.replace(replacement, target)
            applied.append((target, backup))
    except Exception:
        for target, backup in reversed(applied):
            if backup and backup.is_file():
                rollback = target.with_name(target.name + ".aee-rollback")
                shutil.copy2(backup, rollback)
                os.replace(rollback, target)
            else:
                target.unlink(missing_ok=True)
        raise

    shutil.rmtree(stage, ignore_errors=True)
    (stage.parent / f"AI_Evidence_Extractor_Update_{expected_version}.zip").unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 5:
        return 2
    stage, app_dir = Path(argv[0]).resolve(), Path(argv[1]).resolve()
    parent_pid, version, app_exe = int(argv[2]), argv[3], Path(argv[4]).resolve()
    logging.basicConfig(filename=_log_file(), level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        logging.info("Waiting for app process %s to close before applying %s", parent_pid, version)
        _wait_for_process(parent_pid)
        _apply(stage, app_dir, version)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        subprocess.Popen([str(app_exe)], cwd=str(app_dir), close_fds=True,
                         creationflags=flags)
        logging.info("Applied and restarted app at version %s", version)
        return 0
    except Exception:
        logging.exception("Could not apply staged app update")
        try:
            if app_exe.is_file():
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                subprocess.Popen([str(app_exe)], cwd=str(app_dir), close_fds=True,
                                 creationflags=flags)
        except Exception:
            logging.exception("Could not reopen the app after an update failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
