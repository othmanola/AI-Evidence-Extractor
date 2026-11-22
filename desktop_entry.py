"""Small executable entry point; launches the installed private Python app."""

from __future__ import annotations

import os
import subprocess
import sys
import importlib.util
from pathlib import Path


def _launch_installed_sources(app_dir: Path, data_dir: Path) -> int:
    module_names = (
        "account_client",
        "update_apply",
        "update_client",
        "app",
        "desktop_launcher",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    previous_source_mode = os.environ.get("AEE_SOURCE_MODE")
    os.environ["AEE_SOURCE_MODE"] = "1"
    sys.path.insert(0, str(app_dir))
    try:
        for name in module_names:
            source = app_dir / f"{name}.py"
            spec = importlib.util.spec_from_file_location(name, source)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load installed application module: {source}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return int(sys.modules["desktop_launcher"].main() or 0)
    finally:
        sys.path.remove(str(app_dir))
        if previous_source_mode is None:
            os.environ.pop("AEE_SOURCE_MODE", None)
        else:
            os.environ["AEE_SOURCE_MODE"] = previous_source_mode
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def main() -> int:
    data_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AI Evidence Extractor"
    configured_dir = os.environ.get("AEE_APP_DIR")
    if configured_dir:
        app_dir = Path(configured_dir).expanduser().resolve()
    elif getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
    else:
        app_dir = Path(__file__).resolve().parent
    venv = data_dir / "runtimeenv"
    cuda_venv = data_dir / "runtimeenv_cuda2"
    cuda_pythonw = cuda_venv / "Scripts" / "pythonw.exe"
    cuda_ready = cuda_venv / ".aee_cuda_ready"
    if cuda_ready.is_file() and cuda_pythonw.is_file():
        venv = cuda_venv
    pythonw = venv / "Scripts" / "pythonw.exe"
    launcher = app_dir / "desktop_launcher.py"
    runtime_dir = app_dir / "runtime"
    if not pythonw.is_file():
        pythonw = runtime_dir / "pythonw.exe"
    log_path = data_dir / "startup.log"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not pythonw.is_file() and getattr(sys, "frozen", False):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                "Bundled Python launcher unavailable; starting installed sources "
                "inside the packaged runtime.\n"
            )
        return _launch_installed_sources(app_dir, data_dir)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"Desktop bootstrap: executable={sys.executable!r}; app_dir={str(app_dir)!r}; "
            f"pythonw_exists={pythonw.is_file()}; launcher_exists={launcher.is_file()}; "
            f"app_dir_exists={runtime_dir.is_dir()}\n"
        )

    if not pythonw.is_file() or not launcher.is_file() or not runtime_dir.is_dir():
        missing = []
        if not pythonw.is_file():
            missing.append(str(pythonw))
        if not launcher.is_file():
            missing.append(str(launcher))
        if not runtime_dir.is_dir():
            missing.append(str(runtime_dir))
        detail = "Missing app files:\n" + "\n".join(missing)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(detail + "\n")
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "AI Evidence Extractor",
                "The app is not fully installed yet. Run AI_Evidence_Extractor_Setup.exe again.\n\n"
                + detail,
            )
            root.destroy()
        except Exception:
            pass
        return 1

    env = os.environ.copy()
    env["AEE_DATA_DIR"] = str(data_dir)
    env["AEE_APP_DIR"] = str(app_dir)
    env["AEE_SOURCE_MODE"] = "1"
    try:
        process = subprocess.Popen(
            [str(pythonw.resolve()), str(launcher.resolve())],
            cwd=str(runtime_dir.resolve()),
            env=env,
            close_fds=True,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"Started desktop window process {process.pid}\n")
        return 0
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"Could not start desktop window: {exc!r}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
