"""Small executable entry point; launches the installed private Python app."""

from __future__ import annotations

import os
import subprocess
import sys
import importlib.util
import queue
import shutil
import threading
from pathlib import Path


def _run_gpu_setup(python: Path, app_dir: Path, data_dir: Path) -> tuple[int, list[str]]:
    environment = os.environ.copy()
    environment.update({
        "AEE_APP_DIR": str(app_dir),
        "AEE_DATA_DIR": str(data_dir),
        "AEE_SOURCE_MODE": "1",
        "YOLO_AUTOINSTALL": "false",
    })
    code = (
        "import app; "
        "app._install_realtime_gpu(progress=lambda fraction, description: "
        "print('AEE_PROGRESS:' + description, flush=True))"
    )
    process = subprocess.Popen(
        [str(python), "-c", code],
        cwd=str(app_dir),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.stdout is None:
        raise RuntimeError("The GPU setup process did not provide progress output.")

    messages = queue.Queue()
    output = []

    def read_output():
        for line in process.stdout:
            messages.put(line.rstrip())
        messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        import tkinter as tk
        from tkinter import ttk

        window = tk.Tk()
        window.title("AI Evidence Extractor")
        window.resizable(False, False)
        window.geometry("440x150")
        window.protocol("WM_DELETE_WINDOW", lambda: None)
        container = ttk.Frame(window, padding=24)
        container.pack(fill="both", expand=True)
        status = tk.StringVar(value="Connecting to your NVIDIA GPU…")
        ttk.Label(container, textvariable=status, wraplength=390).pack(anchor="w")
        progress = ttk.Progressbar(container, mode="indeterminate")
        progress.pack(fill="x", pady=(16, 8))
        progress.start(12)
        ttk.Label(
            container,
            text="First-time setup downloads CUDA components. Please wait.",
            wraplength=390,
        ).pack(anchor="w")
        window.update_idletasks()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        window.geometry(f"+{(screen_width - 440) // 2}+{(screen_height - 150) // 2}")

        finished = False

        def update_progress():
            nonlocal finished
            while True:
                try:
                    line = messages.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    finished = True
                    continue
                output.append(line)
                del output[:-80]
                if line.startswith("AEE_PROGRESS:"):
                    status.set(line.removeprefix("AEE_PROGRESS:"))
            if finished:
                progress.stop()
                window.destroy()
            else:
                window.after(100, update_progress)

        window.after(100, update_progress)
        window.mainloop()
    except Exception:
        reader.join()
        while True:
            try:
                line = messages.get_nowait()
            except queue.Empty:
                break
            if line is not None:
                output.append(line)
                del output[:-80]

    return process.wait(), output


def _prepare_cuda_runtime(app_dir: Path, data_dir: Path) -> bool:
    if sys.platform != "win32":
        return False

    data_dir.mkdir(parents=True, exist_ok=True)
    cuda_venv = data_dir / "runtimeenv_cuda2"
    cuda_pythonw = cuda_venv / "Scripts" / "pythonw.exe"
    ready_marker = cuda_venv / ".aee_cuda_ready"
    if ready_marker.is_file() and cuda_pythonw.is_file():
        os.environ.pop("AEE_GPU_AUTOCONNECT_ERROR", None)
        return True

    if os.environ.get("AEE_GPU_SETUP_ATTEMPTED") == "1":
        return False

    if not shutil.which("nvidia-smi"):
        os.environ.pop("AEE_GPU_AUTOCONNECT_ERROR", None)
        return False

    bootstrap_candidates = (
        data_dir / "runtimeenv" / "Scripts" / "python.exe",
        app_dir / "runtime" / "python.exe",
    )
    bootstrap_python = next(
        (candidate for candidate in bootstrap_candidates if candidate.is_file()),
        None,
    )
    os.environ["AEE_GPU_SETUP_ATTEMPTED"] = "1"
    if bootstrap_python is None:
        error = "The app's private Python runtime was not found."
        os.environ["AEE_GPU_AUTOCONNECT_ERROR"] = error
        with (data_dir / "startup.log").open("a", encoding="utf-8") as log:
            log.write(f"Automatic NVIDIA GPU setup failed: {error}\n")
        return False

    try:
        return_code, output = _run_gpu_setup(
            bootstrap_python, app_dir, data_dir
        )
    except Exception as exc:
        output = [str(exc)]
        return_code = 1

    with (data_dir / "startup.log").open("a", encoding="utf-8") as log:
        log.write("Automatic NVIDIA GPU setup output:\n")
        log.write("\n".join(output[-40:]) + "\n")

    if (
        return_code == 0
        and ready_marker.is_file()
        and cuda_pythonw.is_file()
    ):
        os.environ.pop("AEE_GPU_AUTOCONNECT_ERROR", None)
        return True

    detail = next(
        (line for line in reversed(output) if line and not line.startswith("AEE_PROGRESS:")),
        "GPU setup did not create a verified CUDA runtime.",
    )
    os.environ["AEE_GPU_AUTOCONNECT_ERROR"] = detail[-500:]
    return False


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
    data_dir = Path(os.environ.get(
        "AEE_DATA_DIR",
        str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AI Evidence Extractor"),
    )).resolve()
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
    launcher = app_dir / "desktop_launcher.py"
    runtime_dir = app_dir / "runtime"
    log_path = data_dir / "startup.log"
    data_dir.mkdir(parents=True, exist_ok=True)
    _prepare_cuda_runtime(app_dir, data_dir)
    if cuda_ready.is_file() and cuda_pythonw.is_file():
        venv = cuda_venv
    pythonw = venv / "Scripts" / "pythonw.exe"
    if not pythonw.is_file():
        pythonw = runtime_dir / "pythonw.exe"

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
