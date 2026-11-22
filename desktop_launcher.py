"""Run the existing Gradio interface in a native Windows desktop window."""

from __future__ import annotations

import base64
import ctypes
import logging
import os
import subprocess
import sys
from html import escape as _html_escape
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

APP_DIR = Path(os.environ.get(
    "AEE_APP_DIR",
    str(Path(__file__).resolve().parent),
)).resolve()
_BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
APP_ICON = next(
    (candidate for candidate in (
        APP_DIR / "assets" / "aee.ico",
        _BUNDLE_DIR / "assets" / "aee.ico",
    ) if candidate.is_file()),
    APP_DIR / "assets" / "aee.ico",
)
WINDOWS_APP_ID = "AI.EvidenceExtractor.Desktop"
APP_DATA = Path(os.environ.get(
    "AEE_DATA_DIR",
    str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AI Evidence Extractor"),
))
APP_DATA.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=APP_DATA / "desktop.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("aee.desktop")


def _configure_installed_source_mode(app_dir: Path) -> None:
    app_dir = app_dir.resolve()
    os.environ["AEE_APP_DIR"] = str(app_dir)
    os.environ["AEE_SOURCE_MODE"] = "1"
    app_path = str(app_dir)
    if app_path in sys.path:
        sys.path.remove(app_path)
    sys.path.insert(0, app_path)


def _migrate_legacy_shortcuts(app_dir: Path) -> None:
    if sys.platform != "win32":
        return

    data_dir = Path(os.environ.get("AEE_DATA_DIR", str(APP_DATA)))
    shortcut_marker = data_dir / "source_launcher_shortcuts_v1"
    if shortcut_marker.is_file():
        return

    cuda_venv = data_dir / "runtimeenv_cuda2"
    cuda_pythonw = cuda_venv / "Scripts" / "pythonw.exe"
    runtimeenv_pythonw = data_dir / "runtimeenv" / "Scripts" / "pythonw.exe"
    bundled_pythonw = app_dir / "runtime" / "pythonw.exe"
    if (cuda_venv / ".aee_cuda_ready").is_file() and cuda_pythonw.is_file():
        pythonw = cuda_pythonw
    elif runtimeenv_pythonw.is_file():
        pythonw = runtimeenv_pythonw
    elif bundled_pythonw.is_file():
        pythonw = bundled_pythonw
    elif not getattr(sys, "frozen", False):
        pythonw = Path(sys.executable)
    else:
        log.warning("Could not migrate app shortcuts: no source Python launcher was found")
        return

    script = """
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
$oldTarget = [IO.Path]::GetFullPath($env:AEE_SHORTCUT_OLD_TARGET)
$newTarget = [IO.Path]::GetFullPath($env:AEE_SHORTCUT_NEW_TARGET)
$launcher = [IO.Path]::GetFullPath($env:AEE_SHORTCUT_LAUNCHER)
$appDir = [IO.Path]::GetFullPath($env:AEE_SHORTCUT_APP_DIR)
$programs = Join-Path $env:APPDATA 'Microsoft\\Windows\\Start Menu\\Programs'
$roots = @(
  [Environment]::GetFolderPath('Desktop'),
  (Join-Path $programs 'AI Evidence Extractor'),
  (Join-Path $env:APPDATA 'Microsoft\\Internet Explorer\\Quick Launch\\User Pinned\\TaskBar')
)
foreach ($root in $roots) {
  if (-not (Test-Path -LiteralPath $root)) { continue }
  Get-ChildItem -LiteralPath $root -Filter '*.lnk' -File |
    ForEach-Object {
      $shortcut = $shell.CreateShortcut($_.FullName)
      if ($shortcut.TargetPath -and
          [IO.Path]::GetFullPath($shortcut.TargetPath).Equals(
            $oldTarget, [StringComparison]::OrdinalIgnoreCase)) {
        $shortcut.TargetPath = $newTarget
        $shortcut.Arguments = '"' + $launcher + '"'
        $shortcut.WorkingDirectory = $appDir
        $shortcut.Save()
        Write-Output $_.FullName
      }
    }
}
"""
    env = os.environ.copy()
    env.update({
        "AEE_SHORTCUT_OLD_TARGET": str(app_dir / "AI Evidence Extractor.exe"),
        "AEE_SHORTCUT_NEW_TARGET": str(pythonw),
        "AEE_SHORTCUT_LAUNCHER": str(app_dir / "desktop_launcher.py"),
        "AEE_SHORTCUT_APP_DIR": str(app_dir),
    })
    encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    try:
        result = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive",
                "-WindowStyle", "Hidden", "-EncodedCommand", encoded_script,
            ],
            capture_output=True,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=env,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        log.warning("Could not migrate app shortcuts: %s", detail)
        return
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Could not migrate app shortcuts: %s", exc)
        return

    try:
        shortcut_marker.parent.mkdir(parents=True, exist_ok=True)
        shortcut_marker.write_text("1\n", encoding="utf-8")
    except OSError as exc:
        log.warning("Could not save app shortcut migration state: %s", exc)
    changed = result.stdout.strip()
    if changed:
        log.info("Migrated app shortcuts to the installed source launcher: %s", changed)


def set_windows_app_identity() -> None:
    """Give Windows a stable taskbar identity for the branded app window."""
    if os.name != "nt":
        return
    try:
        set_app_id = ctypes.WinDLL("shell32").SetCurrentProcessExplicitAppUserModelID
        set_app_id.argtypes = [ctypes.c_wchar_p]
        set_app_id.restype = ctypes.c_long
        result = set_app_id(WINDOWS_APP_ID)
        if result != 0:
            log.warning("Could not set Windows taskbar app identity: HRESULT 0x%08X", result)
    except Exception:
        log.exception("Could not set Windows taskbar app identity")


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(url: str, thread: threading.Thread,
                    startup_error: list[BaseException], timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if startup_error:
            raise RuntimeError("Could not start the local AI Evidence Extractor service") from startup_error[0]
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        if not thread.is_alive():
            raise RuntimeError("The local AI Evidence Extractor service stopped during startup")
        time.sleep(0.4)
    raise TimeoutError(f"The local app did not become ready in {timeout:.0f}s: {last_error}")


def show_error(message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("AI Evidence Extractor", message)
        root.destroy()
    except Exception:
        log.error("Unable to display error dialog: %s", message)


def main() -> None:
    set_windows_app_identity()
    if getattr(sys, "frozen", False):
        os.environ["YOLO_AUTOINSTALL"] = "false"
    import webview
    # pywebview cancels WebView2 DownloadStarting events unless this setting is enabled.
    # Gradio serves report files from the app-data outputs folder, so allow its buttons
    # to open the native Save As dialog.
    webview.settings["ALLOW_DOWNLOADS"] = True
    port = free_local_port()
    url = f"http://127.0.0.1:{port}/"
    startup_error: list[BaseException] = []
    blocks_holder = []

    splash = """<!doctype html><html><head><meta charset="utf-8"><style>
      *{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;
      background:#0b1020;color:#e6edf7;font:16px 'Segoe UI',sans-serif}
      main{text-align:center;padding:40px}h1{font-size:30px;margin:0 0 14px}
      p{color:#9aa9c2;margin:0}.spinner{width:34px;height:34px;margin:28px auto 18px;
      border:3px solid #334155;border-top-color:#38bdf8;border-radius:50%;
      animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
      </style></head><body><main><h1>AI Evidence Extractor</h1>
      <div class="spinner"></div><p>Loading AI models. The app window will be ready shortly.</p>
      <p style="margin-top:10px;font-size:13px">First startup can take up to a minute.</p>
      </main></body></html>"""

    window = webview.create_window(
        "AI Evidence Extractor",
        html=splash,
        width=1440,
        height=940,
        min_size=(1024, 720),
        text_select=True,
        confirm_close=False,
        background_color="#0b1020",
    )

    def set_window_icon(*_args) -> None:
        try:
            import clr
            from System.Drawing import Icon

            window.native.Icon = Icon(str(APP_ICON))
            log.info("Applied AI Evidence Extractor window and taskbar icon")
        except Exception as exc:
            log.info("Native title bar icon unavailable: %s", exc)

    def on_loaded() -> None:
        log.info("Desktop window content loaded")

    def run_app() -> None:
        try:
            _configure_installed_source_mode(APP_DIR)
            _migrate_legacy_shortcuts(APP_DIR)
            import gradio as gr
            import app as application

            updater = application._update_client
            log.info(
                "Loaded installed app: executable=%r; app_dir=%r; app_module=%r; "
                "updater_module=%r; version_file=%r; installed_version=%s",
                sys.executable,
                str(APP_DIR),
                str(Path(application.__file__).resolve()),
                str(Path(updater.__file__).resolve()),
                str(updater.VERSION_FILE.resolve()),
                updater.current_version(),
            )
            blocks = application.build_ui().queue(
                max_size=8, default_concurrency_limit=1
            )
            blocks_holder.append(blocks)

            def run_server() -> None:
                try:
                    blocks.launch(
                        share=False,
                        server_name="127.0.0.1",
                        server_port=port,
                        show_error=True,
                        theme=gr.themes.Base(),
                        css=application.CSS,
                        head=application._JS_HEAD,
                        max_file_size=None,
                        allowed_paths=[
                            str(application.OUTPUT_DIR.resolve()),
                            str(application._update_client.UPDATE_DIR.resolve()),
                        ],
                        prevent_thread_lock=True,
                    )
                except BaseException as exc:
                    startup_error.append(exc)
                    log.exception("Gradio server startup failed")

            server_thread = threading.Thread(
                target=run_server, name="aee-local-server", daemon=True
            )
            server_thread.start()
            wait_for_server(url, server_thread, startup_error)
            window.load_url(url)
            log.info("AI Evidence Extractor page loaded in desktop window: %s", url)
        except Exception as exc:
            log.exception("Desktop app startup failed")
            error = _html_escape(str(exc))
            try:
                window.load_html(
                    "<!doctype html><html><meta charset='utf-8'><body style=\""
                    "font:16px Segoe UI,sans-serif;background:#0b1020;color:#e6edf7;"
                    "padding:48px\"><h2>AI Evidence Extractor could not start</h2>"
                    f"<p>{error}</p><p>See startup.log in { _html_escape(str(APP_DATA)) }</p>"
                    "</body></html>"
                )
            except Exception:
                log.exception("Could not show startup error in desktop window")

    def close_server() -> None:
        try:
            if blocks_holder:
                blocks_holder[0].close()
        except Exception:
            log.exception("Could not close the local server cleanly")

    window.events.loaded += on_loaded
    window.events.before_show += set_window_icon
    threading.Thread(target=run_app, name="aee-startup", daemon=True).start()
    webview.start(
        gui="edgechromium",
        debug=False,
        private_mode=False,
        storage_path=str(APP_DATA / "webview"),
        icon=str(APP_ICON),
    )
    close_server()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        details = "".join(traceback.format_exception(exc))
        log.error("Desktop app failed:\n%s", details)
        show_error(
            "AI Evidence Extractor could not start its desktop window.\n\n"
            f"{exc}\n\nDetails: {APP_DATA / 'desktop.log'}"
        )
