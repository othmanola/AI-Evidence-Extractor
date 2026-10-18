"""Run the existing Gradio interface in a native Windows desktop window."""

from __future__ import annotations

import ctypes
import logging
import os
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
APP_DATA = Path(__import__("os").environ.get(
    "AEE_DATA_DIR", str(Path.home() / "AppData" / "Local" / "AI Evidence Extractor")
))
APP_DATA.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=APP_DATA / "desktop.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("aee.desktop")


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
            import gradio as gr
            import app as application

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
