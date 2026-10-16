"""Secure update checks, downloads, staging, and in-app update launch."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Callable


APP_ID = "ai-evidence-extractor"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "AEE_DATA_DIR",
    str(Path.home() / "AppData" / "Local" / "AI Evidence Extractor"),
))
UPDATE_DIR = DATA_DIR / "updates"
VERSION_FILE = APP_DIR / "version.txt"
CONFIG_FILE = APP_DIR / "update_config.json"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 512 * 1024 * 1024
MAX_INSTALLER_BYTES = 2 * 1024 * 1024 * 1024  # compatibility with the legacy downloader
UPDATE_FILES = frozenset({
    "app.py", "desktop_launcher.py", "desktop_entry.py", "update_client.py",
    "update_apply.py", "version.txt", "update_config.json", "assets/aee.ico",
})


class UpdateError(RuntimeError):
    """A safe, user-facing update error."""


class UpdateNotConfigured(UpdateError):
    """Raised when the publisher has not configured an update feed."""


def current_version() -> str:
    try:
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise UpdateError("The installed app version could not be read.") from exc
    if not VERSION_RE.fullmatch(version):
        raise UpdateError("The installed app version file is invalid.")
    return version


def _version_key(value: str) -> tuple[int, int, int]:
    if not VERSION_RE.fullmatch(value):
        raise UpdateError("The update feed contains an invalid version number.")
    return tuple(int(part) for part in value.split("."))


def _manifest_url() -> str:
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UpdateNotConfigured(
            "Updates are not connected yet. The publisher needs to configure the release feed."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise UpdateError("The update configuration file could not be read.") from exc

    url = str(config.get("manifest_url", "")).strip()
    if not url:
        raise UpdateNotConfigured(
            "Updates are not connected yet. The publisher needs to configure the release feed."
        )
    if not url.lower().startswith("https://"):
        raise UpdateError("The update feed must use a secure HTTPS address.")
    return url


def check_for_updates() -> dict:
    url = _manifest_url()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AI-Evidence-Extractor-Updater/1", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if not response.geturl().lower().startswith("https://"):
                raise UpdateError("The update feed redirected to an insecure address.")
            payload = response.read(MAX_MANIFEST_BYTES + 1)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise UpdateError(
            "Could not reach the update service. Check your internet connection and try again."
        ) from exc
    if len(payload) > MAX_MANIFEST_BYTES:
        raise UpdateError("The update information is larger than expected.")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateError("The update service returned invalid release information.") from exc
    if not isinstance(manifest, dict) or manifest.get("app_id") != APP_ID:
        raise UpdateError("The update service returned information for a different app.")

    latest = str(manifest.get("version", "")).strip()
    latest_key = _version_key(latest)
    installed = current_version()
    package_url = str(manifest.get("package_url", "")).strip()
    package_digest = str(manifest.get("package_sha256", "")).strip()
    if package_url and not package_url.lower().startswith("https://"):
        raise UpdateError("The update package must be hosted over HTTPS.")
    if package_url and not SHA256_RE.fullmatch(package_digest):
        raise UpdateError("The update package is missing a valid SHA-256 checksum.")

    # Keep the installer fields for older app versions during the one-time
    # migration to the in-app updater. New clients use the package fields.
    download_url = str(manifest.get("download_url", "")).strip()
    digest = str(manifest.get("sha256", "")).strip()
    if download_url and not download_url.lower().startswith("https://"):
        raise UpdateError("The legacy installer must be hosted over HTTPS.")
    if download_url and not SHA256_RE.fullmatch(digest):
        raise UpdateError("The legacy installer is missing a valid SHA-256 checksum.")
    if latest_key > _version_key(installed) and not package_url:
        raise UpdateError("This release has no in-app update package yet.")

    return {
        "app_id": APP_ID,
        "installed_version": installed,
        "version": latest,
        "available": latest_key > _version_key(installed),
        "download_url": download_url,
        "sha256": digest.lower(),
        "package_url": package_url,
        "package_sha256": package_digest.lower(),
        "release_notes": str(manifest.get("release_notes", "")).strip(),
        "published_at": str(manifest.get("published_at", "")).strip(),
        "manifest_url": url,
    }


def download_update_installer(
    update: dict,
    progress: Callable | None = None,
) -> Path:
    """Download and checksum-verify the installer without running it."""
    if not isinstance(update, dict) or update.get("app_id") != APP_ID:
        raise UpdateError("Check for updates again before installing.")
    if not update.get("available"):
        raise UpdateError("There is no newer version to install.")
    version = str(update.get("version", ""))
    if _version_key(version) <= _version_key(current_version()):
        raise UpdateError("The installed version is already current. Check again for updates.")
    url = str(update.get("download_url", ""))
    expected_hash = str(update.get("sha256", "")).lower()
    if not url.lower().startswith("https://") or not SHA256_RE.fullmatch(expected_hash):
        raise UpdateError("The release information is incomplete or invalid.")

    output_dir = UPDATE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    installer = output_dir / f"AI_Evidence_Extractor_Setup_{version}.exe"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AI-Evidence-Extractor-Updater/1"},
    )
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if not response.geturl().lower().startswith("https://"):
                raise UpdateError("The installer download redirected to an insecure address.")
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length and content_length.isdigit() else 0
            if total > MAX_INSTALLER_BYTES:
                raise UpdateError("The update installer exceeds the supported file size.")
            with installer.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_INSTALLER_BYTES:
                        raise UpdateError("The update installer exceeds the supported file size.")
                    target.write(chunk)
                    digest.update(chunk)
                    if progress:
                        fraction = min(downloaded / total, 0.99) if total else 0
                        progress(fraction, desc="Downloading update installer")
    except UpdateError:
        installer.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        installer.unlink(missing_ok=True)
        raise UpdateError("The installer download failed. Check your connection and try again.") from exc

    if downloaded == 0 or digest.hexdigest().lower() != expected_hash:
        installer.unlink(missing_ok=True)
        raise UpdateError("The installer checksum did not match. The file was discarded for safety.")

    if progress:
        progress(1, desc="Installer downloaded and verified")
    return installer


def download_and_stage_update(
    update: dict,
    progress: Callable | None = None,
) -> dict:
    """Download a verified in-app package and stage its allowlisted files."""
    if not isinstance(update, dict) or update.get("app_id") != APP_ID:
        raise UpdateError("Check for updates again before downloading.")
    if not update.get("available"):
        raise UpdateError("There is no newer version to download.")
    version = str(update.get("version", ""))
    if _version_key(version) <= _version_key(current_version()):
        raise UpdateError("The installed app is already current. Check again for updates.")
    package_url = str(update.get("package_url", ""))
    expected_hash = str(update.get("package_sha256", "")).lower()
    if not package_url.lower().startswith("https://") or not SHA256_RE.fullmatch(expected_hash):
        raise UpdateError("This release does not contain a valid in-app update package.")

    UPDATE_DIR.mkdir(parents=True, exist_ok=True)
    package = UPDATE_DIR / f"AI_Evidence_Extractor_Update_{version}.zip"
    partial = package.with_suffix(".zip.partial")
    request = urllib.request.Request(package_url, headers={
        "User-Agent": "AI-Evidence-Extractor-Updater/2",
        "Accept": "application/octet-stream",
    })
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if not response.geturl().lower().startswith("https://"):
                raise UpdateError("The update package redirected to an insecure address.")
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length and content_length.isdigit() else 0
            if total > MAX_PACKAGE_BYTES:
                raise UpdateError("The update package exceeds the supported file size.")
            with partial.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_PACKAGE_BYTES:
                        raise UpdateError("The update package exceeds the supported file size.")
                    target.write(chunk)
                    digest.update(chunk)
                    if progress:
                        fraction = min(downloaded / total, 0.99) if total else 0
                        progress(fraction, desc="Downloading update package")
        if downloaded == 0 or digest.hexdigest().lower() != expected_hash:
            raise UpdateError("The update package checksum did not match. The package was discarded.")
        os.replace(partial, package)
    except UpdateError:
        partial.unlink(missing_ok=True)
        package.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise UpdateError("The update package download failed. Check your connection and try again.") from exc

    stage_dir = UPDATE_DIR / f"staged_{version}"
    try:
        if stage_dir.exists():
            if stage_dir.resolve().parent != UPDATE_DIR.resolve():
                raise UpdateError("The staged update path is invalid.")
            shutil.rmtree(stage_dir)
        stage_dir.mkdir()
        found: set[str] = set()
        total_uncompressed = 0
        with zipfile.ZipFile(package) as archive:
            for item in archive.infolist():
                name = item.filename
                if item.is_dir():
                    if name.rstrip("/") != "assets":
                        raise UpdateError("The update package contains an unexpected folder.")
                    continue
                relative = PurePosixPath(name)
                mode = item.external_attr >> 16
                if ("\\" in name or relative.is_absolute() or
                        any(part in ("", ".", "..") for part in relative.parts) or
                        name not in UPDATE_FILES or (mode & 0o170000) == 0o120000):
                    raise UpdateError("The update package contains an unsafe file path.")
                total_uncompressed += item.file_size
                if total_uncompressed > MAX_EXTRACTED_BYTES:
                    raise UpdateError("The extracted update is larger than expected.")
                destination = stage_dir.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                found.add(name)
        required = UPDATE_FILES - {"assets/aee.ico"}
        if not required.issubset(found):
            raise UpdateError("The update package is incomplete.")
        staged_version = (stage_dir / "version.txt").read_text(encoding="utf-8").strip()
        if staged_version != version:
            raise UpdateError("The package version does not match the release information.")
    except UpdateError:
        shutil.rmtree(stage_dir, ignore_errors=True)
        package.unlink(missing_ok=True)
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        shutil.rmtree(stage_dir, ignore_errors=True)
        package.unlink(missing_ok=True)
        raise UpdateError("The update package could not be safely prepared.") from exc
    if progress:
        progress(1, desc="Update downloaded, verified, and staged")
    return {"app_id": APP_ID, "version": version, "stage_dir": str(stage_dir),
            "package_path": str(package), "staged": True}


def apply_staged_update(staged: dict, progress: Callable | None = None) -> None:
    """Install staged app files now and report per-file progress to the UI."""
    if not isinstance(staged, dict) or staged.get("app_id") != APP_ID or not staged.get("staged"):
        raise UpdateError("Download and prepare the update before applying it.")
    version = str(staged.get("version", ""))
    _version_key(version)
    stage_dir = Path(str(staged.get("stage_dir", ""))).resolve()
    if stage_dir.parent != UPDATE_DIR.resolve() or not stage_dir.is_dir():
        raise UpdateError("The prepared update folder is missing or invalid.")
    if _version_key(version) <= _version_key(current_version()):
        raise UpdateError("The installed app is already current.")
    try:
        from update_apply import apply_staged_update as apply
        apply(stage_dir, APP_DIR, version, progress)
    except Exception as exc:
        raise UpdateError(f"The update could not be installed: {exc}") from exc


def _restart_command(data_dir: Path, app_dir: Path) -> tuple[list[str], Path]:
    import sys

    app_exe = app_dir / "AI Evidence Extractor.exe"
    launcher = app_dir / "desktop_launcher.py"
    runtime_dir = app_dir / "runtime"
    venv = data_dir / "runtimeenv"
    cuda_venv = data_dir / "runtimeenv_cuda2"
    cuda_pythonw = cuda_venv / "Scripts" / "pythonw.exe"
    if (cuda_venv / ".aee_cuda_ready").is_file() and cuda_pythonw.is_file():
        venv = cuda_venv
    pythonw = venv / "Scripts" / "pythonw.exe"

    if app_exe.is_file():
        return [str(app_exe.resolve())], app_dir
    if pythonw.is_file() and launcher.is_file() and runtime_dir.is_dir():
        return [str(pythonw.resolve()), str(launcher.resolve())], runtime_dir
    if launcher.is_file():
        return [sys.executable, str(launcher.resolve())], app_dir
    raise UpdateError("The app launcher could not be found. Reinstall the app and try again.")


def _wait_for_process_exit(pid: int) -> None:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int

        handle = kernel32.OpenProcess(0x00100000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:
                return
            raise OSError(error, ctypes.FormatError(error))
        try:
            result = kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
            if result != 0:
                error = ctypes.get_last_error()
                raise OSError(error, ctypes.FormatError(error))
        finally:
            kernel32.CloseHandle(handle)
        return

    import time

    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        time.sleep(0.1)


def _restart_after_exit(pid: int, command: list[str], cwd: str,
                        settle_seconds: float) -> None:
    import subprocess
    import time

    _wait_for_process_exit(pid)
    time.sleep(settle_seconds)
    subprocess.Popen(command, cwd=cwd, env=os.environ.copy(), close_fds=True)


def restart_app() -> None:
    """Close the current process before relaunching to release WebView2 resources."""
    import subprocess
    import sys
    import threading

    data_dir = DATA_DIR
    command, cwd = _restart_command(data_dir, APP_DIR)

    env = os.environ.copy()
    env["AEE_DATA_DIR"] = str(data_dir)
    env["AEE_APP_DIR"] = str(APP_DIR)
    try:
        flags = 0
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        helper_code = (
            "import json,sys; "
            "from update_client import _restart_after_exit; "
            "_restart_after_exit(int(sys.argv[1]), json.loads(sys.argv[2]), "
            "sys.argv[3], float(sys.argv[4]))"
        )
        log_path = data_dir / "startup.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            subprocess.Popen(
                [
                    sys.executable, "-c", helper_code, str(os.getpid()),
                    json.dumps(command), str(cwd), "2.0",
                ],
                cwd=str(APP_DIR),
                env=env,
                close_fds=True,
                creationflags=flags,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
    except Exception as exc:
        raise UpdateError(f"Could not restart the application automatically: {exc}") from exc

    def _terminate():
        import time
        time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=_terminate, daemon=True).start()
