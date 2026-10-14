"""Secure version checks and installer downloads for the desktop app."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable


APP_ID = "ai-evidence-extractor"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "AEE_DATA_DIR",
    str(Path.home() / "AppData" / "Local" / "AI Evidence Extractor"),
))
VERSION_FILE = APP_DIR / "version.txt"
CONFIG_FILE = APP_DIR / "update_config.json"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_INSTALLER_BYTES = 2 * 1024 * 1024 * 1024


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
    download_url = str(manifest.get("download_url", "")).strip()
    digest = str(manifest.get("sha256", "")).strip()
    if not download_url.lower().startswith("https://"):
        raise UpdateError("The release installer must be hosted over HTTPS.")
    if not SHA256_RE.fullmatch(digest):
        raise UpdateError("The release information is missing a valid SHA-256 checksum.")

    return {
        "app_id": APP_ID,
        "installed_version": installed,
        "version": latest,
        "available": latest_key > _version_key(installed),
        "download_url": download_url,
        "sha256": digest.lower(),
        "release_notes": str(manifest.get("release_notes", "")).strip(),
        "published_at": str(manifest.get("published_at", "")).strip(),
        "manifest_url": url,
    }


def download_and_launch_installer(
    update: dict,
    progress: Callable | None = None,
) -> Path:
    """Download and checksum-verify the published setup, then open its wizard."""
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

    output_dir = DATA_DIR / "updates"
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
        progress(1, desc="Starting the installer")
    try:
        if os.name == "nt":
            os.startfile(str(installer))  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen([str(installer)], cwd=str(output_dir), close_fds=True)
    except OSError as exc:
        raise UpdateError(
            f"The installer downloaded, but Windows could not open it. You can run it from: {installer}"
        ) from exc
    return installer
