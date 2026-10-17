"""Supabase authentication and profile-only account storage for the desktop app."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import html
import json
import logging
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_DATA = Path(os.environ.get(
    "AEE_DATA_DIR",
    str(Path.home() / "AppData" / "Local" / "AI Evidence Extractor"),
))
CONFIG_FILE = APP_DATA / "account_config.json"
SESSION_FILE = APP_DATA / "account_session.bin"
GUEST_MODE_FILE = APP_DATA / "guest_mode"
GUEST_PROFILE_FILE = APP_DATA / "guest_profile.json"
GUEST_AVATAR_DIR = APP_DATA / "guest_profile_photos"
AVATAR_BUCKET = "profile-photos"
MAX_AVATAR_BYTES = 5 * 1024 * 1024
PROFILE_SETTINGS_KEYS = ("username",)
GOOGLE_SIGN_IN_TIMEOUT = 180
log = logging.getLogger("aee.account")


class AccountError(RuntimeError):
    """A safe, user-facing account or authentication error."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def guest_mode_selected() -> bool:
    try:
        return GUEST_MODE_FILE.is_file()
    except OSError as exc:
        raise AccountError("The saved guest-mode choice could not be read.") from exc


def _validate_display_name(display_name: str) -> str:
    display_name = display_name.strip()
    if not display_name or len(display_name) > 80:
        raise AccountError("Your display name must contain 1 to 80 characters.")
    return display_name


def _validate_profile_settings(
    display_name: str,
    username: str,
    email: str,
    *,
    require_email: bool,
) -> dict[str, str]:
    display_name = _validate_display_name(display_name)
    username = username.strip().removeprefix("@")
    if username and not re.fullmatch(r"[A-Za-z0-9_]{3,24}", username):
        raise AccountError(
            "Username must be 3 to 24 letters, numbers, or underscores."
        )
    email = _validate_email(email, required=require_email)
    return {
        "display_name": display_name,
        "username": username,
        "email": email,
    }


def _validate_profile_photo(extension: str, content: bytes) -> str:
    if not content or len(content) > MAX_AVATAR_BYTES:
        raise AccountError("Choose a non-empty profile photo smaller than 5 MB.")
    content_types = {
        ".jpg": ("image/jpeg", b"\xff\xd8\xff"),
        ".jpeg": ("image/jpeg", b"\xff\xd8\xff"),
        ".png": ("image/png", b"\x89PNG\r\n\x1a\n"),
        ".webp": ("image/webp", b"RIFF"),
    }
    spec = content_types.get(extension)
    if not spec or not content.startswith(spec[1]):
        raise AccountError("Use a valid JPG, PNG, or WebP image for your profile photo.")
    if extension == ".webp" and content[8:12] != b"WEBP":
        raise AccountError("The selected WebP profile photo is invalid.")
    return spec[0]


def _read_profile_photo(image_path: str) -> tuple[str, bytes, str]:
    if not image_path:
        raise AccountError("Choose a profile photo to upload.")
    source = Path(image_path)
    try:
        size = source.stat().st_size
        if not source.is_file() or size <= 0 or size > MAX_AVATAR_BYTES:
            raise AccountError("Choose a non-empty profile photo smaller than 5 MB.")
        content = source.read_bytes()
    except OSError as exc:
        raise AccountError("The selected profile photo could not be read.") from exc
    extension = source.suffix.lower()
    content_type = _validate_profile_photo(extension, content)
    return extension, content, content_type


def _guest_profile_data() -> dict[str, str]:
    try:
        data = json.loads(GUEST_PROFILE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "display_name": "", "avatar_extension": "", "contact_email": "",
            "username": "", "country": "", "phone_country_code": "+1",
            "phone_number": "", "time_zone": "UTC",
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise AccountError("The saved guest profile could not be read.") from exc
    if not isinstance(data, dict):
        raise AccountError("The saved guest profile is invalid.")
    defaults = {
        "display_name": "", "avatar_extension": "", "contact_email": "",
        "username": "", "country": "", "phone_country_code": "+1",
        "phone_number": "", "time_zone": "UTC",
    }
    if any(not isinstance(data.get(key, default), str)
           for key, default in defaults.items()):
        raise AccountError("The saved guest profile is invalid.")
    data = {key: data.get(key, default) for key, default in defaults.items()}
    display_name = data["display_name"]
    avatar_extension = data["avatar_extension"]
    if (len(display_name) > 80
            or avatar_extension not in ("", ".jpg", ".jpeg", ".png", ".webp")):
        raise AccountError("The saved guest profile is invalid.")
    _validate_profile_settings(
        display_name or "Guest", data["username"], data["contact_email"],
        require_email=False,
    )
    return data


def _save_guest_profile_data(data: dict[str, str]) -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    temporary = GUEST_PROFILE_FILE.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, GUEST_PROFILE_FILE)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AccountError("The guest profile could not be saved.") from exc


def load_guest_profile() -> dict[str, str]:
    data = _guest_profile_data()
    avatar_url = ""
    extension = data["avatar_extension"]
    if extension:
        photo_path = GUEST_AVATAR_DIR / f"avatar{extension}"
        try:
            size = photo_path.stat().st_size
            if size <= 0 or size > MAX_AVATAR_BYTES:
                raise AccountError("The saved guest profile photo is invalid.")
            content = photo_path.read_bytes()
        except OSError as exc:
            raise AccountError("The saved guest profile photo could not be read.") from exc
        content_type = _validate_profile_photo(extension, content)
        avatar_url = f"data:{content_type};base64,{base64.b64encode(content).decode('ascii')}"
    return {**data, "avatar_url": avatar_url}


def save_guest_display_name(display_name: str) -> dict[str, str]:
    display_name = _validate_display_name(display_name)
    data = _guest_profile_data()
    data["display_name"] = display_name
    _save_guest_profile_data(data)
    return load_guest_profile()


def save_guest_profile_settings(
    display_name: str,
    username: str,
    email: str,
    image_path: str | None = None,
) -> dict[str, str]:
    settings = _validate_profile_settings(
        display_name, username, email, require_email=False,
    )
    data = _guest_profile_data()
    photo = _read_profile_photo(image_path) if image_path else None
    data.update({
        "display_name": settings["display_name"],
        "username": settings["username"],
    })
    if settings["email"]:
        data["contact_email"] = settings["email"]
    if photo:
        extension, content, _content_type = photo
        GUEST_AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        destination = GUEST_AVATAR_DIR / f"avatar{extension}"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise AccountError("The guest profile photo could not be saved.") from exc
        data["avatar_extension"] = extension
    _save_guest_profile_data(data)
    if photo:
        for old_extension in (".jpg", ".jpeg", ".png", ".webp"):
            old_photo = GUEST_AVATAR_DIR / f"avatar{old_extension}"
            if old_photo != destination:
                try:
                    old_photo.unlink(missing_ok=True)
                except OSError as exc:
                    raise AccountError(
                        "The profile was saved, but an old photo could not be removed."
                    ) from exc
    return load_guest_profile()


def save_guest_profile_photo(image_path: str) -> dict[str, str]:
    extension, content, _content_type = _read_profile_photo(image_path)
    data = _guest_profile_data()
    GUEST_AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    destination = GUEST_AVATAR_DIR / f"avatar{extension}"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, destination)
        for old_extension in (".jpg", ".jpeg", ".png", ".webp"):
            old_photo = GUEST_AVATAR_DIR / f"avatar{old_extension}"
            if old_photo != destination:
                old_photo.unlink(missing_ok=True)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AccountError("The guest profile photo could not be saved.") from exc
    data["avatar_extension"] = extension
    _save_guest_profile_data(data)
    return load_guest_profile()


def _save_guest_mode() -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    temporary = GUEST_MODE_FILE.with_suffix(".tmp")
    try:
        temporary.write_text("guest\n", encoding="utf-8")
        os.replace(temporary, GUEST_MODE_FILE)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AccountError("Your guest-mode choice could not be saved.") from exc


def _validate_config(project_url: str, anon_key: str) -> dict[str, str]:
    project_url = project_url.strip()
    if project_url and "://" not in project_url:
        project_url = "https://" + project_url
    anon_key = anon_key.strip()
    parsed = urllib.parse.urlsplit(project_url)
    if (parsed.scheme == "http" and parsed.hostname
            and parsed.hostname.lower().endswith(".supabase.co")
            and not parsed.username and not parsed.password):
        parsed = parsed._replace(scheme="https")
        project_url = urllib.parse.urlunsplit(parsed)
    project_path = [part for part in parsed.path.split("/") if part]
    if (parsed.scheme == "https" and parsed.hostname == "supabase.com"
            and len(project_path) == 3 and project_path[:2] == ["dashboard", "project"]
            and re.fullmatch(r"[a-z0-9]+", project_path[2])):
        project_url = f"https://{project_path[2]}.supabase.co"
        parsed = urllib.parse.urlsplit(project_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AccountError("The Supabase project URL contains an invalid port.") from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or (port is not None and not 1 <= port <= 65535)):
        raise AccountError(
            "Enter the Supabase project API URL, such as "
            "https://your-project-ref.supabase.co. A project reference or "
            "Supabase dashboard URL is accepted too."
        )
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise AccountError("The project URL must be the base URL, without a path or query.")
    if not anon_key or len(anon_key) > 4096:
        raise AccountError("Enter the Supabase public anon key from your project settings.")
    project_url = urllib.parse.urlunsplit(
        parsed._replace(path="", query="", fragment="")
    ).rstrip("/")
    return {"project_url": project_url, "anon_key": anon_key}


def load_config() -> dict[str, str]:
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"project_url": "", "anon_key": ""}
    except (OSError, json.JSONDecodeError) as exc:
        raise AccountError("The saved account connection settings could not be read.") from exc
    if not isinstance(data, dict):
        raise AccountError("The saved account connection settings are invalid.")
    project_url = data.get("project_url", "")
    anon_key = data.get("anon_key", "")
    if not isinstance(project_url, str) or not isinstance(anon_key, str):
        raise AccountError("The saved account connection settings are invalid.")
    if not project_url and not anon_key:
        return {"project_url": "", "anon_key": ""}
    return _validate_config(project_url, anon_key)


def save_config(project_url: str, anon_key: str) -> bool:
    config = _validate_config(project_url, anon_key)
    try:
        previous = load_config()
    except AccountError:
        previous = None
    APP_DATA.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG_FILE.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temporary, CONFIG_FILE)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AccountError("The account connection settings could not be saved.") from exc
    changed = previous != config
    if changed:
        try:
            SESSION_FILE.unlink(missing_ok=True)
        except OSError as exc:
            raise AccountError(
                "The connection was saved, but the previous account session could not be removed."
            ) from exc
    return changed


def test_connection(project_url: str | None = None, anon_key: str | None = None) -> None:
    if project_url is None and anon_key is None:
        config = _require_config()
    elif project_url is None or anon_key is None:
        raise AccountError("Enter both the Supabase project URL and public API key.")
    else:
        config = _validate_config(project_url, anon_key)
    result = _api(config, "GET", "/auth/v1/settings")
    if not isinstance(result, dict):
        raise AccountError("Supabase returned an invalid connection response.")


def config_is_ready() -> bool:
    config = load_config()
    return bool(config["project_url"] and config["anon_key"])


def _require_config() -> dict[str, str]:
    config = load_config()
    if not config["project_url"] or not config["anon_key"]:
        raise AccountError(
            "Connect a Supabase project in Account > Connection setup before signing in."
        )
    return config


def _decode_response(response: bytes) -> Any:
    if not response:
        return {}
    try:
        return json.loads(response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AccountError("The account service returned an invalid response.") from exc


def _same_https_origin(candidate, expected) -> bool:
    if (candidate.scheme != "https" or expected.scheme != "https"
            or not candidate.hostname or not expected.hostname
            or candidate.username or candidate.password
            or expected.username or expected.password):
        return False
    try:
        return (
            candidate.hostname.lower(),
            candidate.port or 443,
        ) == (
            expected.hostname.lower(),
            expected.port or 443,
        )
    except ValueError:
        return False


def _api(
    config: dict[str, str],
    method: str,
    path: str,
    body: dict[str, Any] | bytes | None = None,
    token: str | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    url = config["project_url"] + "/" + path.lstrip("/")
    request_headers = {
        "Accept": "application/json",
        "apikey": config["anon_key"],
        "User-Agent": "AI-Evidence-Extractor/1",
    }
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    if headers:
        request_headers.update(headers)
    if isinstance(body, bytes):
        payload = body
    elif body is not None:
        payload = json.dumps(body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    else:
        payload = None
    request = urllib.request.Request(
        url, data=payload, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            final_url = urllib.parse.urlsplit(response.geturl())
            project = urllib.parse.urlsplit(config["project_url"])
            if not _same_https_origin(final_url, project):
                raise AccountError("The account service redirected to an unexpected address.")
            payload = response.read(8 * 1024 * 1024 + 1)
            if len(payload) > 8 * 1024 * 1024:
                raise AccountError("The account service response is larger than expected.")
            return _decode_response(payload)
    except urllib.error.HTTPError as exc:
        details = exc.read(16_384)
        message = "The account service rejected the request."
        try:
            data = json.loads(details.decode("utf-8"))
            if isinstance(data, dict):
                message = str(
                    data.get("msg") or data.get("message")
                    or data.get("error_description") or data.get("error")
                    or message
                )
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        raise AccountError(f"{message} (HTTP {exc.code})", status=exc.code) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AccountError(
            "Could not reach Supabase. Check the project URL and your internet connection."
        ) from exc


def _require_session(session: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(session, dict) or not session.get("refresh_token"):
        raise AccountError("Sign in to your account to continue.")
    return session


def _protect_windows(data: bytes) -> bytes:
    if os.name != "nt":
        raise AccountError("Secure account sessions are currently supported on Windows only.")

    class DataBlob(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(DataBlob), ctypes.c_wchar_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
        ctypes.POINTER(DataBlob),
    ]
    crypt32.CryptProtectData.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    source_buffer = ctypes.create_string_buffer(data)
    source = DataBlob(
        len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte))
    )
    encrypted = DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(source), "AI Evidence Extractor account session",
        None, None, None, 0x1, ctypes.byref(encrypted),
    ):
        error = ctypes.get_last_error()
        raise AccountError(
            f"Windows could not protect the saved account session (error {error})."
        )
    try:
        return ctypes.string_at(encrypted.pbData, encrypted.cbData)
    finally:
        kernel32.LocalFree(encrypted.pbData)


def _unprotect_windows(data: bytes) -> bytes:
    if os.name != "nt":
        raise AccountError("Secure account sessions are currently supported on Windows only.")

    class DataBlob(ctypes.Structure):
        _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(DataBlob), ctypes.POINTER(ctypes.c_wchar_p),
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
        ctypes.POINTER(DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    source_buffer = ctypes.create_string_buffer(data)
    source = DataBlob(
        len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte))
    )
    decrypted = DataBlob()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(decrypted)
    ):
        error = ctypes.get_last_error()
        raise AccountError(
            f"Windows could not unlock the saved account session (error {error})."
        )
    try:
        return ctypes.string_at(decrypted.pbData, decrypted.cbData)
    finally:
        kernel32.LocalFree(decrypted.pbData)


def _save_session(session: dict[str, Any]) -> dict[str, Any]:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(session, separators=(",", ":")).encode("utf-8")
    protected = _protect_windows(payload)
    temporary = SESSION_FILE.with_suffix(".tmp")
    try:
        temporary.write_bytes(protected)
        os.replace(temporary, SESSION_FILE)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AccountError("The account session could not be saved securely.") from exc
    return session


def load_session() -> dict[str, Any] | None:
    try:
        protected = SESSION_FILE.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AccountError("The saved account session could not be read.") from exc
    try:
        data = json.loads(_unprotect_windows(protected).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AccountError("The saved account session is damaged and could not be read.") from exc
    if not isinstance(data, dict) or not data.get("user_id") or not data.get("refresh_token"):
        raise AccountError("The saved account session is incomplete.")
    return data


def _session_from_auth(data: dict[str, Any]) -> dict[str, Any]:
    user = data.get("user") if isinstance(data, dict) else None
    if not isinstance(user, dict) or not data.get("access_token") or not data.get("refresh_token"):
        raise AccountError(
            "Sign-in did not return a session. Check your email confirmation settings and try again."
        )
    metadata = user.get("user_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    expires_in = data.get("expires_in", 3600)
    try:
        expires_at = int(data.get("expires_at") or time.time() + int(expires_in))
    except (TypeError, ValueError):
        expires_at = int(time.time()) + 3600
    session = {
        "user_id": str(user.get("id", "")),
        "email": str(user.get("email", "")),
        "display_name": str(
            metadata.get("display_name") or metadata.get("full_name")
            or metadata.get("name") or ""
        ),
        "username": str(metadata.get("username") or ""),
        "country": str(metadata.get("country") or ""),
        "phone_country_code": str(metadata.get("phone_country_code") or "+1"),
        "phone_number": str(metadata.get("phone_number") or ""),
        "time_zone": str(metadata.get("time_zone") or "UTC"),
        "created_at": str(user.get("created_at") or ""),
        "access_token": str(data["access_token"]),
        "refresh_token": str(data["refresh_token"]),
        "expires_at": expires_at,
    }
    if not session["user_id"]:
        raise AccountError("The account service returned an invalid user identity.")
    return session


def _refresh_session(config: dict[str, str], session: dict[str, Any]) -> dict[str, Any]:
    data = _api(
        config, "POST", "/auth/v1/token?grant_type=refresh_token",
        {"refresh_token": session["refresh_token"]},
    )
    refreshed = _session_from_auth(data)
    return _save_session(refreshed)


def _authenticated_api(
    config: dict[str, str],
    session: dict[str, Any],
    method: str,
    path: str,
    body: dict[str, Any] | bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    session = _require_session(session)
    if int(session.get("expires_at", 0) or 0) <= int(time.time()) + 60:
        session = _refresh_session(config, session)
    try:
        result = _api(config, method, path, body, session["access_token"], headers)
    except AccountError as exc:
        if exc.status != 401:
            raise
        session = _refresh_session(config, session)
        result = _api(config, method, path, body, session["access_token"], headers)
    return result, session


def _profile_upsert(
    config: dict[str, str],
    session: dict[str, Any],
    display_name: str,
    avatar_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body: dict[str, Any] = {
        "id": session["user_id"],
        "display_name": display_name.strip()[:80],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if avatar_path is not None:
        body["avatar_path"] = avatar_path
    result, session = _authenticated_api(
        config, session, "POST",
        "/rest/v1/profiles?on_conflict=id",
        body,
        {"Prefer": "resolution=merge-duplicates,return=representation"},
    )
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return result[0], session
    return body, session


def _signed_avatar_url(
    config: dict[str, str], session: dict[str, Any], avatar_path: str
) -> tuple[str, dict[str, Any]]:
    escaped_path = urllib.parse.quote(avatar_path, safe="/")
    result, session = _authenticated_api(
        config, session, "POST",
        f"/storage/v1/object/sign/{AVATAR_BUCKET}/{escaped_path}",
        {"expiresIn": 3600},
    )
    if not isinstance(result, dict) or not result.get("signedURL"):
        raise AccountError("The profile photo could not be loaded from storage.")
    signed_url = str(result["signedURL"])
    if signed_url.startswith("/object/"):
        signed_url = "/storage/v1" + signed_url
    if signed_url.startswith("/"):
        return config["project_url"] + signed_url, session
    parsed_url = urllib.parse.urlsplit(signed_url)
    project = urllib.parse.urlsplit(config["project_url"])
    if not _same_https_origin(parsed_url, project):
        raise AccountError("The profile photo service returned an invalid address.")
    return signed_url, session


def current_profile(session: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    config = _require_config()
    session = _require_session(session)
    query = urllib.parse.urlencode({
        "id": f"eq.{session['user_id']}",
        "select": "id,display_name,avatar_path",
        "limit": "1",
    })
    result, session = _authenticated_api(
        config, session, "GET", f"/rest/v1/profiles?{query}"
    )
    if not isinstance(result, list) or not result:
        profile, session = _profile_upsert(
            config, session, session.get("display_name", "")
        )
    else:
        profile = result[0]
    session["display_name"] = str(
        profile.get("display_name") or session.get("display_name") or ""
    )
    avatar_url = ""
    avatar_path = profile.get("avatar_path")
    if avatar_path:
        avatar_url, session = _signed_avatar_url(config, session, str(avatar_path))
    session = _save_session(session)
    return profile, avatar_url, session


def save_display_name(session: dict[str, Any], display_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    display_name = _validate_display_name(display_name)
    config = _require_config()
    profile, session = _profile_upsert(config, _require_session(session), display_name)
    session["display_name"] = display_name
    return profile, _save_session(session)


def save_profile_settings(
    session: dict[str, Any],
    display_name: str,
    username: str,
    email: str,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    session = _require_session(session)
    settings = _validate_profile_settings(
        display_name, username, email, require_email=True,
    )
    metadata = {key: settings[key] for key in PROFILE_SETTINGS_KEYS}
    metadata["display_name"] = settings["display_name"]
    email_changed = settings["email"].casefold() != str(session.get("email", "")).casefold()
    config = _require_config()
    result, session = _authenticated_api(
        config, session, "PUT", "/auth/v1/user",
        {
            "data": metadata,
            **({"email": settings["email"]} if email_changed else {}),
        },
    )
    if not isinstance(result, dict):
        raise AccountError("Supabase did not confirm the profile update.")
    returned_email = str(result.get("email") or session.get("email", ""))
    email_pending = email_changed and returned_email.casefold() != settings["email"].casefold()
    if not email_pending:
        session["email"] = returned_email
    session.update(metadata)
    profile, session = _profile_upsert(
        config, session, settings["display_name"]
    )
    return profile, _save_session(session), email_pending


def upload_profile_photo(session: dict[str, Any], image_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    extension, content, content_type = _read_profile_photo(image_path)
    config = _require_config()
    session = _require_session(session)
    avatar_path = f"{session['user_id']}/avatar{extension}"
    _, session = _authenticated_api(
        config, session, "POST",
        f"/storage/v1/object/{AVATAR_BUCKET}/{urllib.parse.quote(avatar_path, safe='/')}",
        content,
        {
            "Content-Type": content_type,
            "x-upsert": "true",
        },
    )
    profile, session = _profile_upsert(
        config, session, session.get("display_name", ""), avatar_path
    )
    return profile, _save_session(session)


def _validate_email(email: str, *, required: bool = True) -> str:
    email = email.strip()
    if not email and not required:
        return ""
    if (len(email) > 254
            or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email)):
        raise AccountError("Enter a valid email address.")
    return email


def sign_up(display_name: str, email: str, password: str) -> tuple[dict[str, Any] | None, str]:
    display_name = display_name.strip()
    email = _validate_email(email)
    if not display_name or len(display_name) > 80:
        raise AccountError("Enter a display name with 1 to 80 characters.")
    if len(password) < 8:
        raise AccountError("Use a password with at least 8 characters.")
    config = _require_config()
    result = _api(config, "POST", "/auth/v1/signup", {
        "email": email,
        "password": password,
        "data": {"display_name": display_name},
    })
    if not isinstance(result, dict) or not result.get("access_token"):
        return None, (
            "Account created. Check your email for the confirmation link, then return here "
            "and sign in. Your video and analysis data stay on this computer."
        )
    session = _save_session(_session_from_auth(result))
    return session, "Your account is ready."


def sign_in(email: str, password: str) -> dict[str, Any]:
    email = _validate_email(email)
    if not password:
        raise AccountError("Enter your password.")
    config = _require_config()
    result = _api(config, "POST", "/auth/v1/token?grant_type=password", {
        "email": email,
        "password": password,
    })
    if not isinstance(result, dict):
        raise AccountError("The account service returned an invalid sign-in response.")
    return _save_session(_session_from_auth(result))


def _revoke_and_clear_session(session: dict[str, Any] | None) -> str:
    warning = ""
    if isinstance(session, dict) and session.get("access_token"):
        try:
            config = _require_config()
            _api(config, "POST", "/auth/v1/logout", token=session["access_token"])
        except AccountError as exc:
            warning = f" Local sign-out completed, but the server session could not be revoked: {exc}"
    try:
        SESSION_FILE.unlink(missing_ok=True)
    except OSError as exc:
        raise AccountError("The saved account session could not be removed.") from exc
    return warning


def sign_out(session: dict[str, Any] | None) -> str:
    warning = _revoke_and_clear_session(session)
    try:
        _save_guest_mode()
    except OSError as exc:
        raise AccountError("Your guest-mode choice could not be saved.") from exc
    return "You are using guest mode. Your local app data remains on this computer." + warning


def change_account(session: dict[str, Any] | None) -> str:
    warning = _revoke_and_clear_session(session)
    try:
        GUEST_MODE_FILE.unlink(missing_ok=True)
    except OSError as exc:
        raise AccountError("The previous account choice could not be cleared.") from exc
    return "Signed out. Sign in with another account or continue as a guest." + warning


class _OAuthTransaction:
    def __init__(self, server: ThreadingHTTPServer, verifier: str):
        self.server = server
        self.verifier = verifier
        self.event = threading.Event()
        self.session: dict[str, Any] | None = None
        self.error: str | None = None
        self.timer: threading.Timer | None = None


_OAUTH_LOCK = threading.Lock()
_OAUTH_TRANSACTIONS: dict[str, _OAuthTransaction] = {}


def start_google_sign_in() -> str:
    config = _require_config()
    transaction_id = uuid.uuid4().hex
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode("ascii")
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _reply(self, title: str, message: str, status: int = 200) -> None:
            body = (
                "<!doctype html><html><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width'>"
                "<title>AI Evidence Extractor</title>"
                "<style>body{font:16px Segoe UI,sans-serif;background:#0b1020;color:#e6edf7;"
                "min-height:100vh;display:grid;place-items:center;margin:0}.card{max-width:520px;"
                "padding:36px;border:1px solid #334155;border-radius:20px;background:#111827}"
                "h1{font-size:23px}p{color:#a7b3c7;line-height:1.6}</style><main class='card'>"
                f"<h1>{html.escape(title)}</h1><p>{html.escape(message)}</p></main></html>"
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != "/oauth/callback":
                self._reply("Sign-in link not found", "Return to the app and start Google sign-in again.", 404)
                return
            query = urllib.parse.parse_qs(parsed.query)
            error = query.get("error_description", query.get("error", [""]))[0]
            code = query.get("code", [""])[0]
            if error or not code:
                message = error or "Google sign-in did not return an authorization code."
                _complete_oauth_transaction(
                    transaction_id, error=f"Google sign-in could not be completed: {message}"
                )
                self._reply("Sign-in was not completed", "Return to the app and try again.", 400)
                threading.Thread(
                    target=_shutdown_oauth_server, args=(self.server,), daemon=True
                ).start()
                return
            try:
                auth = _api(
                    config, "POST", "/auth/v1/token?grant_type=pkce",
                    {"auth_code": code, "code_verifier": verifier},
                )
                session = _save_session(_session_from_auth(auth))
                if _complete_oauth_transaction(transaction_id, session=session):
                    self._reply("You are signed in", "Return to AI Evidence Extractor to continue.")
                else:
                    self._reply("Sign-in timed out", "Return to the app and start sign-in again.", 400)
            except AccountError as exc:
                _complete_oauth_transaction(transaction_id, error=str(exc))
                self._reply("Sign-in could not be completed", str(exc), 400)
            except Exception as exc:
                log.exception("Google sign-in callback failed")
                message = "Google sign-in failed unexpectedly. Start the sign-in again."
                _complete_oauth_transaction(transaction_id, error=message)
                self._reply("Sign-in could not be completed", message, 500)
            finally:
                threading.Thread(
                    target=_shutdown_oauth_server, args=(self.server,), daemon=True
                ).start()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), CallbackHandler)
    except OSError as exc:
        raise AccountError("The local Google sign-in callback could not be started.") from exc
    server.daemon_threads = True
    server._aee_shutdown_lock = threading.Lock()
    server._aee_shutdown_started = False
    port = int(server.server_address[1])
    redirect_uri = f"http://127.0.0.1:{port}/oauth/callback"
    transaction = _OAuthTransaction(server, verifier)
    timer = threading.Timer(
        GOOGLE_SIGN_IN_TIMEOUT,
        lambda: _expire_oauth_transaction(transaction_id),
    )
    timer.daemon = True
    transaction.timer = timer
    with _OAUTH_LOCK:
        _OAUTH_TRANSACTIONS[transaction_id] = transaction
    timer.start()
    threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.2},
        name="aee-google-oauth-callback",
        daemon=True,
    ).start()
    authorize_url = config["project_url"] + "/auth/v1/authorize?" + urllib.parse.urlencode({
        "provider": "google",
        "redirect_to": redirect_uri,
        "code_challenge": challenge,
        "code_challenge_method": "s256",
    })
    try:
        opened = webbrowser.open(authorize_url, new=2, autoraise=True)
    except OSError as exc:
        _expire_oauth_transaction(transaction_id, "Could not open the default web browser.")
        with _OAUTH_LOCK:
            _OAUTH_TRANSACTIONS.pop(transaction_id, None)
        raise AccountError("Could not open the default web browser for Google sign-in.") from exc
    if not opened:
        _expire_oauth_transaction(transaction_id, "Could not open the default web browser.")
        with _OAUTH_LOCK:
            _OAUTH_TRANSACTIONS.pop(transaction_id, None)
        raise AccountError("Could not open the default web browser for Google sign-in.")
    return transaction_id


def _shutdown_oauth_server(server: ThreadingHTTPServer) -> None:
    with server._aee_shutdown_lock:
        if server._aee_shutdown_started:
            return
        server._aee_shutdown_started = True
    server.shutdown()
    server.server_close()


def _complete_oauth_transaction(
    transaction_id: str,
    session: dict[str, Any] | None = None,
    error: str | None = None,
) -> bool:
    with _OAUTH_LOCK:
        transaction = _OAUTH_TRANSACTIONS.get(transaction_id)
        if not transaction or transaction.event.is_set():
            return False
        transaction.session = session
        transaction.error = error
        transaction.event.set()
        if transaction.timer:
            transaction.timer.cancel()
        return True


def _expire_oauth_transaction(transaction_id: str, message: str | None = None) -> None:
    with _OAUTH_LOCK:
        transaction = _OAUTH_TRANSACTIONS.get(transaction_id)
        if not transaction:
            return
        if transaction.event.is_set():
            return
        transaction.error = message or "Google sign-in timed out. Start the sign-in again."
        transaction.event.set()
    threading.Thread(
        target=_shutdown_oauth_server, args=(transaction.server,), daemon=True
    ).start()


def finish_google_sign_in(transaction_id: str | None) -> dict[str, Any] | None:
    if not transaction_id:
        raise AccountError("Start Google sign-in before completing it.")
    with _OAUTH_LOCK:
        transaction = _OAUTH_TRANSACTIONS.get(transaction_id)
    if not transaction:
        raise AccountError("This Google sign-in attempt expired. Start it again.")
    if not transaction.event.is_set():
        return None
    with _OAUTH_LOCK:
        _OAUTH_TRANSACTIONS.pop(transaction_id, None)
    if transaction.timer:
        transaction.timer.cancel()
    if transaction.error:
        raise AccountError(transaction.error)
    if not transaction.session:
        raise AccountError("Google sign-in finished without an account session.")
    return transaction.session
