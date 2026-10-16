"""Supabase authentication and profile-only account storage for the desktop app."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import html
import json
import logging
import os
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
AVATAR_BUCKET = "profile-photos"
MAX_AVATAR_BYTES = 5 * 1024 * 1024
GOOGLE_SIGN_IN_TIMEOUT = 180
log = logging.getLogger("aee.account")


class AccountError(RuntimeError):
    """A safe, user-facing account or authentication error."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _validate_config(project_url: str, anon_key: str) -> dict[str, str]:
    project_url = project_url.strip().rstrip("/")
    anon_key = anon_key.strip()
    parsed = urllib.parse.urlsplit(project_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AccountError("The Supabase project URL contains an invalid port.") from exc
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or (port is not None and not 1 <= port <= 65535)):
        raise AccountError("Enter your Supabase project URL using HTTPS.")
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise AccountError("The project URL must be the base URL, without a path or query.")
    if not anon_key or len(anon_key) > 4096:
        raise AccountError("Enter the Supabase public anon key from your project settings.")
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


def test_connection() -> None:
    config = _require_config()
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
    display_name = display_name.strip()
    if not display_name or len(display_name) > 80:
        raise AccountError("Your display name must contain 1 to 80 characters.")
    config = _require_config()
    profile, session = _profile_upsert(config, _require_session(session), display_name)
    session["display_name"] = display_name
    return profile, _save_session(session)


def upload_profile_photo(session: dict[str, Any], image_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
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

    config = _require_config()
    session = _require_session(session)
    avatar_path = f"{session['user_id']}/avatar{extension}"
    _, session = _authenticated_api(
        config, session, "POST",
        f"/storage/v1/object/{AVATAR_BUCKET}/{urllib.parse.quote(avatar_path, safe='/')}",
        content,
        {
            "Content-Type": spec[0],
            "x-upsert": "true",
        },
    )
    profile, session = _profile_upsert(
        config, session, session.get("display_name", ""), avatar_path
    )
    return profile, _save_session(session)


def _validate_email(email: str) -> str:
    email = email.strip()
    if len(email) > 254 or "@" not in email or email.startswith("@") or email.endswith("@"):
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


def sign_out(session: dict[str, Any] | None) -> str:
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
    return "You are using guest mode. Your local app data remains on this computer." + warning


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
