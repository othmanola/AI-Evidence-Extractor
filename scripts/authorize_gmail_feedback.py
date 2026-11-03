"""Authorize the Gmail mailbox used by the Supabase feedback relay."""

from __future__ import annotations

import base64
import getpass
import hashlib
import http.server
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser


AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
PROFILE_URL = "https://openidconnect.googleapis.com/v1/userinfo"
SCOPES = (
    "openid",
    "email",
    "https://www.googleapis.com/auth/gmail.send",
)


class OAuthCallback(http.server.BaseHTTPRequestHandler):
    expected_state = ""
    result: dict[str, str] = {}
    completed = threading.Event()

    def do_GET(self):
        if urllib.parse.urlsplit(self.path).path != "/oauth2callback":
            self.send_error(404)
            return
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        if query.get("state", [""])[0] != self.expected_state:
            self.result["error"] = "Google callback state did not match."
            self.completed.set()
            self.send_error(400, "OAuth state did not match.")
            return
        if "error" in query:
            self.result["error"] = query["error"][0]
        else:
            self.result["code"] = query.get("code", [""])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h2>Gmail authorization complete</h2>"
            b"<p>You can close this browser tab and return to the terminal.</p></body></html>"
        )
        self.completed.set()

    def log_message(self, _format, *_args):
        return


def _request_json(url: str, data: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(data).encode("ascii"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"Google OAuth request failed (HTTP {exc.code}): {details}") from exc


def main():
    client_id = input("Google OAuth Desktop client ID: ").strip()
    client_secret = getpass.getpass("Google OAuth Desktop client secret: ").strip()
    if not client_id or not client_secret:
        raise SystemExit("Both OAuth client values are required.")

    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).decode("ascii").rstrip("=")

    OAuthCallback.expected_state = state
    OAuthCallback.result = {}
    OAuthCallback.completed = threading.Event()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), OAuthCallback)
    server.timeout = 1
    port = server.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}/oauth2callback"

    parameters = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    authorization_url = f"{AUTH_URL}?{urllib.parse.urlencode(parameters)}"
    print(
        "\nOpening Google sign-in. Complete approval in the browser, then wait "
        "for it to return to this terminal."
    )
    if not webbrowser.open(authorization_url):
        print(f"If a browser did not open, visit this URL:\n{authorization_url}")

    end_time = time.monotonic() + 600
    try:
        while not OAuthCallback.completed.is_set():
            if time.monotonic() >= end_time:
                break
            server.handle_request()
    finally:
        server.server_close()
    if not OAuthCallback.result.get("code"):
        raise SystemExit(
            "Google authorization did not complete. "
            + OAuthCallback.result.get(
                "error",
                "No callback was received within 10 minutes. Keep this terminal "
                "open and complete the Google consent page in the browser.",
            )
        )

    token_data = _request_json(TOKEN_URL, {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": OAuthCallback.result["code"],
        "code_verifier": verifier,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    })
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        raise SystemExit(
            "Google did not return a refresh token. Revoke this app's access "
            "from your Google Account and run this authorization again."
        )

    profile_request = urllib.request.Request(
        PROFILE_URL,
        headers={"Authorization": f"Bearer {token_data['access_token']}"},
    )
    try:
        with urllib.request.urlopen(profile_request, timeout=30) as response:
            profile = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Could not verify the Gmail account (HTTP {exc.code}).") from exc

    email = profile.get("email")
    if not isinstance(email, str) or not email:
        raise SystemExit("Google did not confirm the authorized account's email address.")
    if profile.get("email_verified") is not True:
        raise SystemExit("The authorized Google account's email address is not verified.")

    print(f"\nAuthorized Gmail account: {email}")
    print("Copy this refresh token directly into the Supabase GMAIL_REFRESH_TOKEN secret.")
    print("Do not paste it into chat, save it in source control, or share it.")
    print(f"\nGMAIL_REFRESH_TOKEN={refresh_token}")


if __name__ == "__main__":
    main()
