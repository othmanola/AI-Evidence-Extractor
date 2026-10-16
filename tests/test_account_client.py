import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

import account_client


class AccountClientTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "project_url": "https://example.supabase.co",
            "anon_key": "public-test-key",
        }
        self.auth_response = {
            "user": {
                "id": "user-123",
                "email": "person@example.com",
                "user_metadata": {"display_name": "Person"},
            },
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
        }

    def test_config_requires_secure_base_url_and_public_key(self):
        for url in (
            "http://example.supabase.co",
            "https://example.supabase.co/path",
            "https://user:password@example.supabase.co",
            "https://example.supabase.co:invalid",
        ):
            with self.subTest(url=url), self.assertRaises(account_client.AccountError):
                account_client._validate_config(url, "public-key")
        with self.assertRaises(account_client.AccountError):
            account_client._validate_config("https://example.supabase.co", "")
        self.assertEqual(
            account_client._validate_config(
                "https://example.supabase.co/", "public-key"
            )["project_url"],
            "https://example.supabase.co",
        )

    def test_api_sends_public_key_and_bearer_session_to_project(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def geturl(self):
                return "https://example.supabase.co/auth/v1/settings"

            def read(self, _limit):
                return b"{}"

        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

        with patch.object(account_client.urllib.request, "urlopen", fake_urlopen):
            account_client._api(
                {
                    **self.config,
                    "project_url": "https://example.supabase.co:443",
                },
                "GET",
                "/auth/v1/settings",
                token="test-access-token",
            )
        headers = captured["request"].headers
        self.assertEqual(headers["Authorization"], "Bearer test-access-token")
        self.assertEqual(headers["Apikey"], "public-test-key")
        self.assertEqual(captured["timeout"], 25)

    def test_config_change_clears_saved_session(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_file = root / "account_config.json"
            session_file = root / "account_session.bin"
            with (
                patch.object(account_client, "APP_DATA", root),
                patch.object(account_client, "CONFIG_FILE", config_file),
                patch.object(account_client, "SESSION_FILE", session_file),
            ):
                self.assertTrue(account_client.save_config(
                    self.config["project_url"], "old-key"
                ))
                session_file.write_bytes(b"encrypted-session")
                self.assertTrue(account_client.save_config(
                    self.config["project_url"], "new-key"
                ))
                self.assertFalse(session_file.exists())
                self.assertEqual(account_client.load_config()["anon_key"], "new-key")

    def test_config_can_replace_corrupted_saved_settings(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_file = root / "account_config.json"
            session_file = root / "account_session.bin"
            config_file.write_text("{invalid", encoding="utf-8")
            session_file.write_bytes(b"encrypted-session")
            with (
                patch.object(account_client, "APP_DATA", root),
                patch.object(account_client, "CONFIG_FILE", config_file),
                patch.object(account_client, "SESSION_FILE", session_file),
            ):
                self.assertTrue(account_client.save_config(
                    self.config["project_url"], self.config["anon_key"]
                ))
                self.assertEqual(account_client.load_config(), self.config)
                self.assertFalse(session_file.exists())

    def test_guest_mode_choice_persists_after_sign_out(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guest_file = root / "guest_mode"
            session_file = root / "account_session.bin"
            session_file.write_bytes(b"saved-session")
            with (
                patch.object(account_client, "APP_DATA", root),
                patch.object(account_client, "GUEST_MODE_FILE", guest_file),
                patch.object(account_client, "SESSION_FILE", session_file),
            ):
                self.assertFalse(account_client.guest_mode_selected())
                account_client.sign_out(None)
                self.assertTrue(account_client.guest_mode_selected())
                self.assertTrue(guest_file.is_file())
                self.assertFalse(session_file.exists())

    def test_change_account_clears_guest_choice_and_saved_session(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guest_file = root / "guest_mode"
            session_file = root / "account_session.bin"
            guest_file.write_text("guest\n", encoding="utf-8")
            session_file.write_bytes(b"saved-session")
            with (
                patch.object(account_client, "GUEST_MODE_FILE", guest_file),
                patch.object(account_client, "SESSION_FILE", session_file),
            ):
                message = account_client.change_account(None)
            self.assertIn("Sign in with another account", message)
            self.assertFalse(guest_file.exists())
            self.assertFalse(session_file.exists())

    def test_email_sign_in_saves_session_without_storing_password(self):
        with (
            patch.object(account_client, "_require_config", return_value=self.config),
            patch.object(account_client, "_api", return_value=self.auth_response) as api,
            patch.object(account_client, "_save_session", side_effect=lambda value: value),
        ):
            session = account_client.sign_in("person@example.com", "temporary-password")
        self.assertEqual(session["user_id"], "user-123")
        self.assertEqual(session["refresh_token"], "refresh-token")
        self.assertNotIn("password", session)
        self.assertEqual(api.call_args.args[2], "/auth/v1/token?grant_type=password")
        self.assertNotIn("temporary-password", repr(session))

    def test_signup_reports_email_confirmation_when_session_not_returned(self):
        user_without_session = {"id": "user-123", "email": "person@example.com"}
        with (
            patch.object(account_client, "_require_config", return_value=self.config),
            patch.object(account_client, "_api", return_value=user_without_session),
        ):
            session, message = account_client.sign_up(
                "Person", "person@example.com", "long-password"
            )
        self.assertIsNone(session)
        self.assertIn("confirmation", message)

    def test_current_profile_creates_missing_row(self):
        session = {
            "user_id": "user-123",
            "email": "person@example.com",
            "display_name": "Person",
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_at": int(time.time()) + 3600,
        }
        responses = iter(([], [{"id": "user-123", "display_name": "Person"}]))
        with (
            patch.object(account_client, "_require_config", return_value=self.config),
            patch.object(account_client, "_api", side_effect=lambda *_a, **_k: next(responses)),
            patch.object(account_client, "_save_session", side_effect=lambda value: value),
        ):
            profile, avatar_url, saved_session = account_client.current_profile(session)
        self.assertEqual(profile["id"], "user-123")
        self.assertEqual(avatar_url, "")
        self.assertEqual(saved_session["user_id"], "user-123")

    def test_avatar_rejects_invalid_image_before_network_access(self):
        with TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "not-an-image.png"
            image.write_bytes(b"not a PNG")
            with patch.object(account_client, "_require_config") as require_config:
                with self.assertRaises(account_client.AccountError):
                    account_client.upload_profile_photo({}, str(image))
            require_config.assert_not_called()

    def test_google_sign_in_uses_pkce_and_loopback_callback(self):
        browser_urls = []
        thread_errors = []
        old_thread_hook = threading.excepthook
        threading.excepthook = lambda args: thread_errors.append(args.exc_value)

        def fake_api(_config, _method, path, body=None, **_kwargs):
            if path == "/auth/v1/token?grant_type=pkce":
                self.assertEqual(body["auth_code"], "one-time-code")
                self.assertEqual(len(body["code_verifier"]), 64)
                return self.auth_response
            raise AssertionError(f"Unexpected API request: {path}")

        try:
            with (
                patch.object(account_client, "_require_config", return_value=self.config),
                patch.object(account_client, "_api", side_effect=fake_api),
                patch.object(account_client, "_save_session", side_effect=lambda value: value),
                patch.object(
                    account_client.webbrowser,
                    "open",
                    side_effect=lambda url, **_kwargs: browser_urls.append(url) or True,
                ),
            ):
                transaction_id = account_client.start_google_sign_in()
                authorize = urlsplit(browser_urls[0])
                query = parse_qs(authorize.query)
                self.assertEqual(query["provider"], ["google"])
                self.assertEqual(query["code_challenge_method"], ["s256"])
                redirect = urlsplit(query["redirect_to"][0])
                self.assertEqual(redirect.hostname, "127.0.0.1")
                callback_url = (
                    f"http://127.0.0.1:{redirect.port}{redirect.path}"
                    "?code=one-time-code"
                )
                with urlopen(callback_url, timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn("signed in", response.read().decode("utf-8").lower())
                session = account_client.finish_google_sign_in(transaction_id)
                self.assertEqual(session["user_id"], "user-123")
                time.sleep(0.2)
                self.assertEqual(thread_errors, [])
        finally:
            threading.excepthook = old_thread_hook

    @unittest.skipUnless(__import__("os").name == "nt", "Windows DPAPI is required")
    def test_windows_session_encryption_round_trip(self):
        contents = b'{"refresh_token":"sensitive"}'
        protected = account_client._protect_windows(contents)
        self.assertNotEqual(protected, contents)
        self.assertEqual(account_client._unprotect_windows(protected), contents)


if __name__ == "__main__":
    unittest.main()
