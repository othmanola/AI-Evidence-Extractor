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
    def test_google_provider_disabled_error_explains_supabase_configuration(self):
        message = account_client._google_provider_error_message(
            "Unsupported provider: provider is not enabled"
        )

        self.assertIn("not enabled in the connected Supabase project", message)
        self.assertIn("Authentication > Sign In / Providers", message)
        self.assertIn("Google OAuth client ID and secret", message)

    def test_google_provider_error_message_preserves_other_errors(self):
        message = account_client._google_provider_error_message(
            "OAuth request was cancelled"
        )

        self.assertEqual(message, "OAuth request was cancelled")

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
            "http://example.com",
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
        self.assertEqual(
            account_client._validate_config(
                "example.supabase.co", "public-key"
            )["project_url"],
            "https://example.supabase.co",
        )
        self.assertEqual(
            account_client._validate_config(
                "https://supabase.com/dashboard/project/abcd1234",
                "public-key",
            )["project_url"],
            "https://abcd1234.supabase.co",
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

    def test_connection_checks_unsaved_publishable_key(self):
        with patch.object(account_client, "_api", return_value={}) as api:
            account_client.test_connection(
                "https://example.supabase.co", "sb_publishable_test-key"
            )

        api.assert_called_once_with(
            {
                "project_url": "https://example.supabase.co",
                "anon_key": "sb_publishable_test-key",
            },
            "GET",
            "/auth/v1/settings",
        )

    def test_connection_requires_both_candidate_values(self):
        with self.assertRaises(account_client.AccountError):
            account_client.test_connection("https://example.supabase.co", None)

    def test_missing_project_config_has_user_safe_guidance(self):
        with (
            patch.object(
                account_client,
                "load_config",
                return_value={"project_url": "", "anon_key": ""},
            ),
            self.assertRaisesRegex(
                account_client.AccountError, "Contact the app administrator"
            ),
        ):
            account_client._require_config()

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

    def test_guest_profile_name_is_saved_locally(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch.object(account_client, "APP_DATA", root),
                patch.object(account_client, "GUEST_PROFILE_FILE",
                             root / "guest_profile.json"),
                patch.object(account_client, "GUEST_AVATAR_DIR",
                             root / "guest_profile_photos"),
            ):
                profile = account_client.save_guest_display_name("  Local Guest  ")
                self.assertEqual(profile["display_name"], "Local Guest")
                self.assertEqual(account_client.load_guest_profile(), profile)
                self.assertTrue((root / "guest_profile.json").is_file())

    def test_guest_profile_settings_persist_identity_without_location_fields(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch.object(account_client, "APP_DATA", root),
                patch.object(account_client, "GUEST_PROFILE_FILE",
                             root / "guest_profile.json"),
                patch.object(account_client, "GUEST_AVATAR_DIR",
                             root / "guest_profile_photos"),
            ):
                (root / "guest_profile.json").write_text(
                    '{"contact_email":"saved@example.com","country":"Canada","phone_country_code":"+1",'
                    '"phone_number":"4165551234","time_zone":"America/Toronto"}',
                    encoding="utf-8",
                )
                profile = account_client.save_guest_profile_settings(
                    "Local Guest", "@local_guest", "guest@example.com",
                )
                loaded = account_client.load_guest_profile()

        self.assertEqual(profile["username"], "local_guest")
        self.assertEqual(profile["contact_email"], "guest@example.com")
        self.assertEqual(profile["country"], "Canada")
        self.assertEqual(profile["phone_country_code"], "+1")
        self.assertEqual(profile["phone_number"], "4165551234")
        self.assertEqual(profile["time_zone"], "America/Toronto")
        self.assertEqual(loaded, profile)

    def test_guest_profile_save_preserves_legacy_contact_email(self):
        with TemporaryDirectory() as temp_dir:
            profile_file = Path(temp_dir) / "guest_profile.json"
            profile_file.write_text(
                '{"display_name":"Guest","contact_email":"saved@example.com"}',
                encoding="utf-8",
            )
            with (
                patch.object(account_client, "APP_DATA", Path(temp_dir)),
                patch.object(account_client, "GUEST_PROFILE_FILE", profile_file),
                patch.object(
                    account_client, "GUEST_AVATAR_DIR",
                    Path(temp_dir) / "guest_profile_photos",
                ),
            ):
                profile = account_client.save_guest_profile_settings(
                    "Guest", "", "",
                )

        self.assertEqual(profile["contact_email"], "saved@example.com")

    def test_old_guest_profile_files_receive_defaults(self):
        with TemporaryDirectory() as temp_dir:
            profile_file = Path(temp_dir) / "guest_profile.json"
            profile_file.write_text(
                '{"display_name":"Older guest","avatar_extension":""}',
                encoding="utf-8",
            )
            with (
                patch.object(account_client, "GUEST_PROFILE_FILE", profile_file),
                patch.object(account_client, "GUEST_AVATAR_DIR",
                             Path(temp_dir) / "guest_profile_photos"),
            ):
                profile = account_client.load_guest_profile()

        self.assertEqual(profile["display_name"], "Older guest")
        self.assertEqual(profile["username"], "")
        self.assertEqual(profile["time_zone"], "UTC")

    def test_profile_settings_validation_rejects_invalid_username_and_email(self):
        valid = ("Person", "person_123", "person@example.com")
        invalid_values = (
            {"username": "x"},
            {"email": "not-an-email"},
        )
        for override in invalid_values:
            values = dict(zip(
                ("display_name", "username", "email"), valid,
            ))
            values.update(override)
            with self.subTest(override=override), self.assertRaises(
                account_client.AccountError
            ):
                account_client._validate_profile_settings(**values, require_email=True)

        self.assertEqual(
            account_client._validate_profile_settings(
                "Person", "", "", require_email=False
            )["username"],
            "",
        )

    def test_guest_profile_photo_is_saved_and_loaded_locally(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            photo = root / "guest.png"
            photo.write_bytes(b"\x89PNG\r\n\x1a\nlocal image bytes")
            with (
                patch.object(account_client, "APP_DATA", root),
                patch.object(account_client, "GUEST_PROFILE_FILE",
                             root / "guest_profile.json"),
                patch.object(account_client, "GUEST_AVATAR_DIR",
                             root / "guest_profile_photos"),
            ):
                profile = account_client.save_guest_profile_photo(str(photo))
                self.assertIn("data:image/png;base64,", profile["avatar_url"])
                self.assertEqual(
                    (root / "guest_profile_photos" / "avatar.png").read_bytes(),
                    photo.read_bytes(),
                )

    def test_guest_profile_name_rejects_blank_values(self):
        with self.assertRaises(account_client.AccountError):
            account_client.save_guest_display_name(" ")

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

    def test_signed_in_profile_settings_store_metadata_and_detect_pending_email(self):
        session = {
            "user_id": "user-123",
            "email": "person@example.com",
            "display_name": "Person",
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_at": int(time.time()) + 3600,
        }
        auth_response = {"email": "person@example.com"}
        with (
            patch.object(account_client, "_require_config", return_value=self.config),
            patch.object(
                account_client, "_authenticated_api",
                return_value=(auth_response, session),
            ) as api,
            patch.object(
                account_client, "_profile_upsert",
                side_effect=lambda _config, saved_session, name:
                    ({"display_name": name}, saved_session),
            ),
            patch.object(account_client, "_save_session", side_effect=lambda value: value),
        ):
            profile, saved_session, email_pending = account_client.save_profile_settings(
                session, "Person", "person_123", "new@example.com",
            )

        self.assertEqual(profile["display_name"], "Person")
        self.assertTrue(email_pending)
        self.assertEqual(saved_session["email"], "person@example.com")
        self.assertEqual(saved_session["username"], "person_123")
        request_body = api.call_args.args[4]
        self.assertEqual(request_body["email"], "new@example.com")
        self.assertEqual(request_body["data"]["username"], "person_123")

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
            if path == "/auth/v1/settings":
                return {"external": {"google": True}}
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

    def test_google_sign_in_reports_when_provider_is_disabled(self):
        with (
            patch.object(account_client, "_require_config", return_value=self.config),
            patch.object(
                account_client, "_api",
                return_value={"external": {"google": False}},
            ),
            patch.object(account_client.webbrowser, "open") as open_browser,
        ):
            with self.assertRaisesRegex(
                account_client.AccountError,
                "Enable Google in Authentication > Sign In / Providers",
            ):
                account_client.start_google_sign_in()

        open_browser.assert_not_called()

    @unittest.skipUnless(__import__("os").name == "nt", "Windows DPAPI is required")
    def test_windows_session_encryption_round_trip(self):
        contents = b'{"refresh_token":"sensitive"}'
        protected = account_client._protect_windows(contents)
        self.assertNotEqual(protected, contents)
        self.assertEqual(account_client._unprotect_windows(protected), contents)


if __name__ == "__main__":
    unittest.main()
