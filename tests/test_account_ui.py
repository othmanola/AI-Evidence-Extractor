import unittest
from unittest.mock import patch

import app
import account_client


class AccountUiTests(unittest.TestCase):
    def test_guest_startup_restores_profile_and_settings_state(self):
        profile = {"display_name": "Local Guest", "avatar_url": ""}
        with (
            patch.object(app._account_client, "load_session", return_value=None),
            patch.object(app._account_client, "guest_mode_selected", return_value=True),
            patch.object(app._account_client, "load_guest_profile", return_value=profile),
        ):
            result = app._account_initial_session()

        self.assertEqual(len(result), 6)
        self.assertTrue(result[3])
        self.assertTrue(result[4])
        self.assertEqual(result[5], "Local Guest")
        self.assertIn("Local Guest", result[2])

    def test_guest_settings_save_name_updates_local_profile_view(self):
        profile = {"display_name": "Local Guest", "avatar_url": ""}
        with patch.object(
            app._account_client, "save_guest_display_name", return_value=profile
        ) as save_name:
            result = app._account_save_name_ui("Local Guest", None)

        save_name.assert_called_once_with("Local Guest")
        self.assertIn("saved on this computer", result[0])
        self.assertIn("Local Guest", result[1])
        self.assertIsNone(result[2])
        self.assertEqual(result[3], "Local Guest")

    def test_guest_connection_save_keeps_profile_and_settings_visible(self):
        with (
            patch.object(app._account_client, "save_config", return_value=False),
            patch.object(app._account_client, "test_connection"),
            patch.object(app._account_client, "guest_mode_selected", return_value=True),
            patch.object(
                app._account_client,
                "load_guest_profile",
                return_value={"display_name": "Local Guest", "avatar_url": ""},
            ),
        ):
            result = app._account_save_config_ui(
                "https://example.supabase.co", "public-key", None, True
            )

        self.assertEqual(len(result), 10)
        self.assertTrue(result[5]["visible"])
        self.assertTrue(result[8]["visible"])
        self.assertFalse(result[9]["visible"])
        self.assertIn("Local Guest", result[1])
        self.assertEqual(result[3], "Local Guest")

    def test_guest_connection_error_keeps_profile_and_settings_visible(self):
        with patch.object(
            app._account_client,
            "save_config",
            side_effect=account_client.AccountError("Invalid settings."),
        ):
            result = app._account_save_config_ui("", "", None, True)

        self.assertEqual(len(result), 10)
        self.assertTrue(result[5]["visible"])
        self.assertTrue(result[8]["visible"])
        self.assertFalse(result[4]["visible"])


if __name__ == "__main__":
    unittest.main()
