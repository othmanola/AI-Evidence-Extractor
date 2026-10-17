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

    def test_project_navigation_exposes_settings_instead_of_account(self):
        navigation = app._side_navigation_html()
        self.assertIn('data-aee-nav-tab="Settings"', navigation)
        self.assertNotIn('data-aee-nav-tab="Account"', navigation)
        self.assertIn('<span>Updates</span>', navigation)
        self.assertIn('<span>About Us</span>', navigation)
        self.assertIn('<span>Contact Us</span>', navigation)

    def test_update_app_shortcut_opens_updates_and_profile_toolbar_opens_settings(self):
        navigation = app._side_navigation_html()
        hero = app._hero_html({"n_models": 0})

        self.assertIn('data-aee-nav-update', navigation)
        self.assertIn('<span>Update the app</span>', navigation)
        self.assertIn('data-aee-nav-update', app._JS_HEAD)
        self.assertIn("activateAeeUtilityWindow('updates')", app._JS_HEAD)
        self.assertIn('data-aee-open-settings', hero)
        self.assertIn('data-aee-profile-avatar', hero)
        self.assertIn("activateAeeUtilityWindow('account')", app._JS_HEAD)
        self.assertIn("#aee-account-profile-source .aee-account-profile", app._JS_HEAD)
        self.assertIn("aee-account-profile-source", app.build_ui.__code__.co_consts)

    def test_topbar_layout_adapts_to_mobile_viewports(self):
        self.assertIn("function syncAeeTopbarLayout()", app._JS_HEAD)
        self.assertIn("window.innerWidth <= 640", app._JS_HEAD)
        self.assertIn("window.innerWidth <= 380", app._JS_HEAD)
        self.assertIn("update.style.left = mobile ? '60px' : ''", app._JS_HEAD)
        self.assertIn(".aee-toolbar{position:fixed;top:12px;right:12px;gap:6px;}", app.CSS)

    def test_google_sign_in_can_start_from_settings(self):
        with patch.object(
            app._account_client, "start_google_sign_in", return_value="oauth-123"
        ):
            result = app._account_google_start_ui()

        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], "oauth-123")
        self.assertFalse(result[2]["interactive"])
        self.assertFalse(result[3]["interactive"])
        self.assertTrue(result[4]["active"])

    def test_google_sign_in_updates_guest_settings_actions(self):
        signed_in = {
            "user_id": "user-123",
            "email": "person@example.com",
            "display_name": "Person",
        }
        with (
            patch.object(
                app._account_client, "finish_google_sign_in",
                return_value=signed_in,
            ),
            patch.object(
                app, "_account_render_signed_in",
                return_value=("<section>Person</section>", signed_in, None),
            ),
        ):
            result = app._account_google_poll_ui("oauth-123", None, True)

        self.assertEqual(len(result), 14)
        self.assertTrue(result[6]["visible"])
        self.assertTrue(result[12]["visible"])
        self.assertFalse(result[13]["visible"])

    def test_guest_settings_show_google_sign_in_and_hide_log_out(self):
        profile = {"display_name": "Local Guest", "avatar_url": ""}
        with (
            patch.object(
                app._account_client, "sign_out",
                return_value="You are using guest mode.",
            ),
            patch.object(
                app._account_client, "load_guest_profile",
                return_value=profile,
            ),
        ):
            result = app._account_guest_ui({"user_id": "user-123"})

        self.assertEqual(len(result), 11)
        self.assertFalse(result[4]["visible"])
        self.assertTrue(result[5]["visible"])
        self.assertTrue(result[8]["visible"])
        self.assertFalse(result[9]["visible"])
        self.assertTrue(result[10]["visible"])

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

        self.assertEqual(len(result), 11)
        self.assertTrue(result[5]["visible"])
        self.assertTrue(result[8]["visible"])
        self.assertFalse(result[9]["visible"])
        self.assertTrue(result[10]["visible"])
        self.assertIn("Local Guest", result[1])
        self.assertEqual(result[3], "Local Guest")

    def test_guest_connection_error_keeps_profile_and_settings_visible(self):
        with patch.object(
            app._account_client,
            "save_config",
            side_effect=account_client.AccountError("Invalid settings."),
        ):
            result = app._account_save_config_ui("", "", None, True)

        self.assertEqual(len(result), 11)
        self.assertTrue(result[5]["visible"])
        self.assertTrue(result[8]["visible"])
        self.assertFalse(result[4]["visible"])


if __name__ == "__main__":
    unittest.main()
