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

        self.assertEqual(len(result), 7)
        self.assertTrue(result[3])
        self.assertTrue(result[4])
        self.assertEqual(result[5], "Local Guest")
        self.assertIn("Local Guest", result[2])
        self.assertEqual(result[6]["display_name"], "Local Guest")
        self.assertEqual(set(result[6]), {"display_name", "username", "email"})

    def test_project_navigation_exposes_settings_instead_of_account(self):
        navigation = app._side_navigation_html()
        self.assertIn('data-aee-nav-tab="Settings"', navigation)
        self.assertNotIn('data-aee-nav-tab="Account"', navigation)
        self.assertIn('<span>Updates</span>', navigation)
        self.assertIn('<span>About Us</span>', navigation)
        self.assertIn('<span>Contact Us</span>', navigation)

    def test_settings_page_has_navigable_language_and_security_sections(self):
        navigation = app._settings_navigation_html()
        self.assertIn('<span>Language</span>', navigation)
        self.assertIn('<option value="English">English</option>', navigation)
        self.assertIn('<option value="Arabic">Arabic</option>', navigation)
        self.assertIn('data-aee-settings-section="security"', navigation)

        constants = app.build_ui.__code__.co_consts
        self.assertIn("aee-settings-profile-section", constants)
        self.assertIn("aee-settings-security-section", constants)
        self.assertIn('data-aee-settings-section', app._JS_HEAD)
        self.assertIn("section.scrollIntoView", app._JS_HEAD)
        self.assertIn("window.aeeToggleTheme = toggleTheme", app._JS_HEAD)
        self.assertIn("Discard your unsaved profile changes?", app._JS_HEAD)
        self.assertIn("applyAeeLanguage", app._JS_HEAD)
        self.assertIn("aee-language", app._JS_HEAD)
        self.assertIn('#aee-settings-profile-section[data-aee-dirty="true"]', app._JS_HEAD)
        self.assertIn("data-aee-modified=\"true\"", app._JS_HEAD)
        self.assertIn("Save changes", constants)
        self.assertNotIn("Country", constants)
        self.assertNotIn("Dial code", constants)
        self.assertNotIn("Phone number", constants)
        self.assertNotIn("Time zone", constants)
        self.assertIn("Change photo", constants)
        self.assertTrue(any(
            isinstance(constant, str) and "Personal information" in constant
            for constant in constants
        ))

    def test_supabase_setup_sql_is_safe_to_rerun(self):
        sql = app._ACCOUNT_SETUP_SQL
        policies = (
            "Users can read their own profile",
            "Users can create their own profile",
            "Users can update their own profile",
            "Users can read their own profile photos",
            "Users can upload their own profile photos",
            "Users can update their own profile photos",
        )
        for policy in policies:
            with self.subTest(policy=policy):
                self.assertLess(
                    sql.index(f'drop policy if exists "{policy}"'),
                    sql.index(f'create policy "{policy}"'),
                )

    def test_update_app_shortcut_opens_updates_and_profile_toolbar_opens_settings(self):
        navigation = app._side_navigation_html()
        hero = app._hero_html({"n_models": 0})

        self.assertNotIn('data-aee-nav-update', navigation)
        self.assertIn('data-aee-nav-update', hero)
        self.assertIn('<span>Update the app</span>', hero)
        self.assertLess(hero.index('data-aee-nav-update'), hero.index('id="theme-toggle"'))
        self.assertIn('data-aee-nav-update', hero.split('id="theme-toggle"')[0])
        self.assertIn('hidden>', hero)
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
        self.assertIn("function syncAeeUpdateShortcut()", app._JS_HEAD)
        self.assertIn("status.dataset.updateAvailable !== 'true'", app._JS_HEAD)
        self.assertIn(".aee-toolbar{position:fixed;top:12px;right:12px;gap:6px;}", app.CSS)
        self.assertIn(".aee-toolbar .aee-update-app{width:auto!important;min-height:32px!important;", app.CSS)

    def test_profile_dashboard_uses_compact_responsive_photo_picker_and_cards(self):
        self.assertIn(".aee-account-shell .aee-account-profile-form{", app.CSS)
        self.assertIn(
            ".aee-settings-photo-actions{flex:0 0 96px!important;width:96px!important;",
            app.CSS,
        )
        self.assertIn("Change photo", app.build_ui.__code__.co_consts)
        self.assertIn(".aee-settings-card>.styler{border-radius:inherit!important;}", app.CSS)
        self.assertIn("html[data-theme=\"light\"] .aee-account-shell .aee-utility-window-header", app.CSS)

    def test_update_shortcut_visibility_tracks_update_check(self):
        available = {
            "available": True,
            "version": "1.2.13",
            "published_at": "2026-10-18",
            "release_notes": "A fix",
            "package_url": "https://example.com/update.zip",
        }
        with patch.object(app._update_client, "check_for_updates", return_value=available):
            result = app._check_updates_ui()

        self.assertIn('data-update-available="true"', result[0])
        self.assertTrue(result[1]["visible"])

        current = {"available": False, "installed_version": "1.2.12"}
        with patch.object(app._update_client, "check_for_updates", return_value=current):
            result = app._check_updates_ui()

        self.assertIn('data-update-available="false"', result[0])
        self.assertFalse(result[1]["visible"])

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

    def test_guest_profile_settings_save_does_not_require_email(self):
        profile = {
            "display_name": "Local Guest",
            "username": "local_guest",
            "contact_email": "guest@example.com",
            "country": "Canada",
            "phone_country_code": "+1",
            "phone_number": "4165551234",
            "time_zone": "America/Toronto",
            "avatar_url": "",
        }
        with patch.object(
            app._account_client, "save_guest_profile_settings", return_value=profile
        ) as save_settings:
            result = app._account_save_profile_settings_ui(
                "Local Guest", "@local_guest", "", None,
            )

        save_settings.assert_called_once_with(
            "Local Guest", "@local_guest", "",
        )
        self.assertIn("saved on this computer", result[0])
        self.assertIn("Local Guest", result[1])
        self.assertIsNone(result[2])

    def test_profile_photo_picker_previews_selected_image(self):
        editor, panel = app._account_preview_photo_ui("profile.png")
        self.assertEqual(editor, {"value": "profile.png", "__type__": "update"})
        self.assertEqual(panel, {"visible": True, "__type__": "update"})

        editor, panel = app._account_preview_photo_ui(None)
        self.assertEqual(editor, {"value": None, "__type__": "update"})
        self.assertEqual(panel, {"visible": False, "__type__": "update"})

    def test_profile_photo_apply_uploads_edited_composite_and_closes_editor(self):
        with patch.object(
            app, "_account_upload_photo_ui",
            return_value=(
                app._account_status_html("Profile photo updated.", "success"),
                "<section>Updated profile</section>",
                None,
            ),
        ) as apply_photo:
            result = app._account_apply_photo_ui(
                {"composite": "cropped-profile.webp"}, None
            )

        apply_photo.assert_called_once_with("cropped-profile.webp", None)
        self.assertIn("Profile photo updated", result[0])
        self.assertEqual(result[1], "<section>Updated profile</section>")
        self.assertEqual(result[3], {"value": None, "__type__": "update"})
        self.assertEqual(result[4], {"visible": False, "__type__": "update"})

    def test_profile_photo_apply_keeps_editor_open_when_upload_fails(self):
        status = app._account_status_html("Photo upload failed.", "error")
        with patch.object(
            app, "_account_upload_photo_ui",
            return_value=(status, "<section>Profile</section>", None),
        ):
            result = app._account_apply_photo_ui(
                {"composite": "cropped-profile.webp"}, None
            )

        self.assertIn("Photo upload failed", result[0])
        self.assertEqual(result[3], {"__type__": "update", "skip": True})
        self.assertEqual(result[4], {"visible": True, "__type__": "update"})

    def test_signed_in_profile_email_change_is_reported_as_pending(self):
        session = {"user_id": "user-123", "email": "person@example.com"}
        with (
            patch.object(
                app._account_client, "save_profile_settings",
                return_value=({"display_name": "Person"}, session, True),
            ),
            patch.object(
                app, "_account_render_signed_in",
                return_value=("<section>Person</section>", session, None),
            ),
        ):
            result = app._account_save_profile_settings_ui(
                "Person", "person_123", "new@example.com", session,
            )

        self.assertIn("Confirm the email change", result[0])
        self.assertIn("new@example.com", result[0])
        self.assertIn("data-profile-saved=\"true\"", result[0])

    def test_guest_profile_email_field_is_hidden(self):
        self.assertEqual(app._account_profile_email_visibility_ui(None),
                         {"visible": False, "__type__": "update"})
        self.assertEqual(
            app._account_profile_email_visibility_ui({"user_id": "user-123"}),
            {"visible": True, "__type__": "update"},
        )

    def test_guest_connection_save_keeps_profile_and_settings_visible(self):
        calls = []

        def test_connection(project_url, api_key):
            calls.append(("test", project_url, api_key))

        def save_config(project_url, api_key):
            calls.append(("save", project_url, api_key))
            return False

        with (
            patch.object(
                app._account_client, "save_config", side_effect=save_config
            ),
            patch.object(
                app._account_client, "test_connection", side_effect=test_connection
            ),
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
        self.assertIn("Supabase Auth connection verified", result[0])
        self.assertIn("Run the setup SQL below", result[0])
        self.assertEqual(calls, [
            ("test", "https://example.supabase.co", "public-key"),
            ("save", "https://example.supabase.co", "public-key"),
        ])

    def test_guest_connection_error_keeps_profile_and_settings_visible(self):
        session = {"user_id": "existing-user"}
        with (
            patch.object(
                app._account_client, "test_connection",
                side_effect=account_client.AccountError("Invalid settings."),
            ),
            patch.object(app._account_client, "save_config") as save_config,
        ):
            result = app._account_save_config_ui("", "", None, True)

        self.assertEqual(len(result), 11)
        self.assertTrue(result[5]["visible"])
        self.assertTrue(result[8]["visible"])
        self.assertFalse(result[4]["visible"])
        self.assertIn("Invalid settings", result[0])
        save_config.assert_not_called()

    def test_failed_connection_check_does_not_clear_active_session(self):
        session = {"user_id": "existing-user"}
        with (
            patch.object(
                app._account_client, "test_connection",
                side_effect=account_client.AccountError("Network unavailable."),
            ),
            patch.object(app._account_client, "save_config") as save_config,
        ):
            result = app._account_save_config_ui(
                "https://example.supabase.co", "public-key", session, True
            )

        self.assertIs(result[2], session)
        save_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
