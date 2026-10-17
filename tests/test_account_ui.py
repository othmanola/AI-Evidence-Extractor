import base64
import json
from pathlib import Path
import tempfile
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

    def test_project_navigation_exposes_account_and_history(self):
        navigation = app._side_navigation_html()
        self.assertIn('data-aee-nav-tab="Account"', navigation)
        self.assertIn('data-aee-nav-tab="History"', navigation)
        self.assertNotIn('data-aee-nav-tab="Settings"', navigation)
        self.assertIn('<span>Updates</span>', navigation)
        self.assertIn('<span>About Us</span>', navigation)
        self.assertIn('<span>Contact Us</span>', navigation)

    def test_utility_windows_have_distinct_descriptive_headers(self):
        constants = app.build_ui.__code__.co_consts
        header_markup = "\n".join(
            value for value in constants
            if isinstance(value, str) and "aee-project-page-heading" in value
        )

        for page_title, eyebrow, subtitle in (
            ("Settings", "YOUR WORKSPACE", "Manage your profile, preferences, and security."),
            ("Updates", "APP CENTER", "Check for the latest improvements and install updates."),
            ("About Us", "THE PROJECT", "Learn about the workspace and how it supports evidence review."),
            ("Contact Us", "SUPPORT", "Get help, share feedback, or report an issue."),
        ):
            with self.subTest(page_title=page_title):
                self.assertIn(f"<h1>{page_title}</h1>", header_markup)
                self.assertIn(eyebrow, header_markup)
                self.assertIn(subtitle, header_markup)

        self.assertIn(".aee-utility-window-header{position:sticky;top:0;z-index:12;", app.CSS)
        self.assertIn("min-height:86px;", app.CSS)
        self.assertIn(".aee-utility-window-shell .aee-project-page-heading p{", app.CSS)
        self.assertIn("html[data-theme=\"light\"] .aee-utility-window-shell .aee-project-page-heading h1", app.CSS)

    def test_google_signed_in_profile_has_google_badge_and_solid_background(self):
        session = {
            "auth_provider": "google",
            "display_name": "Alex User",
            "email": "alex@example.com",
        }
        profile_html = app._account_profile_html(session)

        self.assertIn('data-provider="google"', profile_html)
        self.assertIn("SIGNED IN WITH GOOGLE", profile_html)
        self.assertIn("<svg viewBox=\"0 0 48 48\"", profile_html)
        self.assertIn(".aee-account-shell .aee-account-profile:before{display:none;}", app.CSS)
        self.assertIn(".aee-account-shell .aee-account-profile{", app.CSS)
        self.assertIn("background:#172033;", app.CSS)
        self.assertIn("html[data-theme=\"light\"] .aee-account-shell .aee-account-profile{background:#f1f5f9;}", app.CSS)

    def test_settings_page_has_navigable_language_and_security_sections(self):
        navigation = app._settings_navigation_html()
        self.assertIn('<span>Language</span>', navigation)
        self.assertIn('<option value="English">English</option>', navigation)
        self.assertIn('<option value="Arabic">Arabic</option>', navigation)
        self.assertIn('data-aee-settings-section="security"', navigation)

    def test_history_retains_more_than_one_hundred_analyses(self):
        entries = [
            {
                "id": f"history-{index}",
                "analyzed_at": "2025-01-01T12:00:00+00:00",
                "video": f"video-{index}.mp4",
                "duration_seconds": 12.5,
                "detections": index,
                "high_accuracy": False,
                "case": "case-1",
            }
            for index in range(101)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / "analysis_history.json"
            history_file.write_text(json.dumps(entries), encoding="utf-8")
            with patch.object(app, "HISTORY_FILE", history_file):
                history = app._history_dataframe()
                choices, selected = app._history_choices()

        self.assertEqual(len(history), 101)
        self.assertEqual(len(choices), 101)
        self.assertEqual(selected, "history-100")
        self.assertIn(".aee-history-detail", app.CSS)

    def test_analysis_history_copies_generated_artifacts_and_details(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_video = temp / "original.mp4"
            source_video.write_bytes(b"source")
            capture = temp / "capture.jpg"
            capture.write_bytes(b"capture")
            summary = temp / "summary.mp4"
            summary.write_bytes(b"summary")
            heat = temp / "heat.jpg"
            heat.write_bytes(b"heat")
            blend = temp / "blend.jpg"
            blend.write_bytes(b"blend")
            master = temp / "master.mp4"
            master.write_bytes(b"master")
            report = temp / "report.pdf"
            report.write_bytes(b"report")
            work_root = temp / "work"
            highlights = work_root / "sample" / "highlights"
            highlights.mkdir(parents=True)
            highlight = highlights / "highlight.mp4"
            highlight.write_bytes(b"highlight")

            result = app.Result(
                video_name="sample.mp4",
                source_path=str(source_video),
                metadata=app.Meta(fps=25.0, frame_count=250, width=640,
                                  height=480, duration=10.0),
                detections=[{"timestamp": 1.0, "label": "Firearm",
                             "confidence": 0.9}],
                captures=[str(capture)],
                object_summaries={"Firearm": str(summary)},
                summary_stats={"Firearm": {"clips": 1, "duration": 3.0,
                                           "avg_confidence": 0.9}},
                notes="Analysis notes",
            )
            history_file = temp / "analysis_history.json"
            history_dir = temp / "history"
            with (
                patch.object(app, "HISTORY_FILE", history_file),
                patch.object(app, "HISTORY_DIR", history_dir),
            ):
                warnings = app._record_analysis_history(
                    [result], str(master), str(report), "case-1",
                    case_meta={"examiner": "Analyst"}, case_notes="Case notes",
                    tracker_rows=[{"Video": "sample.mp4", "Track": 4}],
                    cam_items=[{"video": "sample.mp4", "heat": str(heat),
                                "blend": str(blend)}],
                    work_root=work_root,
                )
                saved_entries = json.loads(history_file.read_text(encoding="utf-8"))
                detail = app._history_detail_ui(saved_entries[0]["id"])

        self.assertEqual(warnings, [])
        self.assertEqual(saved_entries[0]["source_path"], str(source_video))
        self.assertEqual(saved_entries[0]["details"]["case_notes"], "Case notes")
        self.assertEqual(len(saved_entries[0]["details"]["captures"]), 1)
        self.assertEqual(len(saved_entries[0]["details"]["highlights"]), 1)
        self.assertEqual(len(saved_entries[0]["details"]["heatmaps"]), 2)
        self.assertTrue(Path(saved_entries[0]["details"]["report_pdf"]).is_file())
        self.assertIn("Case notes", detail[0])
        self.assertEqual(len(detail[1]), 3)
        self.assertEqual(len(detail[2]), 5)
        self.assertEqual(detail[3].iloc[0]["label"], "Firearm")
        self.assertEqual(detail[4].iloc[0]["Track"], 4)

        constants = app.build_ui.__code__.co_consts
        self.assertIn("aee-settings-profile-section", constants)
        self.assertIn("aee-settings-security-section", constants)
        self.assertIn('data-aee-settings-section', app._JS_HEAD)
        self.assertIn("section.scrollIntoView", app._JS_HEAD)
        self.assertIn("window.aeeToggleTheme = toggleTheme", app._JS_HEAD)
        self.assertIn("Discard your unsaved profile changes?", app._JS_HEAD)
        self.assertIn("applyAeeLanguage", app._JS_HEAD)
        self.assertIn("aee-language", app._JS_HEAD)
        self.assertIn(
            '<details class="aee-settings-language-control">',
            navigation,
        )
        self.assertIn(
            ".aee-settings-language-control>summary{min-height:42px;",
            app.CSS,
        )
        self.assertIn(
            ".aee-settings-language-menu{position:absolute;z-index:20;",
            app.CSS,
        )
        self.assertIn(".aee-settings-language-native{display:none!important;}", app.CSS)
        self.assertIn(".aee-settings-nav{display:flex;flex-wrap:wrap;gap:6px;padding:8px;overflow:visible;}", app.CSS)
        self.assertIn("data-aee-language-option=\"English\"", navigation)
        self.assertIn("data-aee-language-option=\"Arabic\"", navigation)
        self.assertIn("languageDropdown.open = false", app._JS_HEAD)
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
        self.assertIn(".aee-utility-window-header{position:sticky;top:0;z-index:12;", app.CSS)

    def test_profile_fields_are_ordered_and_logout_uses_destructive_gradient(self):
        constants = app.build_ui.__code__.co_consts
        field_positions = [
            next(index for index, value in enumerate(constants)
                 if value == label)
            for label in ("Display name", "Username", "Email")
        ]

        self.assertEqual(field_positions, sorted(field_positions))
        self.assertIn(
            ".aee-settings-fields{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr));",
            app.CSS,
        )
        self.assertIn(
            "#aee-account-signout,#aee-account-signout button,",
            app.CSS,
        )
        self.assertIn("background:linear-gradient(115deg,#ef5350,#d9364f)!important;", app.CSS)
        self.assertIn("aee-account-signout", constants)
        self.assertIn('elem_id="aee-account-signout"', constants)
        self.assertIn(".aee-settings-fields{grid-template-columns:repeat(2,minmax(0,1fr));}", app.CSS)
        self.assertIn(".aee-settings-fields{grid-template-columns:1fr;gap:8px!important;}", app.CSS)

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

    def test_google_sign_in_and_connection_setup_follow_saved_config(self):
        with patch.object(app._account_client, "config_is_ready", return_value=True):
            self.assertEqual(
                app._account_connection_setup_visibility_ui(),
                {"visible": False, "__type__": "update"},
            )
            self.assertEqual(
                app._account_google_sign_in_visibility_ui(),
                {"visible": True, "__type__": "update"},
            )

        with patch.object(app._account_client, "config_is_ready", return_value=False):
            self.assertEqual(
                app._account_connection_setup_visibility_ui(),
                {"visible": True, "__type__": "update"},
            )
            self.assertEqual(
                app._account_google_sign_in_visibility_ui(),
                {"visible": False, "__type__": "update"},
            )

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
            patch.object(
                app._account_client, "config_is_ready", return_value=True,
            ),
        ):
            result = app._account_guest_ui({"user_id": "user-123"})

        self.assertEqual(len(result), 11)
        self.assertFalse(result[4]["visible"])
        self.assertTrue(result[5]["visible"])
        self.assertTrue(result[8]["visible"])
        self.assertFalse(result[9]["visible"])
        self.assertTrue(result[10]["visible"])

    def test_guest_settings_hide_google_sign_in_without_project_connection(self):
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
            patch.object(app._account_client, "config_is_ready", return_value=False),
        ):
            result = app._account_guest_ui({"user_id": "user-123"})

        self.assertFalse(result[10]["visible"])

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

    def test_profile_photo_editor_has_square_crop_and_zoom_controls(self):
        editor = app._account_photo_editor_html()
        self.assertIn('aria-modal="true"', editor)
        self.assertIn('class="aee-photo-editor-window"', editor)
        self.assertIn('id="aee-photo-crop-canvas"', editor)
        self.assertIn('type="range" min="1" max="3"', editor)
        self.assertIn('data-aee-photo-apply', editor)
        self.assertIn('data-aee-photo-cancel', editor)
        self.assertIn("function initializeAeePhotoEditor()", app._JS_HEAD)
        self.assertIn("function componentControl(id, selector)", app._JS_HEAD)
        self.assertIn("component.matches(selector)", app._JS_HEAD)
        self.assertIn("component.querySelector(selector)", app._JS_HEAD)
        self.assertIn("drawImage(image, sourceX, sourceY, sourceSize", app._JS_HEAD)
        self.assertIn("data-aee-photo-zoom", app._JS_HEAD)
        self.assertIn(
            ".aee-photo-editor-modal{position:fixed;inset:0;z-index:10000;display:grid;",
            app.CSS,
        )
        self.assertIn(
            ".aee-photo-editor-apply{min-width:82px;height:38px;",
            app.CSS,
        )
        self.assertIn("submit.click();\n          closeEditor();", app._JS_HEAD)

    def test_profile_photo_apply_uploads_validated_cropped_jpeg(self):
        image_bytes = b"\xff\xd8\xffcropped photo bytes"
        image_data = "data:image/jpeg;base64," + base64.b64encode(
            image_bytes
        ).decode("ascii")
        uploaded_path = []

        def upload(image_path, session):
            uploaded_path.append(image_path)
            self.assertEqual(Path(image_path).read_bytes(), image_bytes)
            return (
                app._account_status_html(
                    "Profile photo updated.", "success", profile_saved=True,
                ),
                "<section>Updated profile</section>",
                session,
            )

        with patch.object(
            app, "_account_upload_photo_ui", side_effect=upload,
        ):
            result = app._account_apply_photo_data_ui(image_data, None)

        self.assertIn("Profile photo updated", result[0])
        self.assertEqual(result[1], "<section>Updated profile</section>")
        self.assertIn('data-photo-applied="true"', result[3]["value"])
        self.assertFalse(Path(uploaded_path[0]).exists())

    def test_profile_photo_apply_rejects_invalid_image_payload(self):
        image_data = "data:image/jpeg;base64," + base64.b64encode(
            b"not a jpeg"
        ).decode("ascii")
        with patch.object(
            app, "_account_upload_photo_ui",
        ) as apply_photo:
            result = app._account_apply_photo_data_ui(image_data, None)

        apply_photo.assert_not_called()
        self.assertIn('data-kind="error"', result[0])
        self.assertIn('data-photo-applied="false"', result[3]["value"])

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

        self.assertEqual(len(result), 12)
        self.assertTrue(result[5]["visible"])
        self.assertTrue(result[8]["visible"])
        self.assertFalse(result[9]["visible"])
        self.assertTrue(result[10]["visible"])
        self.assertIn("Local Guest", result[1])
        self.assertEqual(result[3], "Local Guest")
        self.assertIn("Supabase Auth connection verified", result[0])
        self.assertIn("Profile and photo features require", result[0])
        self.assertTrue(result[11]["visible"])
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

        self.assertEqual(len(result), 12)
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
