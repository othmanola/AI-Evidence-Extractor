import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import app
import account_client


def _nested_code_constants(code):
    for value in code.co_consts:
        yield value
        if hasattr(value, "co_consts"):
            yield from _nested_code_constants(value)


class AccountUiTests(unittest.TestCase):
    def test_gradio_loading_overlays_use_accessible_motion_aware_styles(self):
        self.assertIn(".gradio-container .wrap.generating", app.CSS)
        self.assertIn(".wrap.generating .progress-bar-wrap", app.CSS)
        self.assertIn("@media(prefers-reduced-motion:reduce)", app.CSS)
        self.assertIn(".wrap.generating::before{animation:none;}", app.CSS)

    def test_updates_badge_uses_installed_version_over_stale_updater_version(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "version.txt").write_text("1.2.56", encoding="utf-8")
            with (
                patch.dict("os.environ", {"AEE_APP_DIR": temp}),
                patch.object(
                    app._update_client, "current_version", return_value="1.2.45"
                ),
            ):
                badge = app._updates_intro_html()

        self.assertIn("Installed version&nbsp; v1.2.56", badge)
        self.assertNotIn("Installed version&nbsp; v1.2.45", badge)

    def test_video_prefetch_keeps_frame_order_and_decode_timestamps(self):
        class Capture:
            def __init__(self):
                self.frames = [("first", 40.0), ("second", 80.0)]
                self.last_position = 0.0
                self.released = False

            def read(self):
                if self.frames:
                    frame, self.last_position = self.frames.pop(0)
                    return True, frame
                return False, None

            def get(self, _property):
                return self.last_position

            def release(self):
                self.released = True

        capture = Capture()
        frames = list(app.Processor._prefetch_frames(capture, queue_size=2))

        self.assertEqual(
            [(index, frame, timestamp) for index, ok, frame, timestamp in frames if ok],
            [(0, "first", 40.0), (1, "second", 80.0)],
        )
        self.assertTrue(capture.released)
        self.assertEqual([index for index, _, _, _ in frames], list(range(len(frames))))

    def test_ensemble_panel_reports_active_inference_device_and_precision(self):
        panel = app._ensemble_panel_html({
            "members": [], "device": "CPU", "precision": "FP32",
        })

        self.assertIn("Inference device: CPU · FP32", panel)

    def test_update_progress_bar_supports_unknown_download_size(self):
        progress = app._update_progress_html(
            0, "Downloading update package (size unknown)", indeterminate=True
        )

        self.assertIn("aee-update-progress-indeterminate", progress)
        self.assertIn('aria-valuenow="0"', progress)
        self.assertIn("Downloading update package (size unknown)", progress)

    def test_update_download_and_install_keep_completed_progress_visible(self):
        staged = {"app_id": app._update_client.APP_ID, "version": "1.2.51"}
        download_steps = [
            (0.5, "Downloading update package", None, None, False),
            (1, "Update downloaded", staged, None, True),
        ]
        with patch.object(app, "_run_update_task", return_value=iter(download_steps)):
            download_result = list(app._download_update_ui({"available": True}))
        self.assertIn("Download complete and verified", download_result[-1][0])
        self.assertIn('aria-valuenow="100"', download_result[-1][0])

        install_steps = [
            (0.5, "Installing update: app.py", None, None, False),
            (1, "Update installed", None, None, True),
        ]
        with patch.object(app, "_run_update_task", return_value=iter(install_steps)):
            install_result = list(app._apply_update_ui(staged))
        self.assertIn("Installation complete", install_result[-1][0])
        self.assertIn('aria-valuenow="100"', install_result[-1][0])

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
        self.assertNotIn('data-aee-nav-tab="History"', navigation)
        self.assertIn('data-aee-nav-tab="Detailed History"', navigation)
        self.assertNotIn('data-aee-nav-tab="Settings"', navigation)
        self.assertIn('<span>Updates</span>', navigation)
        self.assertIn('<span>About Us</span>', navigation)
        self.assertIn('<span>Contact Us</span>', navigation)
        ui_constants = tuple(_nested_code_constants(app.build_ui.__code__))
        self.assertTrue(any(
            isinstance(value, str) and "Recent analyzed videos" in value
            for value in ui_constants
        ))
        self.assertIn("Download all history ZIP", ui_constants)
        self.assertIn("Download ZIP", ui_constants)
        self.assertIn("Details", ui_constants)
        self.assertIn("Remove", ui_constants)
        self.assertIn("Clear all history", ui_constants)
        self.assertIn("#dc2626", app.CSS)
        self.assertIn("#aee-history-remove button", app.CSS)
        self.assertIn("#aee-history-clear button", app.CSS)
        self.assertIn("#aee-history-details button", app.CSS)
        self.assertIn("#aee-history-download button", app.CSS)
        self.assertIn("#aee-history-refresh button", app.CSS)
        self.assertIn("#aee-history-download-all button", app.CSS)
        self.assertIn("linear-gradient(135deg,#ef4444,#b91c1c)", app.CSS)
        self.assertIn("justify-content:flex-end!important", app.CSS)
        self.assertIn("min-height:27px!important", app.CSS)

    def test_detailed_history_opens_as_a_fullscreen_utility_window(self):
        constants = tuple(_nested_code_constants(app.build_ui.__code__))
        self.assertIn("History", constants)
        self.assertIn("aee-detailed-history-window", constants)
        self.assertIn("Detailed History", app._JS_HEAD)
        self.assertIn("activateAeeUtilityWindow(windowName)", app._JS_HEAD)
        self.assertIn(".aee-history-window .aee-history-page", app.CSS)
        self.assertIn(".aee-history-window .aee-history-toolbar", app.CSS)
        self.assertIn(".aee-history-row-actions", app.CSS)
        self.assertNotIn('with gr.Tab("Detailed History")', app.build_ui.__code__.co_consts)

    def test_history_list_is_newest_first_with_row_actions(self):
        entries = [
            {
                "id": "older",
                "analyzed_at": "2025-01-01T10:00:00+00:00",
                "video": "older.mp4",
                "detections": 2,
            },
            {
                "id": "newer",
                "analyzed_at": "2025-01-02T10:00:00+00:00",
                "video": "newer.mp4",
                "detections": 5,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / "analysis_history.json"
            history_file.write_text(json.dumps(entries), encoding="utf-8")
            with patch.object(app, "HISTORY_FILE", history_file):
                items = app._history_list_items()

        self.assertEqual([item["id"] for item in items], ["newer", "older"])
        self.assertEqual(items[0]["video"], "newer.mp4")
        self.assertEqual(items[0]["detections"], 5)

    def test_remove_history_entry_deletes_only_managed_per_video_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history_file = root / "analysis_history.json"
            history_dir = root / "outputs" / "history"
            run_dir = history_dir / "run-1"
            first_video_dir = run_dir / "001-first"
            second_video_dir = run_dir / "002-second"
            first_capture = first_video_dir / "captures" / "capture.jpg"
            second_capture = second_video_dir / "captures" / "capture.jpg"
            first_capture.parent.mkdir(parents=True)
            second_capture.parent.mkdir(parents=True)
            first_capture.write_bytes(b"first")
            second_capture.write_bytes(b"second")
            shared_report = run_dir / "reports" / "report.pdf"
            shared_report.parent.mkdir()
            shared_report.write_bytes(b"report")
            source_video = root / "source.mp4"
            source_video.write_bytes(b"original")
            outside_artifact = root / "outside.jpg"
            outside_artifact.write_bytes(b"outside")
            entries = [
                {
                    "id": "first",
                    "source_path": str(source_video),
                    "details": {
                        "captures": [str(first_capture)],
                        "report_pdf": str(shared_report),
                    },
                },
                {
                    "id": "second",
                    "details": {
                        "captures": [str(second_capture)],
                        "heatmaps": [str(outside_artifact)],
                    },
                },
            ]
            history_file.write_text(json.dumps(entries), encoding="utf-8")
            with (
                patch.object(app, "HISTORY_FILE", history_file),
                patch.object(app, "HISTORY_DIR", history_dir),
                patch.object(app, "OUTPUT_DIR", root / "outputs"),
            ):
                remaining_count, cleanup_warnings = app._remove_history_entry("first")

            saved = json.loads(history_file.read_text(encoding="utf-8"))
            self.assertEqual(remaining_count, 1)
            self.assertEqual(cleanup_warnings, [])
            self.assertEqual([entry["id"] for entry in saved], ["second"])
            self.assertFalse(first_video_dir.exists())
            self.assertTrue(second_capture.is_file())
            self.assertTrue(shared_report.is_file())
            self.assertTrue(source_video.is_file())
            self.assertTrue(outside_artifact.is_file())
            with (
                patch.object(app, "HISTORY_FILE", history_file),
                patch.object(app, "HISTORY_DIR", history_dir),
                patch.object(app, "OUTPUT_DIR", root / "outputs"),
            ):
                remaining_count, cleanup_warnings = app._remove_history_entry("second")
            self.assertEqual(remaining_count, 0)
            self.assertEqual(cleanup_warnings, [])
            self.assertFalse(run_dir.exists())
            self.assertTrue(source_video.is_file())
            self.assertTrue(outside_artifact.is_file())

    def test_clear_analysis_history_removes_managed_files_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history_file = root / "analysis_history.json"
            history_dir = root / "outputs" / "history"
            capture = history_dir / "run-1" / "001-video" / "captures" / "capture.jpg"
            capture.parent.mkdir(parents=True)
            capture.write_bytes(b"capture")
            source_video = root / "source.mp4"
            source_video.write_bytes(b"original")
            export_dir = root / "outputs" / "history_exports"
            export_dir.mkdir(parents=True)
            export = export_dir / "saved-id.zip"
            export.write_bytes(b"zip")
            history_file.write_text(json.dumps([{
                "id": "saved-id",
                "source_path": str(source_video),
                "details": {"captures": [str(capture)]},
            }]), encoding="utf-8")

            with (
                patch.object(app, "HISTORY_FILE", history_file),
                patch.object(app, "HISTORY_DIR", history_dir),
                patch.object(app, "OUTPUT_DIR", root / "outputs"),
            ):
                removed_count, cleanup_warnings = app._clear_analysis_history()

            self.assertEqual(removed_count, 1)
            self.assertEqual(cleanup_warnings, [])
            self.assertEqual(json.loads(history_file.read_text(encoding="utf-8")), [])
            self.assertFalse(capture.parent.parent.parent.exists())
            self.assertFalse(export.exists())
            self.assertTrue(source_video.is_file())

    def test_utility_windows_have_distinct_descriptive_headers(self):
        constants = tuple(_nested_code_constants(app.build_ui.__code__))
        header_markup = "\n".join(
            value for value in constants
            if isinstance(value, str) and "aee-project-page-heading" in value
        )

        for page_title, eyebrow, subtitle in (
            ("Account", "ACCOUNT CENTER", "Manage your profile, sign-in, and privacy."),
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

    def test_google_signed_in_profile_has_account_identity_and_provider_badge(self):
        session = {
            "auth_provider": "google",
            "display_name": "Alex User",
            "email": "alex@example.com",
            "created_at": "2024-03-12T00:00:00Z",
        }
        profile_html = app._account_profile_html(session)

        self.assertIn('data-provider="google"', profile_html)
        self.assertIn("SIGNED IN WITH GOOGLE", profile_html)
        self.assertIn("<svg viewBox=\"0 0 48 48\"", profile_html)
        self.assertIn('data-account-type="online"', profile_html)
        self.assertIn("Account profile", profile_html)
        self.assertIn("alex@example.com", profile_html)
        self.assertIn("Member since Mar 2024", profile_html)
        self.assertIn(".aee-account-shell .aee-account-profile:before{content:", app.CSS)
        self.assertIn(".aee-account-shell .aee-account-profile{", app.CSS)
        self.assertIn("linear-gradient(125deg,#182238", app.CSS)
        self.assertIn("html[data-theme=\"light\"] .aee-account-shell .aee-account-profile{", app.CSS)

    def test_guest_profile_header_explains_local_storage(self):
        profile_html = app._account_profile_html(
            None, {"display_name": "Local Guest"}
        )

        self.assertIn('data-account-type="guest"', profile_html)
        self.assertIn("Local profile", profile_html)
        self.assertIn("Your profile stays on this device", profile_html)
        self.assertIn("GUEST · LOCAL ONLY", profile_html)

    def test_account_profile_header_escapes_identity_content(self):
        profile_html = app._account_profile_html(
            {"display_name": "<Alex>", "email": "alex&example.com"}
        )

        self.assertIn("&lt;Alex&gt;", profile_html)
        self.assertIn("alex&amp;example.com", profile_html)
        self.assertNotIn("<Alex>", profile_html)

    def test_settings_page_has_navigable_language_and_security_sections(self):
        navigation = app._settings_navigation_html()
        self.assertIn('<span>Language</span>', navigation)
        self.assertIn('<option value="English">English</option>', navigation)
        self.assertIn('<option value="Arabic">Arabic</option>', navigation)
        self.assertIn('data-aee-settings-section="security"', navigation)
        constants = tuple(_nested_code_constants(app.build_ui.__code__))
        self.assertIn("aee-settings-profile-section", constants)
        self.assertIn("aee-settings-security-section", constants)
        self.assertIn('data-aee-settings-section', app._JS_HEAD)
        self.assertIn("section.scrollIntoView", app._JS_HEAD)
        self.assertIn("window.aeeToggleTheme = toggleTheme", app._JS_HEAD)
        self.assertIn("Discard your unsaved profile changes?", app._JS_HEAD)
        self.assertIn("applyAeeLanguage", app._JS_HEAD)
        self.assertIn("AEE_ARABIC_TRANSLATIONS", app._JS_HEAD)
        self.assertIn("'Manage your profile, sign-in, and privacy.'", app._JS_HEAD)
        self.assertIn("'أدر ملفك الشخصي وتسجيل الدخول وإعدادات الخصوصية.'", app._JS_HEAD)
        self.assertIn("'Personal information':'المعلومات الشخصية'", app._JS_HEAD)
        self.assertIn("trimmed.split('AI Evidence Extractor').join(brandMarker)", app._JS_HEAD)
        self.assertIn("value.trim().replace(/\\s+/g, ' ')", app._JS_HEAD)
        self.assertIn("aeeTranslateSubtree(document.documentElement, arabic)", app._JS_HEAD)
        self.assertIn("new MutationObserver(records =>", app._JS_HEAD)
        self.assertIn("aee-language", app._JS_HEAD)
        self.assertIn(
            'html[dir="rtl"] .gradio-container [dir="ltr"]{direction:rtl!important;}',
            app.CSS,
        )
        self.assertIn('html[dir="rtl"] .aee-nav-drawer{right:0;left:auto;', app.CSS)
        self.assertIn(
            'html[dir="rtl"] .aee-account-shell .aee-settings-photo-actions{right:auto!important;left:14px!important;}',
            app.CSS,
        )
        self.assertIn(
            'html[dir="rtl"] .aee-account-shell .aee-settings-photo-actions{\n'
            '    right:auto!important;left:9px!important;}',
            app.CSS,
        )
        self.assertIn('html[dir="rtl"] .gradio-container{', app.CSS)
        self.assertIn('font-family:Tahoma,Arial,sans-serif!important;', app.CSS)
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

    def test_history_zip_contains_analysis_details_and_generated_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "outputs"
            artifact_dir = output_dir / "history" / "analysis-1"
            artifact_dir.mkdir(parents=True)
            capture = artifact_dir / "capture.jpg"
            capture.write_bytes(b"capture-data")
            source_video = root / "original.mp4"
            source_video.write_bytes(b"source-video")
            history_file = root / "analysis_history.json"
            history_file.write_text(json.dumps([
                {
                    "id": "analysis-1",
                    "analyzed_at": "2026-10-17T12:00:00+00:00",
                    "video": "sample.mp4",
                    "source_path": str(source_video),
                    "duration_seconds": 8.0,
                    "detections": 1,
                    "details": {
                        "captures": [str(capture)],
                        "detections": [{"label": "Firearm", "timestamp": 2.0}],
                        "tracks": [{"Track": 3}],
                        "case_notes": "Review notes",
                    },
                },
                {
                    "id": "analysis-2",
                    "analyzed_at": "2026-10-16T12:00:00+00:00",
                    "video": "second.mp4",
                    "duration_seconds": 4.0,
                    "detections": 0,
                    "details": {"detections": [], "tracks": []},
                },
            ]), encoding="utf-8")
            with (
                patch.object(app, "OUTPUT_DIR", output_dir),
                patch.object(app, "HISTORY_FILE", history_file),
            ):
                selected_zip, selected_count, selected_missing = app._history_zip("analysis-1")
                all_zip, all_count, all_missing = app._history_zip()
                self.assertTrue(Path(all_zip).is_file())

            with zipfile.ZipFile(selected_zip) as archive:
                names = archive.namelist()
                analysis_path = next(name for name in names if name.endswith("/analysis.json"))
                analysis = json.loads(archive.read(analysis_path))

        self.assertEqual(selected_count, 1)
        self.assertEqual(all_count, 2)
        self.assertEqual(selected_missing, 0)
        self.assertEqual(all_missing, 0)
        self.assertTrue(any(name.endswith("/captures/0001-capture.jpg") for name in names))
        self.assertEqual(analysis["details"]["detections"][0]["label"], "Firearm")
        self.assertEqual(analysis["details"]["tracks"][0]["Track"], 3)
        self.assertEqual(analysis["details"]["case_notes"], "Review notes")
        self.assertNotIn(str(source_video), names)
        self.assertNotIn("original.mp4", names)

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
                self.assertTrue(Path(saved_entries[0]["details"]["report_pdf"]).is_file())

        self.assertEqual(warnings, [])
        self.assertEqual(saved_entries[0]["source_path"], str(source_video))
        self.assertEqual(saved_entries[0]["details"]["case_notes"], "Case notes")
        self.assertEqual(len(saved_entries[0]["details"]["captures"]), 1)
        self.assertEqual(len(saved_entries[0]["details"]["highlights"]), 1)
        self.assertEqual(len(saved_entries[0]["details"]["heatmaps"]), 2)
        self.assertIn("Case notes", detail[0])
        self.assertEqual(len(detail[1]), 3)
        self.assertEqual(len(detail[2]), 5)
        self.assertEqual(detail[3].iloc[0]["label"], "Firearm")
        self.assertEqual(detail[4].iloc[0]["Track"], 4)

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
        self.assertIn(
            "aee-account-profile-source",
            tuple(_nested_code_constants(app.build_ui.__code__)),
        )

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
            ".aee-settings-photo-actions{flex:0 0 88px!important;width:88px!important;",
            app.CSS,
        )
        self.assertIn("Change photo", tuple(_nested_code_constants(app.build_ui.__code__)))
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
        self.assertTrue(any(
            isinstance(value, tuple) and "elem_id" in value
            for value in _nested_code_constants(app.build_ui.__code__)
        ))
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
            result = list(app._check_updates_ui())[-1]

        self.assertIn('data-update-available="true"', result[0])
        self.assertTrue(result[1]["visible"])

        current = {"available": False, "installed_version": "1.2.12"}
        with patch.object(app._update_client, "check_for_updates", return_value=current):
            result = list(app._check_updates_ui())[-1]

        self.assertIn('data-update-available="false"', result[0])
        self.assertFalse(result[1]["visible"])

    def test_frozen_update_check_requires_in_app_package(self):
        available = {
            "available": True,
            "version": "1.2.13",
            "download_url": "https://example.com/setup.exe",
        }
        with (
            patch.object(app._update_client, "IS_FROZEN", True),
            patch.object(app._update_client, "check_for_updates", return_value=available),
        ):
            result = list(app._check_updates_ui())[-1]

        self.assertFalse(result[1]["visible"])
        self.assertIn("in-app update package", result[0])

    def test_frozen_update_downloads_and_applies_package_in_app(self):
        update = {
            "app_id": app._update_client.APP_ID,
            "version": "1.2.13",
            "available": True,
            "package_url": "https://example.com/update.zip",
        }
        staged = {
            "app_id": app._update_client.APP_ID,
            "version": "1.2.13",
            "stage_dir": "staged",
            "staged": True,
        }
        with (
            patch.object(app._update_client, "IS_FROZEN", True),
            patch.object(
                app._update_client, "download_and_stage_update", return_value=staged
            ) as download,
            patch.object(app._update_client, "apply_staged_update") as apply,
            patch.object(app.subprocess, "Popen") as launch_installer,
        ):
            downloaded = list(app._download_update_ui(update))[-1]
            applied = list(app._apply_update_ui(downloaded[2]))[-1]

        download.assert_called_once_with(update, unittest.mock.ANY)
        apply.assert_called_once_with(staged, unittest.mock.ANY)
        launch_installer.assert_not_called()
        self.assertTrue(downloaded[1]["visible"])
        self.assertIn("in-app installation progress", downloaded[0])
        self.assertIn("installed successfully", applied[0])
        self.assertTrue(applied[2]["visible"])

    def test_update_progress_is_shown_in_the_gui(self):
        ui_strings = "\n".join(
            value for value in _nested_code_constants(app.build_ui.__code__)
            if isinstance(value, str)
        )
        self.assertIn("Download Update", ui_strings)
        self.assertIn("Install Update", ui_strings)
        self.assertIn("Restart App", ui_strings)
        self.assertIn("progress", app._updates_intro_html().lower())
        progress_html = app._update_progress_html(0.42, "Installing update: app.py")
        self.assertIn('aria-valuenow="42"', progress_html)
        self.assertIn("width:42%", progress_html)
        self.assertIn("Installing update: app.py", progress_html)

    def test_update_check_streams_progress_before_returning_result(self):
        available = {
            "available": True,
            "version": "1.2.13",
            "package_url": "https://example.com/update.zip",
        }

        def check(progress):
            progress(0.25, desc="Connecting to the update service")
            progress(0.8, desc="Reading release information")
            return available

        with patch.object(app._update_client, "check_for_updates", side_effect=check):
            updates = list(app._check_updates_ui())

        self.assertGreaterEqual(len(updates), 4)
        streamed_status = [result[0] for result in updates[:-1]]
        self.assertTrue(any("aria-valuenow=" in status for status in streamed_status))
        self.assertIn('data-update-available="true"', updates[-1][0])

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
