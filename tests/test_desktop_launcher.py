import base64
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import desktop_launcher


class DesktopLauncherTests(unittest.TestCase):
    def test_migrates_legacy_shortcuts_to_the_installed_source_launcher(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            data_dir = root / "data"
            pythonw = data_dir / "runtimeenv" / "Scripts" / "pythonw.exe"
            for path in (
                app_dir / "AI Evidence Extractor.exe",
                app_dir / "desktop_launcher.py",
                pythonw,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with (
                patch.object(desktop_launcher.sys, "platform", "win32"),
                patch("sys.frozen", True, create=True),
                patch.dict("os.environ", {"AEE_DATA_DIR": str(data_dir)}),
                patch.object(desktop_launcher, "APP_DATA", data_dir),
                patch.object(
                    desktop_launcher.subprocess, "run",
                    return_value=MagicMock(stdout=""),
                ) as run,
            ):
                desktop_launcher._migrate_legacy_shortcuts(app_dir)

            arguments = run.call_args.args[0]
            script = base64.b64decode(arguments[-1]).decode("utf-16le")
            self.assertIn("AI Evidence Extractor.exe", run.call_args.kwargs["env"][
                "AEE_SHORTCUT_OLD_TARGET"
            ])
            self.assertEqual(
                run.call_args.kwargs["env"]["AEE_SHORTCUT_NEW_TARGET"],
                str(pythonw),
            )
            self.assertIn("$shortcut.Arguments = '\"' + $launcher + '\"'", script)
            self.assertTrue((data_dir / "source_launcher_shortcuts_v1").is_file())

    def test_shortcut_migration_is_not_repeated_after_success(self):
        with TemporaryDirectory() as temp:
            app_dir = Path(temp) / "app"
            data_dir = Path(temp) / "data"
            app_dir.mkdir()
            data_dir.mkdir()
            (data_dir / "source_launcher_shortcuts_v1").touch()

            with (
                patch.object(desktop_launcher.sys, "platform", "win32"),
                patch.dict("os.environ", {"AEE_DATA_DIR": str(data_dir)}),
                patch.object(desktop_launcher, "APP_DATA", data_dir),
                patch.object(desktop_launcher.subprocess, "run") as run,
            ):
                desktop_launcher._migrate_legacy_shortcuts(app_dir)

        run.assert_not_called()

    def test_shortcut_migration_failure_is_logged_and_can_be_retried(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            data_dir = root / "data"
            pythonw = data_dir / "runtimeenv" / "Scripts" / "pythonw.exe"
            pythonw.parent.mkdir(parents=True)
            pythonw.touch()

            with (
                patch.object(desktop_launcher.sys, "platform", "win32"),
                patch("sys.frozen", True, create=True),
                patch.dict("os.environ", {"AEE_DATA_DIR": str(data_dir)}),
                patch.object(desktop_launcher, "APP_DATA", data_dir),
                patch.object(
                    desktop_launcher.subprocess, "run",
                    side_effect=desktop_launcher.subprocess.CalledProcessError(
                        1, ["powershell.exe"], stderr="shortcut access denied"
                    ),
                ),
                self.assertLogs("aee.desktop", level="WARNING") as logs,
            ):
                desktop_launcher._migrate_legacy_shortcuts(app_dir)

            self.assertIn("shortcut access denied", logs.output[0])
            self.assertFalse((data_dir / "source_launcher_shortcuts_v1").exists())

    def test_server_readiness_succeeds_if_launch_thread_exits_after_http_ready(self):
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
        response = MagicMock()
        response.__enter__.return_value.status = 200

        with patch.object(
            desktop_launcher.urllib.request, "urlopen", return_value=response
        ):
            desktop_launcher.wait_for_server("http://127.0.0.1/", thread, [])

    def test_server_readiness_fails_if_thread_stops_before_http_ready(self):
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()

        with (
            patch.object(
                desktop_launcher.urllib.request, "urlopen",
                side_effect=URLError("connection refused"),
            ),
            self.assertRaisesRegex(RuntimeError, "service stopped during startup"),
        ):
            desktop_launcher.wait_for_server("http://127.0.0.1/", thread, [])


if __name__ == "__main__":
    unittest.main()
