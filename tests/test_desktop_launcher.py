import base64
import os
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.error import URLError

import desktop_launcher


class DesktopLauncherTests(unittest.TestCase):
    def test_auto_connects_and_restarts_in_cuda_runtime_when_nvidia_is_available(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            data_dir = root / "data"
            launcher = app_dir / "desktop_launcher.py"
            launcher.parent.mkdir(parents=True)
            launcher.touch()
            cuda_pythonw = (
                data_dir / "runtimeenv_cuda2" / "Scripts" / "pythonw.exe"
            )
            window = MagicMock()

            def install(progress):
                progress(0.5, "Installing CUDA runtime")
                cuda_pythonw.parent.mkdir(parents=True)
                cuda_pythonw.touch()
                (cuda_pythonw.parents[1] / ".aee_cuda_ready").write_text(
                    '{"gpu":"NVIDIA GPU"}', encoding="utf-8"
                )
                return {"gpu": "NVIDIA GPU", "torch": "2.7.1+cu128"}

            application = SimpleNamespace(
                _resolve_inference_device=lambda: ("cpu", False),
                _nvidia_gpu_name=lambda: "NVIDIA GPU",
                _install_realtime_gpu=Mock(side_effect=install),
            )
            with (
                patch.object(desktop_launcher.sys, "platform", "win32"),
                patch.object(desktop_launcher, "APP_DIR", app_dir),
                patch.object(desktop_launcher, "APP_DATA", data_dir),
                patch.object(desktop_launcher.subprocess, "Popen") as popen,
                patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("AEE_GPU_AUTOCONNECT_ERROR", None)
                connected = desktop_launcher._auto_connect_gpu(
                    application, window
                )

            self.assertTrue(connected)
            application._install_realtime_gpu.assert_called_once()
            self.assertIn("GPU verified", window.evaluate_js.call_args.args[0])
            popen.assert_called_once()
            self.assertEqual(popen.call_args.args[0][0], str(cuda_pythonw.resolve()))
            self.assertEqual(
                popen.call_args.kwargs["env"]["YOLO_AUTOINSTALL"], "false"
            )
            window.destroy.assert_called_once()

    def test_auto_connect_falls_back_to_cpu_if_gpu_setup_fails(self):
        application = SimpleNamespace(
            _resolve_inference_device=lambda: ("cpu", False),
            _nvidia_gpu_name=lambda: "NVIDIA GPU",
            _install_realtime_gpu=MagicMock(
                side_effect=RuntimeError("CUDA package installation failed")
            ),
        )
        with (
            patch.object(desktop_launcher.sys, "platform", "win32"),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("AEE_GPU_AUTOCONNECT_ERROR", None)
            connected = desktop_launcher._auto_connect_gpu(
                application, MagicMock()
            )
            setup_error = os.environ.get("AEE_GPU_AUTOCONNECT_ERROR")

        self.assertFalse(connected)
        self.assertEqual(setup_error, "CUDA package installation failed")

    def test_cpu_launcher_restarts_in_a_prepared_cuda_runtime(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            data_dir = root / "data"
            launcher = app_dir / "desktop_launcher.py"
            launcher.parent.mkdir(parents=True)
            launcher.touch()
            cuda_pythonw = (
                data_dir / "runtimeenv_cuda2" / "Scripts" / "pythonw.exe"
            )
            cuda_pythonw.parent.mkdir(parents=True)
            cuda_pythonw.touch()
            (cuda_pythonw.parents[1] / ".aee_cuda_ready").touch()
            with (
                patch.object(desktop_launcher.sys, "platform", "win32"),
                patch.object(desktop_launcher, "APP_DIR", app_dir),
                patch.object(desktop_launcher, "APP_DATA", data_dir),
                patch.object(desktop_launcher.subprocess, "Popen") as popen,
            ):
                relaunched = desktop_launcher._launch_prepared_cuda_runtime()

        self.assertTrue(relaunched)
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0][0], str(cuda_pythonw.resolve()))

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
