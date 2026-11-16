import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import desktop_entry
import desktop_launcher


class DesktopEntryTests(unittest.TestCase):
    def test_startup_launches_the_app_with_cuda_python_after_gpu_setup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            data_dir = root / "data"
            runtime_dir = app_dir / "runtime"
            cuda_venv = data_dir / "runtimeenv_cuda2"
            cuda_pythonw = cuda_venv / "Scripts" / "pythonw.exe"
            launcher = app_dir / "desktop_launcher.py"
            runtime_dir.mkdir(parents=True)
            launcher.touch()
            cuda_pythonw.parent.mkdir(parents=True)
            cuda_pythonw.touch()
            (cuda_venv / ".aee_cuda_ready").touch()

            with (
                patch.object(desktop_entry.sys, "platform", "win32"),
                patch.object(desktop_entry, "_prepare_cuda_runtime", return_value=True),
                patch.object(
                    desktop_entry.subprocess, "Popen",
                    return_value=type("Process", (), {"pid": 42})(),
                ) as popen,
                patch.dict(os.environ, {
                    "AEE_APP_DIR": str(app_dir),
                    "AEE_DATA_DIR": str(data_dir),
                }),
            ):
                result = desktop_entry.main()

        self.assertEqual(result, 0)
        self.assertEqual(popen.call_args.args[0][0], str(cuda_pythonw.resolve()))
        self.assertEqual(
            popen.call_args.kwargs["env"]["AEE_DATA_DIR"], str(data_dir)
        )

    def test_prepares_cuda_runtime_before_app_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            data_dir = root / "data"
            bootstrap_python = data_dir / "runtimeenv" / "Scripts" / "python.exe"
            cuda_venv = data_dir / "runtimeenv_cuda2"
            cuda_pythonw = cuda_venv / "Scripts" / "pythonw.exe"
            bootstrap_python.parent.mkdir(parents=True)
            bootstrap_python.touch()

            def setup(python, requested_app_dir, requested_data_dir):
                self.assertEqual(python, bootstrap_python)
                self.assertEqual(requested_app_dir, app_dir)
                self.assertEqual(requested_data_dir, data_dir)
                cuda_pythonw.parent.mkdir(parents=True)
                cuda_pythonw.touch()
                (cuda_venv / ".aee_cuda_ready").touch()
                return 0, ["AEE_PROGRESS:GPU acceleration is ready"]

            with (
                patch.object(desktop_entry.sys, "platform", "win32"),
                patch.object(desktop_entry.shutil, "which", return_value="nvidia-smi"),
                patch.object(desktop_entry, "_run_gpu_setup", side_effect=setup),
                patch.dict(os.environ, {"AEE_GPU_AUTOCONNECT_ERROR": "stale error"}),
            ):
                ready = desktop_entry._prepare_cuda_runtime(app_dir, data_dir)

            self.assertTrue(ready)
            self.assertNotIn("AEE_GPU_AUTOCONNECT_ERROR", os.environ)
            self.assertIn(
                "GPU acceleration is ready",
                (data_dir / "startup.log").read_text(encoding="utf-8"),
            )

    def test_failed_startup_gpu_setup_keeps_cpu_fallback_and_records_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            data_dir = root / "data"
            bootstrap_python = data_dir / "runtimeenv" / "Scripts" / "python.exe"
            bootstrap_python.parent.mkdir(parents=True)
            bootstrap_python.touch()
            with (
                patch.object(desktop_entry.sys, "platform", "win32"),
                patch.object(desktop_entry.shutil, "which", return_value="nvidia-smi"),
                patch.object(
                    desktop_entry, "_run_gpu_setup",
                    return_value=(1, ["CUDA installation failed"]),
                ),
                patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("AEE_GPU_AUTOCONNECT_ERROR", None)
                ready = desktop_entry._prepare_cuda_runtime(app_dir, data_dir)
                setup_error = os.environ.get("AEE_GPU_AUTOCONNECT_ERROR")

            self.assertFalse(ready)
            self.assertEqual(setup_error, "CUDA installation failed")
            self.assertIn(
                "CUDA installation failed",
                (data_dir / "startup.log").read_text(encoding="utf-8"),
            )

    def test_desktop_launcher_forces_installed_source_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            app_dir = Path(temp)
            app_path = str(app_dir.resolve())
            original_app_dir = os.environ.get("AEE_APP_DIR")
            original_source_mode = os.environ.get("AEE_SOURCE_MODE")
            original_sys_path = list(sys.path)
            try:
                os.environ.pop("AEE_APP_DIR", None)
                os.environ.pop("AEE_SOURCE_MODE", None)
                desktop_launcher._configure_installed_source_mode(app_dir)

                self.assertEqual(os.environ["AEE_APP_DIR"], app_path)
                self.assertEqual(os.environ["AEE_SOURCE_MODE"], "1")
                self.assertEqual(sys.path[0], app_path)
            finally:
                sys.path[:] = original_sys_path
                if original_app_dir is None:
                    os.environ.pop("AEE_APP_DIR", None)
                else:
                    os.environ["AEE_APP_DIR"] = original_app_dir
                if original_source_mode is None:
                    os.environ.pop("AEE_SOURCE_MODE", None)
                else:
                    os.environ["AEE_SOURCE_MODE"] = original_source_mode

    def test_frozen_fallback_runs_loose_modules_inside_packaged_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            app_dir = Path(temp)
            data_dir = app_dir / "data"
            (app_dir / "account_client.py").write_text(
                "SOURCE_MODULE = 'account'\n", encoding="utf-8")
            (app_dir / "update_apply.py").write_text(
                "SOURCE_MODULE = 'apply'\n", encoding="utf-8")
            (app_dir / "update_client.py").write_text(
                "import os\n"
                "from pathlib import Path\n"
                "RESOURCE_DIR = Path(os.environ['AEE_APP_DIR'])\n",
                encoding="utf-8")
            (app_dir / "app.py").write_text(
                "import account_client\n"
                "import update_client\n"
                "SOURCE_MODULES = (account_client.SOURCE_MODULE, "
                "update_client.RESOURCE_DIR.name)\n",
                encoding="utf-8")
            (app_dir / "desktop_launcher.py").write_text(
                "import app\n"
                "def main():\n"
                "    assert app.SOURCE_MODULES[0] == 'account'\n"
                "    return 7\n",
                encoding="utf-8")
            previous_app_dir = os.environ.get("AEE_APP_DIR")
            os.environ["AEE_APP_DIR"] = str(app_dir)
            try:
                with patch.dict(os.environ, {"AEE_SOURCE_MODE": "original"}, clear=False):
                    result = desktop_entry._launch_installed_sources(app_dir, data_dir)
                    self.assertEqual(result, 7)
                    self.assertEqual(os.environ["AEE_SOURCE_MODE"], "original")
            finally:
                if previous_app_dir is None:
                    os.environ.pop("AEE_APP_DIR", None)
                else:
                    os.environ["AEE_APP_DIR"] = previous_app_dir


if __name__ == "__main__":
    unittest.main()
