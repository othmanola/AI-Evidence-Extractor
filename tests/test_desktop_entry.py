import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import desktop_entry
import desktop_launcher


class DesktopEntryTests(unittest.TestCase):
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
