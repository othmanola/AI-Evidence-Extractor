import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import update_apply
import update_client


class UpdateMigrationTests(unittest.TestCase):
    def test_account_module_is_allowed_but_not_required_for_compatibility_updates(self):
        self.assertIn("account_client.py", update_client.UPDATE_FILES)
        self.assertIn("account_client.py", update_client.OPTIONAL_UPDATE_FILES)
        self.assertIn("account_client.py", update_apply.ALLOWED)
        self.assertNotIn("account_client.py", update_apply.REQUIRED)

    def test_restart_prefers_python_launcher_over_self_extracting_executable(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            data_dir = root / "data"
            runtime_dir = app_dir / "runtime"
            pythonw = data_dir / "runtimeenv" / "Scripts" / "pythonw.exe"
            launcher = app_dir / "desktop_launcher.py"
            executable = app_dir / "AI Evidence Extractor.exe"
            for path in (runtime_dir, pythonw.parent):
                path.mkdir(parents=True, exist_ok=True)
            for path in (pythonw, launcher, executable):
                path.touch()

            command, cwd = update_client._restart_command(data_dir, app_dir)

        self.assertEqual(command, [str(pythonw.resolve()), str(launcher.resolve())])
        self.assertEqual(cwd, runtime_dir)


if __name__ == "__main__":
    unittest.main()
