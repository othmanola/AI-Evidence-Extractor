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

    def test_restart_uses_bundled_python_when_user_runtime_environment_is_missing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "app"
            data_dir = root / "data"
            runtime_dir = app_dir / "runtime"
            pythonw = runtime_dir / "pythonw.exe"
            launcher = app_dir / "desktop_launcher.py"
            executable = app_dir / "AI Evidence Extractor.exe"
            runtime_dir.mkdir(parents=True)
            for path in (pythonw, launcher, executable):
                path.touch()

            command, cwd = update_client._restart_command(data_dir, app_dir)

        self.assertEqual(command, [str(pythonw.resolve()), str(launcher.resolve())])
        self.assertEqual(cwd, app_dir)

    def test_installer_packages_loose_sources_for_internal_updates(self):
        spec = (Path(__file__).resolve().parents[1] / "AI Evidence Extractor.spec").read_text(
            encoding="utf-8"
        )
        entry = (Path(__file__).resolve().parents[1] / "app_entry.py").read_text(
            encoding="utf-8"
        )
        installer = (Path(__file__).resolve().parents[1] / "installer.iss").read_text(
            encoding="utf-8-sig"
        )
        desktop_entry = (Path(__file__).resolve().parents[1] / "desktop_entry.py").read_text(
            encoding="utf-8"
        )

        for module in ("app.py", "desktop_entry.py", "desktop_launcher.py",
                       "update_client.py", "update_apply.py"):
            with self.subTest(module=module):
                self.assertIn(f'"{module}"', spec)
        self.assertIn("from desktop_entry import main as launch_source_app", entry)
        self.assertIn(
            'Source: "dist\\AI Evidence Extractor\\*"; DestDir: "{app}"; '
            "Flags: ignoreversion recursesubdirs createallsubdirs",
            installer,
        )
        self.assertIn("return _launch_installed_sources(app_dir, data_dir)", desktop_entry)


if __name__ == "__main__":
    unittest.main()
