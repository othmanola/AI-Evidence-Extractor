import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import zipfile

import build_update_package
import update_apply
import update_client


class UpdateMigrationTests(unittest.TestCase):
    def test_installer_includes_and_installs_signed_microsoft_runtime_prerequisite(self):
        root = Path(__file__).resolve().parents[1]
        installer = (root / "installer.iss").read_text(encoding="utf-8-sig")
        build_script = (root / "build_windows.ps1").read_text(encoding="utf-8-sig")
        workflow = (root / ".github" / "workflows" / "publish-update.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("PrivilegesRequired=admin", installer)
        self.assertIn('DestName: "vc_redist.x64.exe"; Flags: dontcopy', installer)
        self.assertIn("ExtractTemporaryFile('vc_redist.x64.exe')", installer)
        self.assertIn("IsVCRedistInstalled", installer)
        self.assertIn("RegQueryDWordValue(HKLM64", installer)
        self.assertIn("'/install /quiet /norestart'", installer)
        self.assertIn('DestName: "MicrosoftEdgeWebview2Setup.exe"; Flags: dontcopy', installer)
        self.assertIn("IsWebView2RuntimeInstalled", installer)
        self.assertIn("F3017226-FE2A-4295-8BDF-00C3A9A7E4C5", installer)
        self.assertIn("'/silent /install'", installer)
        self.assertIn("https://aka.ms/vs/17/release/vc_redist.x64.exe", build_script)
        self.assertIn("https://go.microsoft.com/fwlink/p/?LinkId=2124703", build_script)
        self.assertIn("Get-AuthenticodeSignature", build_script)
        self.assertIn("Microsoft Corporation", build_script)
        self.assertIn("IntToStr(ResultCode)", installer)
        self.assertIn("      - installer.iss", workflow)
        self.assertIn("      - build_windows.ps1", workflow)

    def test_account_module_is_allowed_but_not_required_for_compatibility_updates(self):
        self.assertIn("account_client.py", update_client.UPDATE_FILES)
        self.assertIn("account_client.py", update_client.OPTIONAL_UPDATE_FILES)
        self.assertIn("account_client.py", update_apply.ALLOWED)
        self.assertNotIn("account_client.py", update_apply.REQUIRED)
        self.assertIn("account_defaults.json", update_client.UPDATE_FILES)
        self.assertIn("account_defaults.json", update_client.OPTIONAL_UPDATE_FILES)
        self.assertIn("account_defaults.json", update_apply.ALLOWED)
        self.assertNotIn("account_defaults.json", update_apply.REQUIRED)

    def test_release_workflow_uses_bundled_supabase_defaults(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (
            root / ".github" / "workflows" / "publish-update.yml"
        ).read_text(encoding="utf-8")
        defaults = json.loads(
            (root / "account_defaults.json").read_text(encoding="utf-8")
        )
        build_script = (root / "build_windows.ps1").read_text(
            encoding="utf-8-sig"
        )

        self.assertIn("account_defaults.json", workflow)
        self.assertNotIn("vars.AEE_SUPABASE_URL", workflow)
        self.assertNotIn("vars.AEE_SUPABASE_PUBLISHABLE_KEY", workflow)
        self.assertIn("account_defaults.json", build_script)
        self.assertTrue(defaults["project_url"].startswith("https://"))
        self.assertTrue(defaults["anon_key"].startswith("sb_publishable_"))

    def test_update_package_includes_bundled_connection_defaults(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "source"
            output = root / "dist"
            defaults = root / "account_defaults.json"
            defaults.write_text(
                json.dumps({
                    "project_url": "https://example.supabase.co",
                    "anon_key": "public-test-key",
                }),
                encoding="utf-8",
            )
            for name in build_update_package.FILES:
                source = source_root / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(
                    "1.2.3" if name == "version.txt" else "test",
                    encoding="utf-8",
                )
            with (
                patch.object(build_update_package, "ROOT", source_root),
                patch.object(build_update_package, "DIST", output),
                patch.dict(os.environ, {
                    "AEE_ACCOUNT_DEFAULTS_PATH": str(defaults),
                }),
            ):
                build_update_package.main()

            with zipfile.ZipFile(output / "AI_Evidence_Extractor_Update.zip") as package:
                bundled = json.loads(package.read("account_defaults.json"))

        self.assertEqual(bundled["project_url"], "https://example.supabase.co")
        self.assertEqual(bundled["anon_key"], "public-test-key")

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

    def test_frozen_restart_uses_installed_executable_when_private_runtime_is_missing(self):
        with TemporaryDirectory() as temp:
            app_dir = Path(temp) / "app"
            app_dir.mkdir()
            executable = app_dir / "AI Evidence Extractor.exe"
            executable.touch()

            with patch("sys.frozen", True, create=True):
                command, cwd = update_client._restart_command(Path(temp) / "data", app_dir)

        self.assertEqual(command, [str(executable.resolve())])
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
