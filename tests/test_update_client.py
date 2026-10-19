import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import update_client


class UpdateClientTests(unittest.TestCase):
    def test_installed_version_marker_takes_precedence_over_frozen_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "installed"
            bundle_dir = root / "bundle"
            app_dir.mkdir()
            bundle_dir.mkdir()
            (app_dir / "version.txt").write_text("1.2.53", encoding="utf-8")
            (bundle_dir / "version.txt").write_text("1.2.45", encoding="utf-8")

            resource_dir = update_client._resource_dir(app_dir, bundle_dir, False)
            self.assertEqual(resource_dir, app_dir)
            self.assertEqual(
                (resource_dir / "version.txt").read_text(encoding="utf-8"), "1.2.53"
            )

    def test_frozen_bundle_version_is_used_before_any_update_is_installed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app_dir = root / "installed"
            bundle_dir = root / "bundle"
            app_dir.mkdir()
            bundle_dir.mkdir()
            (bundle_dir / "version.txt").write_text("1.2.45", encoding="utf-8")

            resource_dir = update_client._resource_dir(app_dir, bundle_dir, False)

        self.assertEqual(resource_dir, bundle_dir)

    def test_check_for_updates_uses_fresh_manifest_request(self):
        manifest = {
            "app_id": update_client.APP_ID,
            "version": "1.2.13",
            "package_url": "https://example.com/update.zip",
            "package_sha256": "a" * 64,
        }

        class Response(io.BytesIO):
            headers = {}

            def geturl(self):
                return "https://example.com/latest.json"

        response = Response(json.dumps(manifest).encode("utf-8"))
        with (
            patch.object(
                update_client, "_manifest_url",
                return_value="https://example.com/latest.json",
            ),
            patch.object(update_client, "current_version", return_value="1.2.12"),
            patch.object(
                update_client.urllib.request, "urlopen", return_value=response
            ) as urlopen,
        ):
            result = update_client.check_for_updates()

        request = urlopen.call_args.args[0]
        self.assertIn("aee_check=", request.full_url)
        self.assertEqual(request.get_header("Cache-control"), "no-cache, no-store, max-age=0")
        self.assertEqual(request.get_header("Pragma"), "no-cache")
        self.assertEqual(result["version"], "1.2.13")
        self.assertTrue(result["available"])

    def test_frozen_build_requires_in_app_update_package(self):
        manifest = {
            "app_id": update_client.APP_ID,
            "version": "1.2.13",
            "download_url": "https://example.com/setup.exe",
            "sha256": "b" * 64,
        }

        class Response(io.BytesIO):
            def geturl(self):
                return "https://example.com/latest.json"

        response = Response(json.dumps(manifest).encode("utf-8"))
        with (
            patch.object(update_client, "IS_FROZEN", True),
            patch.object(
                update_client, "_manifest_url",
                return_value="https://example.com/latest.json",
            ),
            patch.object(update_client, "current_version", return_value="1.2.12"),
            patch.object(update_client.urllib.request, "urlopen", return_value=response),
        ):
            with self.assertRaisesRegex(update_client.UpdateError, "in-app update package"):
                update_client.check_for_updates()

    def test_frozen_build_uses_package_even_when_setup_asset_is_present(self):
        manifest = {
            "app_id": update_client.APP_ID,
            "version": "1.2.13",
            "download_url": "https://example.com/setup.exe",
            "sha256": "b" * 64,
            "package_url": "https://example.com/update.zip",
            "package_sha256": "a" * 64,
        }

        class Response(io.BytesIO):
            def geturl(self):
                return "https://example.com/latest.json"

        response = Response(json.dumps(manifest).encode("utf-8"))
        with (
            patch.object(update_client, "IS_FROZEN", True),
            patch.object(
                update_client, "_manifest_url",
                return_value="https://example.com/latest.json",
            ),
            patch.object(update_client, "current_version", return_value="1.2.12"),
            patch.object(update_client.urllib.request, "urlopen", return_value=response),
        ):
            result = update_client.check_for_updates()

        self.assertTrue(result["available"])
        self.assertEqual(result["package_url"], "https://example.com/update.zip")

    def test_new_builds_use_latest_release_asset_as_update_feed(self):
        config = json.loads(update_client.CONFIG_FILE.read_text(encoding="utf-8"))
        self.assertEqual(
            config["manifest_url"],
            "https://github.com/othmanola/AI-Evidence-Extractor/releases/latest/download/latest.json",
        )


if __name__ == "__main__":
    unittest.main()
