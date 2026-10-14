# Publishing app updates

Installed copies use a small HTTPS JSON manifest to check the current release. GitHub Releases can host each setup file, while `latest.json` in the repository's default branch points apps to the newest setup.

## Connect this build to your release feed

1. The public repository serves the manifest from its `main` branch. This build is configured to check:
   `https://raw.githubusercontent.com/othmanola/AI-Evidence-Extractor/main/latest.json`
2. Keep that URL in every installer you distribute. If the repository becomes private or its default branch changes, configure another public HTTPS manifest URL in `update_config.json` and rebuild.

## Publish a new version

1. Increase `version.txt` to the next three-part version, such as `1.0.5` (the prepared release in this build is `1.0.4`).
2. Build `dist/AI_Evidence_Extractor_Setup.exe` with `build_windows.ps1`.
3. Create a GitHub Release with tag `v1.0.5` and upload that setup file as a release asset. Use the same version in the tag and manifest.
4. Calculate the installer checksum in PowerShell:

   ```powershell
   (Get-FileHash .\dist\AI_Evidence_Extractor_Setup.exe -Algorithm SHA256).Hash.ToLower()
   ```

5. Copy `latest.example.json` to `latest.json`. Set `version`, `published_at`, `download_url` to the release asset's HTTPS URL, `sha256` to the checksum, and `release_notes` to a short description. The current `latest.json` is updated for each release and must be committed after its matching setup asset is uploaded.
6. Commit or upload `latest.json` to the repository's default branch. Existing apps read it from the stable raw URL and show the update on their next check.
7. Users of builds created before the Updates feature was added must install the first update-enabled release manually. Subsequent releases can be installed from the app's Updates page.

The updater verifies the HTTPS manifest, app identifier, version, HTTPS installer URL, and SHA-256 checksum before making an installer available to download. Downloading never launches the installer or changes the installed app. Users choose when and where to run the downloaded setup. Keep previous setup assets available in old GitHub Releases so users can still download them if needed.

## Current release values

The app version is stored in `version.txt` and is embedded into the setup version by `installer.iss`. `latest.json` is intentionally a release-time file: generate and publish it after the setup file has been built and its SHA-256 hash is known.
