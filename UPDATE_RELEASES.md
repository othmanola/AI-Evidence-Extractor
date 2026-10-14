# Publishing app updates

Installed copies use a small HTTPS JSON manifest to check the current release. GitHub Releases host both the full setup installer and a source update archive. The app downloads the archive and installs it in-place when the user selects Install Update; a progress indicator reports each file. Restart the app afterward to load the new version.

## Connect this build to your release feed

1. The public repository serves the manifest from its `main` branch. This build is configured to check:
   `https://raw.githubusercontent.com/othmanola/AI-Evidence-Extractor/main/latest.json`
2. Keep that URL in every installer you distribute. If the repository becomes private or its default branch changes, configure another public HTTPS manifest URL in `update_config.json` and rebuild.

## Publish a new version

1. Increase `version.txt` to the next three-part version (the prepared release in this build is `1.0.9`).
2. Build `dist/AI_Evidence_Extractor_Setup.exe` and `dist/AI_Evidence_Extractor_Update.zip` with `build_windows.ps1`.
3. Create a GitHub Release with the matching version tag and upload both files. The setup is for first installs and recovery; the update archive is used by the in-app updater.
4. Calculate both checksums in PowerShell:

   ```powershell
   (Get-FileHash .\dist\AI_Evidence_Extractor_Setup.exe -Algorithm SHA256).Hash.ToLower()
   (Get-FileHash .\dist\AI_Evidence_Extractor_Update.zip -Algorithm SHA256).Hash.ToLower()
   ```

5. Copy `latest.example.json` to `latest.json`. Set `version`, `published_at`, `download_url` and `sha256` to the setup URL and checksum for compatibility with older clients. Set `package_url` and `package_sha256` to the update archive URL and checksum. Set `release_notes` to a short description.
6. Push `latest.json` to `main`. Existing apps read it from the stable raw URL and show the update on their next check.
7. Apps built before internal updates must be upgraded once with the setup installer. Later releases download and apply their archive from inside the app.

The updater verifies HTTPS, the app identifier, the version, the package checksum, and every archive path before staging files. It only writes staged files under the app's user-data folder while downloading. When the user selects **Install Update**, it replaces the allowlisted app files in-process, shows progress for each file, and rolls back if any replacement fails. The user then restarts the app to load the new version. Keep the setup asset for first installs and recovery.

## Current release values

The app version is stored in `version.txt` and is embedded into the setup version by `installer.iss`. Generate `latest.json` after both release assets have been built and their SHA-256 hashes are known.
