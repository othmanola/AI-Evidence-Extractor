# Publishing app updates

Installed copies use a small HTTPS JSON manifest to check the current release. GitHub Releases host both the full setup installer and a source update archive. The app downloads the archive and installs it in-place when the user selects Install Update; a progress indicator reports each file. Restart the app afterward to load the new version.

Version 1.2.22 presents the profile-photo editor in a centered window, closes
it when Apply is clicked, styles Apply to match the app, improves the language
selector layout, and checks the Supabase Google provider before sign-in with
actionable setup guidance when it is disabled.
Version 1.2.21 fixes the photo editor's Gradio handoff and hides saved
connection settings from users after the first successful setup. Version
1.2.20 added a square profile-photo crop editor with positioning and zoom
controls inspired by X. Version 1.2.19 added crop and resize controls
before applying a profile photo, hid email for guest profiles, added an
English/Arabic language selector, and accepted common Supabase dashboard and
project URL formats. Version 1.2.18 added a dedicated profile photo picker,
improved app-wide visual consistency, and verified Supabase connection
settings before saving. Version
1.2.17 simplified profile settings and made Supabase connection setup safe to
rerun. Version 1.2.16 refined the profile dashboard with a cleaner, compact
photo picker and more consistent responsive card styling. Version 1.2.15 added
editable account details, photo preview, validation, and local guest-profile
persistence.

## Connect this build to your release feed

1. New builds check the latest published release asset:
   `https://github.com/othmanola/AI-Evidence-Extractor/releases/latest/download/latest.json`
2. Keep that URL in every installer you distribute. If the repository becomes private or its default branch changes, configure another public HTTPS manifest URL in `update_config.json` and rebuild.
3. Each check adds a cache-busting query and sends no-cache headers so intermediary caches are less likely to serve an older manifest.

## Publish a new version

1. Increase `version.txt` to the next three-part version (this release is `1.2.21`) and include that change in a push to `main`.
2. The **Publish update release** GitHub Actions workflow automatically builds the installer and in-app update archive, calculates their SHA-256 checksums, creates the matching GitHub Release, and publishes `latest.json` both as a release asset and on `main`.
3. Confirm the workflow completed successfully and that the release is marked latest. Do not manually reuse a release tag or upload assets from a different build.
4. If automatic publishing needs to be retried, use the workflow's **Run workflow** action on `main` after fixing the failure. It refreshes the assets and manifest for that version in place.
5. The workflow updates `latest.json` on `main` for compatibility with older installed clients; new builds use the release asset as the primary feed.
6. Apps built before internal updates must be upgraded once with the setup installer. Later releases download and apply their archive from inside the app.

The updater verifies HTTPS, the app identifier, the version, the package checksum, and every archive path before staging files. It only writes staged files under the app's user-data folder while downloading. When the user selects **Install Update**, it replaces the allowlisted app files in-process, shows progress for each file, and rolls back if any replacement fails. The user then restarts the app to load the new version. Keep the setup asset for first installs and recovery.

The app version is stored in `version.txt` and is embedded into the setup version by `installer.iss`. The publishing workflow generates the manifest only after both release assets have been built and hashed.
