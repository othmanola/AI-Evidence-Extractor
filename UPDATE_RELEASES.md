# Publishing app updates

Version 1.2.60 installs missing Microsoft Visual C++ and WebView2 runtimes
during setup, preventing first-launch failures on new Windows PCs. The app's
Python runtime and application dependencies remain bundled with the installer.

Version 1.2.59 adds a complete Arabic interface with right-to-left layout,
including mirrored navigation and account pages, and fixes mobile placement of
the profile-photo actions in RTL mode.

Version 1.2.58 refreshes the Account page with clearer profile identity,
improved local and signed-in profile summaries, and responsive styling.

Version 1.2.57 migrates existing desktop and Start Menu shortcuts from an
older frozen executable to the installed Python launcher, so in-app source
updates remain active on later launches.

Version 1.2.56 reads the installed version marker directly for each update
check, even if a frozen bundle still contains an older version marker.

Version 1.2.55 forces the desktop launcher to load updateable modules and
version metadata from the installed app folder, and logs the resolved module
and version paths for startup diagnostics.

Version 1.2.54 prefers the installed version marker after an in-app update,
preventing an older version embedded in the executable from causing the same
update to be offered on every launch.

Version 1.2.53 starts the installed source modules inside the packaged
Python runtime when no separate runtime is present, so a fresh installation
does not depend on another Python installation on that computer.

Version 1.2.52 overlaps sequential inference with bounded video-frame
prefetching, preserving analyzed frame order and selection, and reports the
active inference device and precision in the ensemble panel.

Version 1.2.51 keeps the completed download and installation progress bars
visible and shows an animated progress indicator when the download size is
unknown.

Version 1.2.50 fixes the Detailed History remove action output mapping so saved
files, detection tables, tracking tables, and download controls receive the
correct values.

Installed copies use a small HTTPS JSON manifest to check the current release. GitHub Releases host both the full setup installer and an in-app update archive. The app downloads and verifies the archive, then applies it in place when the user selects Install Update. The app shows download progress and file-by-file installation progress; no setup file is opened. Restart the app afterward to load the new version. The setup installer remains for first installations and recovery.

Version 1.2.40 presents Detailed History as a newest-first video list with
per-analysis details, ZIP downloads, and safe removal of saved analysis data.
The release workflow installs the project and packaging dependencies before
building the Windows installer and in-app update package.
Version 1.2.41 installs updates from the Updates page using a quiet setup
progress window, closing and automatically restarting the app around file
replacement.
Version 1.2.42 makes Detailed History actions smaller and right-aligned,
highlights destructive actions in red, and adds a safe Clear all history
action that preserves original source videos.
Version 1.2.43 fixes startup when building the Detailed History controls.
Version 1.2.44 further compacts and right-aligns Detailed History actions,
including the refresh, ZIP, and clear-history controls.
Version 1.2.45 applies updates internally in packaged builds and shows
download and file-by-file installation progress in the Updates window.
Version 1.2.46 gives Detailed History destructive actions explicit red
gradients, stronger button selectors, and modern hover styling.
Version 1.2.47 streams a visible progress bar through update checks, downloads,
and internal installation in the Updates window.
Version 1.2.48 modernizes Details, ZIP download, Refresh, and Download all ZIP
actions in Detailed History with coordinated gradients and hover states.
Version 1.2.49 launches packaged installs through the installed Python runtime
and loose source modules so future in-app updates affect the code loaded on restart.
Version 1.2.30 gives the setup wizard a modern dark polar theme and creates
the desktop AI Evidence Extractor shortcut during installation.
Version 1.2.29 restores the basic History summary view and adds a separate
Detailed History page with ZIP exports for an individual analysis or the full
local archive; source videos are not included in ZIP downloads.
Version 1.2.28 adds Account and History navigation, a modern analysis detail browser,
and unlimited local retention of complete analysis records and generated
evidence artifacts while keeping original videos in their source locations.
Version 1.2.27 gives Log out explicit ID-based gradient styling and changes
the language selector to a collapsed, accessible click-to-open menu. Version
1.2.26 improves profile form field order and responsive spacing,
modernizes account cards, and gives the Log out action a clear red gradient.
Version 1.2.25 gives the account profile card a calm, solid background and
identifies Google sign-ins with a Google icon and provider badge. Version
1.2.24 replaces the decorative utility-page headers with a compact,
consistent navigation bar, clearer back-to-app actions, and refined dark/light
and mobile styling. Version 1.2.23 refreshed the Settings, Updates, About Us,
and Contact Us headers with section labels and descriptions. Version 1.2.22
presents the profile-photo editor in a centered window, closes
it when Apply is clicked, styles Apply to match the app, improves the
language-selector layout, and checks the Supabase Google provider before
sign-in with actionable setup guidance when it is disabled.
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

1. Increase `version.txt` to the next three-part version and include that change in a push to `main`.
2. The **Publish update release** GitHub Actions workflow automatically builds the installer and in-app update archive, calculates their SHA-256 checksums, creates the matching GitHub Release, and publishes `latest.json` both as a release asset and on `main`.
3. Confirm the workflow completed successfully and that the release is marked latest. Do not manually reuse a release tag or upload assets from a different build.
4. If automatic publishing needs to be retried, use the workflow's **Run workflow** action on `main` after fixing the failure. It refreshes the assets and manifest for that version in place.
5. The workflow updates `latest.json` on `main` for compatibility with older installed clients; new builds use the release asset as the primary feed.
6. Apps built before internal updates must be upgraded once with the setup installer. Later releases download and apply their archive from inside the app.

The updater verifies HTTPS, the app identifier, the version, the package checksum, and every archive path before staging files. It only writes staged files under the app's user-data folder while downloading. When the user selects **Install Update**, it replaces the allowlisted app files in-process, shows progress for each file in the GUI, and rolls back if any replacement fails. The user then restarts the app to load the new version. Keep the setup asset for first installs and recovery.

The app version is stored in `version.txt` and is embedded into the setup version by `installer.iss`. The publishing workflow generates the manifest only after both release assets have been built and hashed.
