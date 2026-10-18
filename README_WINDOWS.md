# AI Evidence Extractor for Windows

This is a Windows desktop app for the existing video evidence triage project. The packaged executable opens the existing Gradio interface in a native WebView2 window and does not open a separate browser or terminal. The one-folder distribution includes the Python runtime and application dependencies; WebView2 Runtime is required on Windows. The app includes persistent, unlimited analysis history with case metadata, notes, detections, object tracks, evidence captures, CAM heatmaps, summaries, highlights, and reports. The history index is stored in `%LOCALAPPDATA%\AI Evidence Extractor\analysis_history.json`; generated per-analysis files are stored in `%LOCALAPPDATA%\AI Evidence Extractor\outputs\history`. Original videos are referenced by their source paths and are not copied into History. The app icon is a custom creative spark mark.

## Install and first launch

For a direct launch, run `dist\AI Evidence Extractor\AI Evidence Extractor.exe`; it works independently of the current working directory and does not require a `.lnk` shortcut. To create an installer, build `AI_Evidence_Extractor_Setup.exe` with the steps below, run it, and optionally use the shortcut it creates. The packaged application includes Python and its dependencies. The first launch needs internet access to download any missing AI model weights from their original Hugging Face repositories into `%LOCALAPPDATA%\AI Evidence Extractor\models`.

The app runs locally on your computer and does not create a public Gradio share link. Video processing uses CPU by default. GPU inference is not configured in this desktop build.

Video object detection samples at 1 fps (one frame every second) to reduce analysis time. Audio analysis and output-video merging continue to use their own sampling and frame-rate settings. Object IDs are associated across sampled frames using detection overlap and position.

For challenging footage, select **Enhanced accuracy (slower)** before analysis to enable test-time augmentation on supported PyTorch detectors. This can improve detection robustness, especially for small or difficult objects, at the cost of longer analysis. It is off by default so normal analysis speed is unchanged. Validate any accuracy changes against a labeled set using `evaluation.py` before relying on them.

FFmpeg is installed as a Python package dependency and added to the process path at startup. User files and generated app data are stored under `%LOCALAPPDATA%\AI Evidence Extractor` and the operating system temporary folder.

The left-side menu contains **Account**, **History**, **Detailed History**, **Updates**, **About Us**, and **Contact Us**. Account opens profile details and account security. History is the simple analysis summary table; Detailed History opens the complete analysis records, captures, reports, and tracks, with ZIP downloads for a selected analysis or the full archive. ZIP exports contain generated analysis artifacts and metadata, not copies of original videos. The top-right toolbar shows **Update the app** to the left of the theme control only when a newer release is available; selecting it opens the update center. Guests can save a local display name and photo or sign in with Google; signed-in users can log out and manage their profile. The app checks the HTTPS release feed on startup and from the Updates page. In the packaged Windows app, updates download and verify the signed release metadata's setup installer before opening it; complete setup and relaunch the app to finish. Source-based launches continue to use the staged in-app package flow. See [UPDATE_RELEASES.md](UPDATE_RELEASES.md) for publishing releases and connecting the update feed.

The app requires an email/password account, Google sign-in, or an explicit guest-mode choice before opening the workspace. That choice is remembered between launches. Guest profile names and photos are stored only on this computer; signed-in profile data is stored in a connected Supabase project. Source videos, analysis history, reports, and results remain local. Configure accounts using [SUPABASE_SETUP.md](SUPABASE_SETUP.md).

## Build the application and installer

1. Install the project dependencies from `requirements.txt` in the build Python environment, plus PyInstaller.
2. Run `build_windows.ps1` from PowerShell. It creates the one-folder application using `AI Evidence Extractor.spec`.
3. The standalone executable is written to `dist\AI Evidence Extractor\AI Evidence Extractor.exe`. The script also creates the installer and in-app update archive; Inno Setup 6 or 7 is required for the installer step.

The installer places the packaged application files in the selected install directory and creates shortcuts that target the executable directly. Model weights remain external mutable data and are fetched during first launch. For a controlled academic release, pin and archive the exact model revisions before distributing.

## Project evaluation

Use `evaluation.py` and `EVALUATION_PROTOCOL.md` to score detections against manually labeled videos. Automated detections, threat scores, and EigenCAM are review aids and require human verification.
