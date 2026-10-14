# AI Evidence Extractor for Windows

This is a Windows desktop app for the existing video evidence triage project. It shows the existing Gradio interface inside a native WebView2 window, so the same analysis tabs and controls are available without opening a separate browser. The app includes a persistent History tab for the last 100 completed video analyses, with date, duration, detection count, case number, and inference mode. History is stored locally in `%LOCALAPPDATA%\AI Evidence Extractor\analysis_history.json`. The app icon is a custom creative spark mark.

## Install and first launch

Run `AI_Evidence_Extractor_Setup.exe`, then open the desktop shortcut. Setup installs a private Python 3.13 runtime and the Microsoft WebView2 runtime if needed. The first launch needs internet access and several minutes to prepare the Python environment and download approximately 600 MB of AI model weights from their original Hugging Face repositories into `%LOCALAPPDATA%\AI Evidence Extractor\models`.

The app runs locally on your computer and does not create a public Gradio share link. Video processing uses CPU by default. GPU inference is not configured in this desktop build.

Video object detection samples at 1 fps (one frame every second) to reduce analysis time. Audio analysis and output-video merging continue to use their own sampling and frame-rate settings. Object IDs are associated across sampled frames using detection overlap and position.

For challenging footage, select **Enhanced accuracy (slower)** before analysis to enable test-time augmentation on supported PyTorch detectors. This can improve detection robustness, especially for small or difficult objects, at the cost of longer analysis. It is off by default so normal analysis speed is unchanged. Validate any accuracy changes against a labeled set using `evaluation.py` before relying on them.

FFmpeg is installed as a Python package dependency and added to the process path at startup. User files and generated app data are stored under `%LOCALAPPDATA%\AI Evidence Extractor` and the operating system temporary folder.

The **Updates** tab checks the configured HTTPS release feed. When a newer version is available, the app verifies the downloaded setup file's SHA-256 checksum and opens the installer. See [UPDATE_RELEASES.md](UPDATE_RELEASES.md) for publishing releases and connecting the update feed.

## Build the installer

1. Install Inno Setup 6 or 7.
2. Run `build_windows.ps1` from PowerShell.
3. The installer is written to `dist\AI_Evidence_Extractor_Setup.exe`.

The installer distributes the application source and bootstrap scripts; the model weights and Python dependencies are fetched during first launch. This keeps the setup package smaller and ensures each installation downloads current model repository files. For a controlled academic release, pin and archive the exact model revisions before distributing.

## Project evaluation

Use `evaluation.py` and `EVALUATION_PROTOCOL.md` to score detections against manually labeled videos. Automated detections, threat scores, and EigenCAM are review aids and require human verification.
