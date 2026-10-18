# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import torchvision
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

gradio_datas, gradio_binaries, gradio_hidden = collect_all("gradio")
webview_datas, webview_binaries, webview_hidden = collect_all("webview")
safehttpx_datas, safehttpx_binaries, safehttpx_hidden = collect_all("safehttpx")
groovy_datas, groovy_binaries, groovy_hidden = collect_all("groovy")

datas = [
    ("assets", "assets"),
    ("version.txt", "."),
    ("update_config.json", "."),
] + gradio_datas + webview_datas + safehttpx_datas + groovy_datas + collect_data_files("ultralytics")
binaries = (
    gradio_binaries + webview_binaries + safehttpx_binaries + groovy_binaries
    + collect_dynamic_libs("torchvision")
    + [
        (str(extension), "torchvision")
        for extension in Path(torchvision.__file__).parent.glob("*.pyd")
    ]
)
hiddenimports = gradio_hidden + webview_hidden + safehttpx_hidden + groovy_hidden + [
    "app",
    "account_client",
    "update_client",
    "update_apply",
    "desktop_launcher",
    "download_models",
    "clr",
    "pythonnet",
    "soundfile",
    "cv2",
    "sklearn",
    "matplotlib",
    "PIL",
    "imageio_ffmpeg",
    "omegaconf",
    "torchvision",
    "pytorch_grad_cam",
] + collect_submodules("ultralytics")

a = Analysis(
    ["app_entry.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI Evidence Extractor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=["assets/aee.ico"],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AI Evidence Extractor",
)
