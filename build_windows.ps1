$ErrorActionPreference = 'Stop'
$shortcutPath = Join-Path $PSScriptRoot 'dist\AI Evidence Extractor.lnk'
$shortcutDirectory = Split-Path $shortcutPath -Parent
if (-not (Test-Path $shortcutDirectory)) {
  New-Item -ItemType Directory -Path $shortcutDirectory | Out-Null
}
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $env:WINDIR 'System32\cmd.exe'
$shortcut.Arguments = '/c ""%LOCALAPPDATA%\Programs\AI Evidence Extractor\run_app.bat""'
$shortcut.Description = 'Launch AI Evidence Extractor'
$shortcut.IconLocation = '%LOCALAPPDATA%\Programs\AI Evidence Extractor\assets\aee.ico'
$shortcut.Save()

$compiler = @(
  "$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe",
  "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
  "$env:ProgramFiles(x86)\Inno Setup 7\ISCC.exe",
  "$env:ProgramFiles\Inno Setup 7\ISCC.exe",
  "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
  "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $compiler) {
  $cmd = Get-Command winget -ErrorAction SilentlyContinue
  if (-not $cmd) { throw 'Install Inno Setup 6 or 7, then run this script again.' }
  winget install --id JRSoftware.InnoSetup.7 -e --silent --accept-source-agreements --accept-package-agreements
  $compiler = @("$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe", "$env:ProgramFiles(x86)\Inno Setup 7\ISCC.exe", "$env:ProgramFiles\Inno Setup 7\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $compiler) { throw 'Inno Setup compiler was not found after installation.' }
& $compiler (Join-Path $PSScriptRoot 'installer.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }
Write-Host "Installer created at $(Join-Path $PSScriptRoot 'dist\AI_Evidence_Extractor_Setup.exe')"
& python (Join-Path $PSScriptRoot 'build_update_package.py')
if ($LASTEXITCODE -ne 0) { throw "In-app update package build failed with exit code $LASTEXITCODE" }
Write-Host "Shortcut created at $shortcutPath"
