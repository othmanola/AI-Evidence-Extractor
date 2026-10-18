$ErrorActionPreference = 'Stop'
$python = Join-Path $env:LOCALAPPDATA 'Programs\AI Evidence Extractor\runtime\python.exe'
if (-not (Test-Path $python)) { $python = (Get-Command python -ErrorAction Stop).Source }

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
$env:PYTHONPATH = $PSScriptRoot
& $python -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot 'AI Evidence Extractor.spec')
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
& $compiler (Join-Path $PSScriptRoot 'installer.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }
Write-Host "Installer created at $(Join-Path $PSScriptRoot 'dist\AI_Evidence_Extractor_Setup.exe')"
& $python (Join-Path $PSScriptRoot 'build_update_package.py')
if ($LASTEXITCODE -ne 0) { throw "In-app update package build failed with exit code $LASTEXITCODE" }
Write-Host "Standalone app created at $(Join-Path $PSScriptRoot 'dist\AI Evidence Extractor\AI Evidence Extractor.exe')"
