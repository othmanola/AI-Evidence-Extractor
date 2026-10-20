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

$prerequisiteDir = Join-Path $PSScriptRoot 'build\prerequisites'
New-Item -ItemType Directory -Force -Path $prerequisiteDir | Out-Null
$prerequisites = @(
  @{
    Name = 'vc_redist.x64.exe'
    Uri = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
  },
  @{
    Name = 'MicrosoftEdgeWebview2Setup.exe'
    Uri = 'https://go.microsoft.com/fwlink/p/?LinkId=2124703'
  }
)
foreach ($prerequisite in $prerequisites) {
  $file = Join-Path $prerequisiteDir $prerequisite.Name
  try {
    Invoke-WebRequest -Uri $prerequisite.Uri -OutFile $file
  } catch {
    throw "Could not download $($prerequisite.Name): $($_.Exception.Message)"
  }
  $signature = Get-AuthenticodeSignature -FilePath $file
  if ($signature.Status -ne 'Valid' -or
      $signature.SignerCertificate.Subject -notlike '*Microsoft Corporation*' -or
      (Get-Item $file).Length -lt 1MB) {
    throw "The downloaded Microsoft prerequisite failed signature or size validation: $($prerequisite.Name)"
  }
}

$env:PYTHONPATH = $PSScriptRoot
$supabaseUrl = $env:AEE_SUPABASE_URL
$supabaseKey = $env:AEE_SUPABASE_PUBLISHABLE_KEY
if ([bool]$supabaseUrl -ne [bool]$supabaseKey) {
  throw 'Set both AEE_SUPABASE_URL and AEE_SUPABASE_PUBLISHABLE_KEY, or leave both unset to use account_defaults.json.'
}
if (-not $supabaseUrl) {
  $defaultsFile = Join-Path $PSScriptRoot 'account_defaults.json'
  if (Test-Path -LiteralPath $defaultsFile) {
    $defaults = Get-Content -Raw -LiteralPath $defaultsFile | ConvertFrom-Json
    $supabaseUrl = $defaults.project_url
    $supabaseKey = $defaults.anon_key
  }
}

$accountDefaultsPath = $null
try {
  if ($supabaseUrl -and $supabaseKey) {
    $accountDefaultsDirectory = Join-Path $env:TEMP ("aee-account-defaults-" + [guid]::NewGuid().ToString())
    New-Item -ItemType Directory -Path $accountDefaultsDirectory | Out-Null
    $accountDefaultsPath = Join-Path $accountDefaultsDirectory 'account_defaults.json'
    @{
      project_url = $supabaseUrl
      anon_key = $supabaseKey
    } | ConvertTo-Json -Compress | Set-Content -Path $accountDefaultsPath -Encoding utf8
    $env:AEE_ACCOUNT_DEFAULTS_PATH = $accountDefaultsPath
  } else {
    Write-Warning 'No default Supabase connection will be bundled. Users will need to configure Connection setup on each new PC.'
  }

  & $python -m PyInstaller --noconfirm --clean (Join-Path $PSScriptRoot 'AI Evidence Extractor.spec')
  if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

  $bundledDefaultsPath = Join-Path $PSScriptRoot 'dist\AI Evidence Extractor\account_defaults.json'
  if ($accountDefaultsPath) {
    Copy-Item -LiteralPath $accountDefaultsPath -Destination $bundledDefaultsPath -Force
  } elseif (Test-Path -LiteralPath $bundledDefaultsPath) {
    Remove-Item -LiteralPath $bundledDefaultsPath -Force
  }

  & $compiler (Join-Path $PSScriptRoot 'installer.iss')
  if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }
  Write-Host "Installer created at $(Join-Path $PSScriptRoot 'dist\AI_Evidence_Extractor_Setup.exe')"
  & $python (Join-Path $PSScriptRoot 'build_update_package.py')
  if ($LASTEXITCODE -ne 0) { throw "In-app update package build failed with exit code $LASTEXITCODE" }
} finally {
  Remove-Item Env:AEE_ACCOUNT_DEFAULTS_PATH -ErrorAction SilentlyContinue
  if ($accountDefaultsPath -and (Test-Path -LiteralPath $accountDefaultsPath)) {
    Remove-Item -LiteralPath $accountDefaultsPath -Force
    Remove-Item -LiteralPath (Split-Path -Parent $accountDefaultsPath) -Force
  }
}

Write-Host "Standalone app created at $(Join-Path $PSScriptRoot 'dist\AI Evidence Extractor\AI Evidence Extractor.exe')"
