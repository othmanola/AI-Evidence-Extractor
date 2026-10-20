#define VersionFileHandle FileOpen("version.txt")
#define AppVersion FileRead(VersionFileHandle)
#expr FileClose(VersionFileHandle)

[Setup]
AppId={{AEE-EXTRACTOR-2026-10}}
AppName=AI Evidence Extractor
AppVersion={#AppVersion}
AppVerName=AI Evidence Extractor {#AppVersion}
AppPublisher=AI Evidence Extractor Project
DefaultDirName={localappdata}\Programs\AI Evidence Extractor
DisableDirPage=no
UsePreviousAppDir=yes
DefaultGroupName=AI Evidence Extractor
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=AI_Evidence_Extractor_Setup
SetupIconFile=assets\aee.ico
UninstallDisplayIcon={app}\_internal\assets\aee.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern dark polar hidebevels includetitlebar
WizardImageFile=assets\setup_wizard.png
WizardSmallImageFile=assets\aee.png
SetupLogging=yes
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=force
RestartApplications=yes

[Messages]
WelcomeLabel1=Welcome to AI Evidence Extractor
WelcomeLabel2=Install the desktop app to analyze videos, review evidence, and export reports.%n%nClick Next to continue.
SelectDirDesc=Choose where AI Evidence Extractor should be installed.
SelectDirLabel3=The app files will be installed in this folder:
SelectDirBrowseLabel=Keep this folder, or click Browse... to choose another install location.%nYour choice will be remembered for future updates.

[Files]
Source: "dist\AI Evidence Extractor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "build\prerequisites\vc_redist.x64.exe"; DestName: "vc_redist.x64.exe"; Flags: dontcopy
Source: "build\prerequisites\MicrosoftEdgeWebview2Setup.exe"; DestName: "MicrosoftEdgeWebview2Setup.exe"; Flags: dontcopy

[Icons]
Name: "{userprograms}\AI Evidence Extractor\AI Evidence Extractor"; Filename: "{app}\AI Evidence Extractor.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\aee.ico"; AppUserModelID: "AI.EvidenceExtractor.Desktop"
Name: "{userdesktop}\AI Evidence Extractor"; Filename: "{app}\AI Evidence Extractor.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\aee.ico"; AppUserModelID: "AI.EvidenceExtractor.Desktop"

[Run]
Filename: "{app}\AI Evidence Extractor.exe"; Description: "Launch AI Evidence Extractor"; WorkingDir: "{app}"; Flags: postinstall unchecked nowait skipifsilent

[Code]
function IsVCRedistInstalled: Boolean;
var
  Installed: Cardinal;
begin
  Result := IsWin64 and
    RegQueryDWordValue(HKLM64,
      'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
      'Installed', Installed) and
    (Installed = 1);
end;

function IsWebView2RuntimeInstalled: Boolean;
var
  Version: String;
begin
  Result :=
    (RegQueryStringValue(HKLM32,
      'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
      'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')) or
    (RegQueryStringValue(HKCU,
      'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
      'pv', Version) and (Version <> '') and (Version <> '0.0.0.0'));
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  RedistPath: String;
  WebViewPath: String;
  ResultCode: Integer;
begin
  Result := '';
  if not IsVCRedistInstalled then
  begin
    ExtractTemporaryFile('vc_redist.x64.exe');
    RedistPath := ExpandConstant('{tmp}\vc_redist.x64.exe');
    if not Exec(RedistPath, '/install /quiet /norestart', '',
        SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    begin
      Result := 'The Microsoft Visual C++ Runtime installer could not be started.';
      Exit;
    end;

    if (ResultCode <> 0) and (ResultCode <> 1638) and (ResultCode <> 3010) then
    begin
      Result := 'The Microsoft Visual C++ Runtime installation failed (error ' +
        IntToStr(ResultCode) + ').';
      Exit;
    end;

    if ResultCode = 3010 then
      NeedsRestart := True;
  end;

  if not IsWebView2RuntimeInstalled then
  begin
    ExtractTemporaryFile('MicrosoftEdgeWebview2Setup.exe');
    WebViewPath := ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe');
    if not Exec(WebViewPath, '/silent /install', '',
        SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    begin
      Result := 'The Microsoft Edge WebView2 Runtime installer could not be started.';
      Exit;
    end;

    if (ResultCode <> 0) and (ResultCode <> 3010) then
    begin
      Result := 'The Microsoft Edge WebView2 Runtime installation failed (error ' +
        IntToStr(ResultCode) + ').';
      Exit;
    end;

    if ResultCode = 3010 then
      NeedsRestart := True;
  end;
end;
