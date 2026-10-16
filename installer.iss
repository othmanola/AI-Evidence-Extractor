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
UninstallDisplayIcon={app}\assets\aee.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern dynamic windows11 hidebevels
WizardImageFile=assets\setup_wizard.png
WizardSmallImageFile=assets\aee.png
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Messages]
WelcomeLabel1=Welcome to AI Evidence Extractor
WelcomeLabel2=Install the desktop app to analyze videos, review evidence, and export reports.%n%nClick Next to continue.
SelectDirDesc=Choose where AI Evidence Extractor should be installed.
SelectDirLabel3=The app files will be installed in this folder:
SelectDirBrowseLabel=Keep this folder, or click Browse... to choose another install location.%nYour choice will be remembered for future updates.

[Files]
Source: "app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "account_client.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "update_client.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "update_apply.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "version.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "update_config.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "desktop_launcher.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "desktop_entry.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\AI Evidence Extractor.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "download_models.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "run_app.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\aee.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "assets\setup_wizard.png"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "assets\python-3.13.16-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "assets\MicrosoftEdgeWebView2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "README_WINDOWS.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "SUPABASE_SETUP.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "UPDATE_RELEASES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AI Evidence Extractor"; Filename: "{app}\AI Evidence Extractor.exe"; WorkingDir: "{app}"; IconFilename: "{app}\assets\aee.ico"; AppUserModelID: "AI.EvidenceExtractor.Desktop"
Name: "{autodesktop}\AI Evidence Extractor"; Filename: "{app}\AI Evidence Extractor.exe"; WorkingDir: "{app}"; IconFilename: "{app}\assets\aee.ico"; AppUserModelID: "AI.EvidenceExtractor.Desktop"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{tmp}\python-3.13.16-amd64.exe"; Parameters: "/quiet InstallAllUsers=0 TargetDir=""{app}\runtime"" Include_launcher=0 Include_pip=1 Include_test=0"; StatusMsg: "Installing the private Python runtime..."; Flags: waituntilterminated; Check: not FileExists(ExpandConstant('{app}\runtime\python.exe'))
Filename: "{tmp}\MicrosoftEdgeWebView2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "Checking the Microsoft WebView desktop component..."; Flags: waituntilterminated
Filename: "{app}\AI Evidence Extractor.exe"; Description: "Launch AI Evidence Extractor"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent
