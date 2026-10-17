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
WizardStyle=modern dark polar hidebevels includetitlebar
WizardImageFile=assets\setup_wizard.png
WizardSmallImageFile=assets\aee.png
SetupLogging=yes
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
Source: "download_models.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "run_app.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\aee.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "assets\setup_wizard.png"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "README_WINDOWS.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "SUPABASE_SETUP.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "UPDATE_RELEASES.md"; DestDir: "{app}"; Flags: ignoreversion

[InstallDelete]
Type: files; Name: "{app}\AI Evidence Extractor.exe"

[Icons]
Name: "{app}\AI Evidence Extractor"; Filename: "{sys}\cmd.exe"; Parameters: "/c ""{app}\run_app.bat"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\aee.ico"; AppUserModelID: "AI.EvidenceExtractor.Desktop"
Name: "{group}\AI Evidence Extractor"; Filename: "{sys}\cmd.exe"; Parameters: "/c ""{app}\run_app.bat"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\aee.ico"; AppUserModelID: "AI.EvidenceExtractor.Desktop"
Name: "{autodesktop}\AI Evidence Extractor"; Filename: "{sys}\cmd.exe"; Parameters: "/c ""{app}\run_app.bat"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\aee.ico"; AppUserModelID: "AI.EvidenceExtractor.Desktop"

[Run]
Filename: "{sys}\cmd.exe"; Parameters: "/c ""{app}\run_app.bat"""; Description: "Launch AI Evidence Extractor"; WorkingDir: "{app}"; Flags: postinstall unchecked nowait skipifsilent
