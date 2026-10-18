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
PrivilegesRequired=lowest
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

[Icons]
Name: "{group}\AI Evidence Extractor"; Filename: "{app}\AI Evidence Extractor.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\aee.ico"; AppUserModelID: "AI.EvidenceExtractor.Desktop"
Name: "{autodesktop}\AI Evidence Extractor"; Filename: "{app}\AI Evidence Extractor.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\aee.ico"; AppUserModelID: "AI.EvidenceExtractor.Desktop"

[Run]
Filename: "{app}\AI Evidence Extractor.exe"; Description: "Launch AI Evidence Extractor"; WorkingDir: "{app}"; Flags: postinstall unchecked nowait skipifsilent
