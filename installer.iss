; Inno Setup: інсталятор InkReel без прав адміністратора + пункт "Відкрити за допомогою".
; Картинки майстра: python tools/make_installer_art.py (installer\art\*.bmp). Збірка: scripts\build_exe.bat
#define AppName "InkReel"
#define AppVer "1.0.0"
#define Exe "InkReel.exe"
; підпис у вікні встановлення й «Програми та засоби»: у бета-каналі це мітка бети (benvid.BETA_LABEL), у стабільному збігається з AppVer
#define AppLabel "1.1.0-beta.1"

[Setup]
AppId={{6B1F6C54-6F0A-4C0E-9B55-1D6C6B2F1A01}
AppName={#AppName}
AppVersion={#AppLabel}
AppVerName={#AppName} {#AppLabel}
AppPublisher={#AppName}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=InkReel-Setup
Compression=lzma2/max
SolidCompression=yes
VersionInfoVersion={#AppVer}.0
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup
SetupIconFile=benvid\assets\inkreel.ico
UninstallDisplayIcon={app}\{#Exe}
UninstallDisplayName={#AppName} {#AppLabel}
WizardStyle=modern
WizardSizePercent=110
DisableWelcomePage=no
WizardImageFile=installer\art\wizard_1x.bmp,installer\art\wizard_2x.bmp,installer\art\wizard_3x.bmp
WizardSmallImageFile=installer\art\small_55.bmp,installer\art\small_82.bmp,installer\art\small_110.bmp
ShowLanguageDialog=auto
LanguageDetectionMethod=uilanguage
ChangesAssociations=yes
AppMutex=InkReelAppMutex
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "ukrainian"; MessagesFile: "compiler:Languages\Ukrainian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
ukrainian.WelcomeLabel2=Майстер встановить [name/ver] на ваш комп'ютер.%n%nУсе рахується на вашому пристрої: програма нічого не надсилає в інтернет.%n%nПеред продовженням закрийте InkReel, якщо він запущений.
english.WelcomeLabel2=This wizard will install [name/ver] on your computer.%n%nEverything is processed on your device: the program sends nothing to the internet.%n%nClose InkReel before you continue if it is running.
ukrainian.FinishedLabel=[name] встановлено. Відкрийте відео, GIF чи фото перетягуванням у вікно або через «Відкрити за допомогою».
english.FinishedLabel=[name] is installed. Open a video, GIF or photo by dragging it into the window or with "Open with".

[CustomMessages]
ukrainian.GroupIntegration=Інтеграція з Windows:
english.GroupIntegration=Windows integration:
ukrainian.AskRemoveData=Видалити також налаштування, журнал помилок і кеші обчислень та відеокарти (%LOCALAPPDATA%\InkReel)?%n%nГотові файли відео, GIF і фото не зачіпаються.
english.AskRemoveData=Also remove settings, the error log and the compute and GPU caches (%LOCALAPPDATA%\InkReel)?%n%nYour finished videos, GIFs and photos are not touched.
ukrainian.IconUninstall=Видалити InkReel
english.IconUninstall=Uninstall InkReel
ukrainian.TaskDesktop=Ярлик на робочому столі
english.TaskDesktop=Desktop shortcut
ukrainian.TaskCtx=Додати в меню правого кліку для фото й GIF: «Відкрити в InkReel» і «Зробити чорно-біле»
english.TaskCtx=Add to the right-click menu for photos and GIFs: "Open in InkReel" and "Make black-and-white"
ukrainian.CtxOpen=Відкрити в InkReel
english.CtxOpen=Open in InkReel
ukrainian.CtxQuick=Зробити чорно-біле (InkReel)
english.CtxQuick=Make black-and-white (InkReel)
ukrainian.IkrType=Відео InkReel (.ikr)
english.IkrType=InkReel video (.ikr)
ukrainian.RunApp=Запустити InkReel
ukrainian.LangCode=uk
english.LangCode=en
english.RunApp=Launch InkReel

[Tasks]
Name: "ctxmenu"; Description: "{cm:TaskCtx}"; GroupDescription: "{cm:GroupIntegration}"
Name: "desktop"; Description: "{cm:TaskDesktop}"; GroupDescription: "{cm:GroupIntegration}"; Flags: unchecked

[Files]
Source: "dist\InkReel\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
; значок файлів .ikr лежить поруч із програмою: шлях до нього в реєстрі не залежить від розкладки збірки PyInstaller
Source: "benvid\assets\ikr.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#Exe}"
Name: "{group}\{cm:IconUninstall}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#Exe}"; Tasks: desktop

[Registry]
; мова, обрана в інсталяторі, - мова застосунку при першому запуску
Root: HKCU; Subkey: "Software\InkReel"; ValueType: string; ValueName: "InstallLang"; ValueData: "{cm:LangCode}"; Flags: uninsdeletevalue
; з'являється у списку "Відкрити за допомогою" для відео, GIF і фото
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "{#AppName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\shell\open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\DefaultIcon"; ValueType: string; ValueData: "{app}\{#Exe},0"
; Тип файлу .ikr (готове відео InkReel): назва, значок і «відкривати в InkReel» за подвійним кліком. Без цього Провідник показує «Файл IKR»
; з безіменним значком і питає, чим відкрити. HKCU - без прав адміністратора; при видаленні програми все прибирається.
Root: HKCU; Subkey: "Software\Classes\.ikr"; ValueType: string; ValueData: "InkReel.Baked"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\.ikr\OpenWithProgids"; ValueType: string; ValueName: "InkReel.Baked"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\InkReel.Baked"; ValueType: string; ValueData: "{cm:IkrType}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\InkReel.Baked"; ValueType: string; ValueName: "FriendlyTypeName"; ValueData: "{cm:IkrType}"
Root: HKCU; Subkey: "Software\Classes\InkReel.Baked\DefaultIcon"; ValueType: string; ValueData: "{app}\ikr.ico"
Root: HKCU; Subkey: "Software\Classes\InkReel.Baked\shell"; ValueType: string; ValueData: "open"
Root: HKCU; Subkey: "Software\Classes\InkReel.Baked\shell\open"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "{#AppName}"
Root: HKCU; Subkey: "Software\Classes\InkReel.Baked\shell\open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""
; >>> SupportedTypes (згенеровано: python tools/sync_installer.py - руками не правити) >>>
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".mp4"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".mkv"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".mov"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".avi"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".webm"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".m4v"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".flv"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".wmv"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".mpg"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".mpeg"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".ts"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".m2ts"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".mts"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".3gp"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".3g2"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".ogv"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".vob"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".asf"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".f4v"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".divx"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".mxf"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".m2v"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".mpe"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".qt"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".rm"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".rmvb"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".ikr"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".gif"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".apng"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".png"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".jpg"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".jpeg"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".jpe"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".jfif"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".bmp"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".dib"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".webp"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".tif"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".tiff"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".avif"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".heic"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".heif"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".jp2"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".j2k"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".ico"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".tga"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".ppm"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".pgm"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".pbm"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".pnm"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".psd"; ValueData: ""
Root: HKCU; Subkey: "Software\Classes\Applications\{#Exe}\SupportedTypes"; ValueType: string; ValueName: ".hdr"; ValueData: ""
; <<< SupportedTypes <<<

; Правий клік по фото й GIF (у Windows 11 - у класичному меню «Показати додаткові параметри»): без прав адміністратора (HKCU).
;   «Відкрити в InkReel» - у звичайному вікні;  «Зробити чорно-біле» - швидко, без вікна програми: InkReel.exe --quick (див. benvid/quick.py).
; >>> ContextMenu (згенеровано: python tools/sync_installer.py - руками не правити) >>>
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.png\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpg\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpe\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpe\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpe\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpe\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpe\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jpe\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jfif\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jfif\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jfif\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jfif\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jfif\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jfif\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.bmp\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.dib\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.dib\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.dib\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.dib\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.dib\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.dib\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.webp\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tif\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tif\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tif\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tif\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tif\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tif\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tiff\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.avif\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.avif\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.avif\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.avif\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.avif\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.avif\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heic\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heic\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heic\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heic\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heic\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heic\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heif\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heif\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heif\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heif\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heif\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.heif\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jp2\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jp2\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jp2\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jp2\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jp2\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.jp2\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.j2k\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.j2k\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.j2k\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.j2k\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.j2k\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.j2k\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tga\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tga\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tga\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tga\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tga\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.tga\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ppm\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ppm\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ppm\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ppm\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ppm\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.ppm\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pgm\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pgm\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pgm\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pgm\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pgm\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pgm\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pbm\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pbm\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pbm\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pbm\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pbm\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pbm\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pnm\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pnm\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pnm\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pnm\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pnm\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.pnm\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.psd\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.psd\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.psd\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.psd\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.psd\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.psd\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.hdr\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.hdr\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.hdr\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.hdr\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.hdr\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.hdr\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.gif\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.gif\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.gif\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.gif\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.gif\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.gif\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.apng\shell\InkReel.Open"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxOpen}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.apng\shell\InkReel.Open"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.apng\shell\InkReel.Open\command"; ValueType: string; ValueData: """{app}\{#Exe}"" ""%1"""; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.apng\shell\InkReel.Quick"; ValueType: string; ValueName: "MUIVerb"; ValueData: "{cm:CtxQuick}"; Tasks: ctxmenu; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.apng\shell\InkReel.Quick"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\{#Exe}"",0"; Tasks: ctxmenu
Root: HKCU; Subkey: "Software\Classes\SystemFileAssociations\.apng\shell\InkReel.Quick\command"; ValueType: string; ValueData: """{app}\{#Exe}"" --quick ""%1"""; Tasks: ctxmenu
; <<< ContextMenu <<<

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/F /IM InkReel.exe /IM InkReel-Worker.exe"; Flags: runhidden; RunOnceId: "KillInkReel"

[Code]
// перед встановленням закрити всі старі InkReel і їхні обробники (інакше файли зайняті, а в процесах висять старі версії).
// Крім імен exe, гасимо ВСІ процеси, що запущені з теки встановлення (ffmpeg з _internal теж тримає файл і давав
// «DeleteFile збій; код 5. Access is denied»), і даємо системі мить відпустити файли
function PrepareToInstall(var NeedsRestart: Boolean): String;
var Code: Integer; Ps: String;
begin
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /IM InkReel.exe /IM InkReel-Worker.exe', '', SW_HIDE, ewWaitUntilTerminated, Code);
  Ps := '-NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith(''' + ExpandConstant('{app}') + '\'', ''OrdinalIgnoreCase'') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"';
  Exec('powershell.exe', Ps, '', SW_HIDE, ewWaitUntilTerminated, Code);
  Sleep(1000);
  Result := '';
end;

// на видаленні питаємо, чи прибирати й дані користувача; тихе видалення (/SILENT, оновлення) їх не чіпає
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if not UninstallSilent then
      if MsgBox(CustomMessage('AskRemoveData'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(ExpandConstant('{localappdata}\InkReel'), True, True, True);
end;

[Run]
Filename: "{app}\{#Exe}"; Description: "{cm:RunApp}"; Flags: nowait postinstall skipifsilent
