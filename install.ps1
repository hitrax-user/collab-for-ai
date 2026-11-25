# Установка Colab LLM на Windows: проверяет и ставит WSL 2 + Ubuntu, uv и colab CLI, кладёт ярлык на рабочий стол.
# Запуск: двойной клик по install.cmd (он вызывает этот файл). Повторный запуск безопасен — сделанное пропускается.
# -Distro — взять другой дистрибутив (по умолчанию первый Ubuntu*, иначе ставится Ubuntu); -NoShortcut, -NoLaunch — для проверки.
param([switch]$AfterReboot, [switch]$NoShortcut, [switch]$NoLaunch, [string]$Distro)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:WSL_UTF8 = '1'  # иначе wsl.exe пишет в UTF-16
$Here = $PSScriptRoot
$Self = $MyInvocation.MyCommand.Path
$RunOnce = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce'

function Say($m) { Write-Host "`n>> $m" -ForegroundColor Cyan }
function Ok($m) { Write-Host "   $m" -ForegroundColor Green }
function Warn($m) { Write-Host "   $m" -ForegroundColor Yellow }
function Stop-Setup($m) {
    Write-Host "`n$m" -ForegroundColor Red
    Read-Host "`nEnter — закрыть" | Out-Null
    exit 2  # install.cmd не ставит лишнюю паузу: она уже была
}

function Wsl {
    # wsl.exe с захватом вывода; код возврата — в $LASTEXITCODE
    (& wsl.exe @args 2>&1 | Out-String) -replace "`0", ''
}

function WslReady {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { return $false }
    $null = Wsl --version
    $LASTEXITCODE -eq 0
}

function Admin([string]$Exe, [string]$Arguments) {
    try { Start-Process $Exe $Arguments -Verb RunAs -Wait }
    catch { Stop-Setup 'Без разрешения администратора WSL не поставить. Запусти install.cmd ещё раз и нажми «Да».' }
}

function Sh([string]$Script, [string]$User) {
    # bash-скрипт внутри Ubuntu; base64 — чтобы PowerShell не испортил кавычки и переводы строк
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($Script -replace "`r", '')))
    $u = if ($User) { @('-u', $User) } else { @() }
    & wsl.exe -d $Distro @u -- bash -c "echo $b64 | base64 -d | bash" | Out-Host
    $LASTEXITCODE
}

function Distros { (Wsl -l -q) -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ } }

$VirtHelp = @"
Похоже, в BIOS/UEFI выключена аппаратная виртуализация — без неё WSL 2 не работает.
  1. Перезагрузи компьютер и зайди в BIOS (обычно Del, F2 или F10 при включении).
  2. Включи Intel Virtualization Technology (VT-x) или AMD SVM Mode — обычно в разделе Advanced / CPU.
  3. Сохрани (F10), загрузи Windows и снова запусти install.cmd.
Проверить: Диспетчер задач → Производительность → ЦП → «Виртуализация: Включено».
"@

function Reboot-AndResume($why) {
    if ($AfterReboot) { Stop-Setup "$why — и после перезагрузки тоже.`n`n$VirtHelp" }
    Set-ItemProperty $RunOnce 'ColabLLMSetup' "powershell -NoProfile -ExecutionPolicy Bypass -File `"$Self`" -AfterReboot"
    Warn "$why. Нужна перезагрузка — после входа в Windows установка продолжится сама."
    if ((Read-Host '   Перезагрузить сейчас? [Y/n]') -notmatch '^[nNтТ]') { Restart-Computer -Force }
    exit 0
}

Write-Host 'Colab LLM — установка' -ForegroundColor White

# 1. Windows и виртуализация
Say 'Проверяю Windows'
$build = [Environment]::OSVersion.Version.Build
if ($build -lt 19041) {
    Stop-Setup "Нужна Windows 10 версии 2004 (сборка 19041) или новее, а здесь сборка $build. Обнови Windows через Центр обновления."
}
Ok "Сборка $build — подходит"
$hyper = (Get-CimInstance Win32_ComputerSystem).HypervisorPresent
$firmware = @(Get-CimInstance Win32_Processor | ForEach-Object VirtualizationFirmwareEnabled)
if ($hyper) { Ok 'Виртуализация работает' }
elseif ($firmware -contains $false) { Warn 'Виртуализация в BIOS, похоже, выключена — если WSL не запустится, будет подсказка, как включить' }
else { Ok 'Виртуализация включена в BIOS' }

# 2. WSL
Say 'Проверяю WSL'
if (-not (WslReady)) {
    Warn 'WSL не установлен или устарел — ставлю (Windows спросит разрешение администратора)'
    if (Get-Command wsl.exe -ErrorAction SilentlyContinue) { Admin wsl.exe '--install --no-distribution' }
    if (-not (WslReady)) {
        # старые сборки Windows 10 не знают --no-distribution: включаем компоненты вручную и обновляем WSL
        $cmd = 'dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart & ' +
               'dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart & wsl --update'
        Admin cmd.exe "/c $cmd"
    }
    if (-not (WslReady)) { Reboot-AndResume 'WSL установлен, но заработает только после перезагрузки' }
}
Ok ((Wsl --version) -split "`r?`n" | Select-Object -First 1)
$null = Wsl --set-default-version 2
$wslconfig = Join-Path $env:USERPROFILE '.wslconfig'
if ((Test-Path $wslconfig) -and (Select-String -Path $wslconfig -Pattern '^\s*localhostForwarding\s*=\s*false' -Quiet)) {
    Warn "В $wslconfig выключен localhostForwarding — страница управления не откроется в браузере. Убери эту строку."
}

# 3. Ubuntu
Say 'Проверяю Ubuntu'
$have = if ($Distro) { Distros | Where-Object { $_ -eq $Distro } }
         else { Distros | Where-Object { $_ -match '^Ubuntu' } | Select-Object -First 1 }
if (-not $Distro) { $Distro = 'Ubuntu' }
if ($have) {
    $Distro = $have
    Ok "Уже есть: $Distro"
} else {
    Warn 'Ubuntu нет — скачиваю (несколько минут)'
    & wsl.exe --install -d $Distro --no-launch
    if (-not (Distros | Where-Object { $_ -eq $Distro })) {
        # старый формат дистрибутива: приложение поставилось, но ещё не распаковано
        $exe = Get-Command ubuntu.exe, ubuntu*.exe -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($exe) { & $exe.Source install --root }
    }
    if (-not (Distros | Where-Object { $_ -eq $Distro })) { Reboot-AndResume 'Ubuntu не установилась' }
}
& wsl.exe -d $Distro -u root -- true  # без захвата вывода: если Ubuntu спросит имя при первом запуске, это будет видно
if ($LASTEXITCODE -ne 0) { Reboot-AndResume 'Ubuntu не запускается' }
Remove-ItemProperty $RunOnce 'ColabLLMSetup' -ErrorAction SilentlyContinue
Ok 'Ubuntu запускается'

# 4. Пользователь Linux: при установке без окна Ubuntu входит под root — заводим обычного
if ((Wsl -d $Distro -- id -u) -match '(?m)^0\s*$') {
    $name = ($env:USERNAME.ToLower() -replace '[^a-z0-9_-]', '')
    if ($name -notmatch '^[a-z_]') { $name = "u$name" }
    if ($name.Length -lt 2) { $name = 'user' }
    Say "Создаю пользователя Linux «$name»"
    $null = Sh @"
id -u $name >/dev/null 2>&1 || useradd -m -s /bin/bash -G sudo $name
touch /etc/wsl.conf
sed -i '/^default=/d' /etc/wsl.conf
if grep -q '^\[user\]' /etc/wsl.conf; then sed -i '/^\[user\]/a default=$name' /etc/wsl.conf
else printf '\n[user]\ndefault=%s\n' $name >> /etc/wsl.conf; fi
"@ root
    $null = Wsl --terminate $Distro
    Ok "Готово. Пароль не задан; если понадобится sudo: wsl -u root passwd $name"
}

# 5. Пакеты, uv и colab CLI
Say 'Ставлю python3, uv и Google Colab CLI'
if ((Sh @'
set -e
if ! command -v python3 >/dev/null || ! command -v curl >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 curl ca-certificates
fi
'@ root) -ne 0) { Stop-Setup 'apt-get не смог поставить пакеты — проверь интернет и запусти install.cmd ещё раз.' }
$project = [regex]::Match((Wsl -d $Distro --exec wslpath -a $Here), '(?m)^/.*$').Value.Trim()
if (-not $project) { Stop-Setup "Не понял, где папка проекта внутри Linux: $Here" }
if ((Sh @"
set -e
export PATH="`$HOME/.local/bin:`$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v colab >/dev/null || uv tool install -q google-colab-cli
cd '$project'
sed -i 's/\r`$//' colab-as menu draw *.py   # если файлы пришли с виндовыми переводами строк
chmod +x colab-as menu draw 2>/dev/null || true
uv tool list | grep '^google-colab' | sed 's/^/   /'
"@) -ne 0) { Stop-Setup 'Не получилось поставить uv или colab CLI — проверь интернет и запусти install.cmd ещё раз.' }

# 6. Ярлык
if (-not $NoShortcut) {
    $lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Colab LLM.lnk'
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $s.TargetPath = "$env:WINDIR\System32\wsl.exe"
    $s.Arguments = "-d $Distro --cd `"$Here`" -- ./menu"
    $s.WorkingDirectory = $Here
    $s.IconLocation = "$env:WINDIR\System32\wsl.exe,0"
    $s.Save()
    Ok 'Ярлык «Colab LLM» на рабочем столе'
}

Say 'Всё готово. Дальше — меню: «+» добавляет Google-аккаунт, номер аккаунта открывает страницу управления.'
if (-not $NoLaunch) { & wsl.exe -d $Distro --cd $Here -- ./menu }
