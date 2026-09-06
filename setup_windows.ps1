<#
    Windows 一键本地化部署脚本

    用途：在一台刚克隆本仓库的 Windows 设备上启动已经训练好的模型 Web 端。
    脚本只负责 CPU 推理环境和 Django 服务启动，不下载 IDRiD 原始数据，也不重新训练模型。
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,

    [ValidateNotNullOrEmpty()]
    [string]$BindAddress = "127.0.0.1",

    [switch]$SkipInstall,
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"

# 【项目路径】使用脚本所在目录，而不是当前终端目录，保证从任意位置调用都能找到项目文件。
$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
Set-Location -LiteralPath $ProjectRoot

$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsFile = Join-Path $ProjectRoot "requirements-render.txt"
$ArtifactFile = Join-Path $ProjectRoot "artifacts\idrid_resnet_model.json"
$CheckpointFile = Join-Path $ProjectRoot "artifacts\idrid_resnet_best_fp16.pt"

function Resolve-SystemPython {
    <# 【Python 检查】优先使用 Python Launcher 的 3.11 版本；没有 Launcher 时使用 PATH 中
       的 python，并要求版本不低于 3.10，避免依赖开发机上的固定盘符路径。 #>
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        $probe = & $launcher.Source -3.11 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe) {
            return [string]($probe | Select-Object -Last 1).Trim()
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        & $python.Source -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $python.Source
        }
    }

    throw "未找到 Python 3.10 或更高版本。请先安装 Python 3.11，并重新运行此脚本。"
}

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string[]]$Arguments
    )

    & $script:VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python 命令执行失败（退出码 $LASTEXITCODE）：$($Arguments -join ' ')"
    }
}

Write-Host "== 糖尿病视网膜病变分类系统：Windows 本地部署 ==" -ForegroundColor Cyan

# 【模型文件检查】JSON artifact 记录模型配置，FP16 checkpoint 保存实际 ResNet-18 权重。
# 两者都在 GitHub 仓库中，缺少任意一个时不应继续启动 Web 服务。
$missingFiles = @($ArtifactFile, $CheckpointFile) | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
if ($missingFiles.Count -gt 0) {
    $display = $missingFiles -join [Environment]::NewLine
    throw "模型文件缺失，请确认已完整克隆仓库：$display"
}

if (-not (Test-Path -LiteralPath $RequirementsFile -PathType Leaf)) {
    throw "依赖文件不存在：$RequirementsFile"
}

# 【虚拟环境】首次运行创建 .venv；后续运行复用同一环境，避免污染系统 Python。
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    if ($SkipInstall) {
        throw "未找到 .venv，不能使用 -SkipInstall。请先不带 -SkipInstall 运行一次。"
    }
    $SystemPython = Resolve-SystemPython
    Write-Host "创建虚拟环境：$VenvDir" -ForegroundColor Yellow
    & $SystemPython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "创建虚拟环境失败。"
    }
}

if (-not $SkipInstall) {
    # 【依赖安装】requirements-render.txt 使用 CPU 版 PyTorch，适合没有 NVIDIA GPU 的本地 Web 推理。
    Write-Host "安装/更新 CPU 推理依赖，这一步首次运行可能需要几分钟……" -ForegroundColor Yellow
    Invoke-ProjectPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-ProjectPython -Arguments @("-m", "pip", "install", "-r", $RequirementsFile)
}

# 【Django 初始化】迁移 SQLite 历史记录表，并检查项目配置；这两步不会训练模型。
Write-Host "初始化 Django 数据库并检查配置……" -ForegroundColor Yellow
Invoke-ProjectPython -Arguments @("web/manage.py", "migrate", "--noinput")
Invoke-ProjectPython -Arguments @("web/manage.py", "check")

$browserHost = if ($BindAddress -eq "0.0.0.0") { "127.0.0.1" } else { $BindAddress }
$BrowserUrl = "http://$browserHost`:$Port/"
$BindTarget = "$BindAddress`:$Port"

if (-not $SkipBrowser) {
    # 【浏览器辅助】延迟两秒打开页面，让 Django 有时间完成启动；-SkipBrowser 可关闭此行为。
    $browserCommand = "Start-Sleep -Seconds 2; Start-Process '$BrowserUrl'"
    Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -ArgumentList @(
        "-NoProfile",
        "-Command",
        $browserCommand
    ) | Out-Null
}

# 【Web 服务启动】Django runserver 以前台方式运行，按 Ctrl+C 停止；默认仅本机访问。
Write-Host "服务地址：$BrowserUrl" -ForegroundColor Green
Write-Host "按 Ctrl+C 停止服务。" -ForegroundColor DarkGray
Invoke-ProjectPython -Arguments @("web/manage.py", "runserver", $BindTarget)
