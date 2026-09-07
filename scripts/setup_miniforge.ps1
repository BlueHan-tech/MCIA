param(
    [ValidateSet("cpu", "gpu")]
    [string]$Variant = "cpu",
    [string]$EnvName = "a-king",
    [string]$InstallDir = "$env:USERPROFILE\miniforge3"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Installer = Join-Path $env:TEMP ("Miniforge3-Windows-x86_64-{0}.exe" -f $PID)
$MiniforgeUrl = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
$CondaBat = Join-Path $InstallDir "Scripts\conda.exe"

if (-not (Test-Path $CondaBat)) {
    Write-Host "Downloading Miniforge installer..."
    Invoke-WebRequest -Uri $MiniforgeUrl -OutFile $Installer

    Write-Host "Installing Miniforge to $InstallDir ..."
    Start-Process -FilePath $Installer -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$InstallDir" -Wait
}

if (-not (Test-Path $CondaBat)) {
    throw "Conda was not found at $CondaBat after installation."
}

$EnvFile = if ($Variant -eq "gpu") {
    Join-Path $ProjectRoot "environment-gpu.yml"
} else {
    Join-Path $ProjectRoot "environment.yml"
}

Write-Host "Creating/updating conda environment '$EnvName' from $EnvFile ..."
& $CondaBat env update --name $EnvName --file $EnvFile --prune

Write-Host ""
Write-Host "Done. Activate it with:"
Write-Host "  $InstallDir\Scripts\activate"
Write-Host "  conda activate $EnvName"
Write-Host ""
Write-Host "Quick check:"
Write-Host "  python -c `"import torch, numpy, scipy, pandas, yaml, sklearn, matplotlib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())`""
