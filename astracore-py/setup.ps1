# setup.ps1 — Install astracore Python bindings on Windows (PowerShell)
# Usage: .\setup.ps1 [-Release]
param([switch]$Release)

$ErrorActionPreference = 'Stop'

Write-Host "=== AstraCore Python Setup ===" -ForegroundColor Cyan
Write-Host ""

# 1. Check Rust / cargo
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "[1/4] Installing Rust via rustup-init.exe..." -ForegroundColor Yellow
    $rustupUrl = 'https://win.rustup.rs/x86_64'
    $installer  = "$env:TEMP\rustup-init.exe"
    Invoke-WebRequest -Uri $rustupUrl -OutFile $installer
    & $installer -y --quiet
    # Reload PATH so cargo is found
    $env:PATH = [System.Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' +
                [System.Environment]::GetEnvironmentVariable('PATH', 'User')
} else {
    $cv = cargo --version
    Write-Host "[1/4] Rust/cargo found: $cv" -ForegroundColor Green
}

# 2. Check / install maturin
if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) {
    Write-Host "[2/4] Installing maturin..." -ForegroundColor Yellow
    pip install --quiet "maturin>=1,<2"
} else {
    $mv = maturin --version
    Write-Host "[2/4] maturin found: $mv" -ForegroundColor Green
}

# 3. Ensure a virtualenv is active (maturin develop requires one)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not $env:VIRTUAL_ENV -and -not $env:CONDA_PREFIX) {
    Write-Host "[3/4] No virtualenv active — creating .venv..." -ForegroundColor Yellow
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
    }
    & ".\.venv\Scripts\Activate.ps1"
    Write-Host "      Activated .venv" -ForegroundColor Green
} else {
    $activeEnv = if ($env:VIRTUAL_ENV) { $env:VIRTUAL_ENV } else { $env:CONDA_PREFIX }
    Write-Host "[3/4] Virtualenv active: $activeEnv" -ForegroundColor Green
}

# 4. Build and install
Write-Host "[4/4] Building astracore Python bindings$(if ($Release) {' (release mode)'})..." -ForegroundColor Yellow

if ($Release) {
    maturin develop --release
} else {
    maturin develop
}

Write-Host ""
Write-Host "Done! Test with:" -ForegroundColor Green
Write-Host '  python -c "import astracore as ac; c = ac.Circuit(2); c.h(0); c.cnot(0,1); c.measure_all(); print(c.run().bitstring())"'
