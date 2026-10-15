# PowerShell Launcher cho Chunker & Bounding Box Inspector
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "   VIETCOMBANK AI BANKING ASSISTANT - CHUNKING & DLA INSPECTOR" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

# 1. Thu uv run
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "[*] Tim thay uv! Dang khoi chay qua: uv run python serve.py ..." -ForegroundColor Cyan
    uv run python "$scriptDir\serve.py"
    exit
}

# 2. Thu .venv
$venvPython = "$scriptDir\..\..\..\.venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Host "[*] Tim thay Python .venv! Dang khoi chay..." -ForegroundColor Cyan
    & $venvPython "$scriptDir\serve.py"
    exit
}

# 3. Thu python3.14
if (Get-Command python3.14 -ErrorAction SilentlyContinue) {
    Write-Host "[*] Tim thay python3.14! Dang khoi chay..." -ForegroundColor Cyan
    python3.14 "$scriptDir\serve.py"
    exit
}

# 4. Fallback mo truc tiep index.html tren trinh duyet
Write-Host "[!] Khong tim thay Python. Dang mo truc tiep tren trinh duyet..." -ForegroundColor Yellow
Start-Process "$scriptDir\index.html"
