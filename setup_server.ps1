$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

if (-not (Test-Path '.env') -and (Test-Path '.env.example')) {
    Copy-Item '.env.example' '.env'
    Write-Host '[server] .env created from .env.example'
}

if (-not (Test-Path '.venv')) {
    py -3 -m venv .venv
    Write-Host '[server] virtual environment created'
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ''
Write-Host '[server] setup complete'
Write-Host '[server] activate: .\.venv\Scripts\Activate.ps1'
Write-Host '[server] run: python -m uvicorn main:app --reload'
