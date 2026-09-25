# Run the chatbot locally using settings from .env
# Uses a dedicated Postgres container on 127.0.0.1:55433

# Usage (PowerShell from the repo root):
#   scripts/run_local.ps1 -Index    # first time or after a parser change, rebuilds the index
#   scripts/run_local.ps1           # start the API, open http://127.0.0.1:8000
param([switch]$Index)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (docker ps --filter "name=^fdr-app-pg$" --format "{{.Names}}")) {
    if (docker ps -a --filter "name=^fdr-app-pg$" --format "{{.Names}}") {
        docker start fdr-app-pg | Out-Null
    } else {
        docker run -d --name fdr-app-pg -v fdr-app-pgdata:/var/lib/postgresql/data `
            -e POSTGRES_USER=fdr -e POSTGRES_PASSWORD=fdr -e POSTGRES_DB=fdr `
            -p 127.0.0.1:55433:5432 pgvector/pgvector:0.8.1-pg16 | Out-Null
    }
    Start-Sleep -Seconds 5
}

# clear variables defined in .env so stale shell values cannot override them
if (-not (Test-Path .env)) { throw ".env not found in $(Get-Location)" }
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
        Remove-Item "Env:$($Matches[1])" -ErrorAction SilentlyContinue
    }
}
# Used only so python can import the application package from src/
$env:PYTHONPATH = "src"   

if ($Index) {
    .venv/Scripts/python.exe scripts/index_corpus.py --reset
    if ($LASTEXITCODE -ne 0) { throw "indexing failed" }
}
.venv/Scripts/python.exe -m uvicorn flight_delay.api:app --host 127.0.0.1 --port 8000
