<#
.SYNOPSIS
    Windows equivalent of the Makefile. Same commands, so local and CI cannot drift.

.EXAMPLE
    .\tasks.ps1 install
    .\tasks.ps1 all
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'install', 'lint', 'format', 'typecheck', 'test', 'frames',
                 'cov', 'all', 'serve', 'web', 'clean')]
    [string]$Task = 'help'
)

$ErrorActionPreference = 'Stop'

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Task) {
    'help' {
        Write-Host "Tasks: install lint format typecheck test frames cov all serve web clean"
        Write-Host "Activate the environment first:  conda activate sih26168"
    }
    'install' {
        Invoke-Step 'pip'        { python -m pip install --upgrade pip }
        Invoke-Step 'install'    { pip install -e ".[dev]" }
        Invoke-Step 'pre-commit' { pre-commit install }
    }
    'lint'      { Invoke-Step 'ruff check' { ruff check . } }
    'format'    { Invoke-Step 'ruff format' { ruff format . } }
    'typecheck' { Invoke-Step 'mypy' { mypy } }
    'test'      { Invoke-Step 'pytest' { pytest -q } }
    'frames'    { Invoke-Step 'frames' { pytest -m frames -v } }
    'cov'       { Invoke-Step 'coverage' { pytest -q --cov --cov-report=term-missing } }
    'all' {
        Invoke-Step 'ruff check' { ruff check . }
        Invoke-Step 'ruff format' { ruff format --check . }
        Invoke-Step 'mypy' { mypy }
        Invoke-Step 'pytest' { pytest -q }
    }
    'serve' { python -m services.gateway --host 0.0.0.0 --port 8000 }
    'web'   { Push-Location apps/web; try { npm run dev } finally { Pop-Location } }
    'clean' {
        foreach ($d in '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov') {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
        foreach ($f in 'coverage.xml', '.coverage') {
            if (Test-Path $f) { Remove-Item -Force $f }
        }
    }
}
