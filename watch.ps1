[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
$paperRoot = Join-Path $repoRoot 'paper'
$pdfRoot = Join-Path $repoRoot 'output\pdf'
$auxRoot = Join-Path $repoRoot 'output\aux'
$mainFile = Join-Path $paperRoot 'main.tex'

if (-not (Test-Path -LiteralPath $mainFile)) {
    throw "LaTeX entry file not found: $mainFile"
}

$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
if (-not $latexmk) {
    throw 'latexmk was not found. Install MiKTeX (including XeLaTeX and latexmk), then reopen the terminal.'
}

New-Item -ItemType Directory -Force -Path $pdfRoot, $auxRoot | Out-Null

$oldLcAll = $env:LC_ALL
$oldLang = $env:LANG
$env:LC_ALL = 'C'
$env:LANG = 'C'

Push-Location $paperRoot
try {
    Write-Host 'Watching LaTeX sources. Press Ctrl+C to stop.' -ForegroundColor Cyan
    & $latexmk.Source `
        -pvc `
        '-view=none' `
        -xelatex `
        '-interaction=nonstopmode' `
        '-file-line-error' `
        '-halt-on-error' `
        '-outdir=../output/pdf' `
        '-auxdir=../output/aux' `
        'main.tex'
}
finally {
    Pop-Location
    $env:LC_ALL = $oldLcAll
    $env:LANG = $oldLang
}
