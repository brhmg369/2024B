[CmdletBinding()]
param(
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = $PSScriptRoot
$paperRoot = Join-Path $repoRoot 'paper'
$pdfRoot = Join-Path $repoRoot 'output\pdf'
$auxRoot = Join-Path $repoRoot 'output\aux'
$mainFile = Join-Path $paperRoot 'main.tex'
$outputPdf = Join-Path $pdfRoot 'main.pdf'

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
    $commonArgs = @(
        '-xelatex',
        '-interaction=nonstopmode',
        '-file-line-error',
        '-halt-on-error',
        '-outdir=../output/pdf',
        '-auxdir=../output/aux'
    )

    if ($Clean) {
        & $latexmk.Source -C '-outdir=../output/pdf' '-auxdir=../output/aux' 'main.tex'
        if ($LASTEXITCODE -ne 0) {
            throw "LaTeX cleanup failed with exit code $LASTEXITCODE."
        }
        Write-Host 'LaTeX build products cleaned.' -ForegroundColor Green
        return
    }

    & $latexmk.Source @commonArgs 'main.tex'
    if ($LASTEXITCODE -ne 0) {
        throw "LaTeX build failed with exit code $LASTEXITCODE. See output\aux\main.log for details."
    }

    if (-not (Test-Path -LiteralPath $outputPdf)) {
        throw "LaTeX reported success but did not create: $outputPdf"
    }

    Write-Host "PDF ready: $outputPdf" -ForegroundColor Green
}
finally {
    Pop-Location
    $env:LC_ALL = $oldLcAll
    $env:LANG = $oldLang
}
