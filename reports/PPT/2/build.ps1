param([switch]$Figures)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if ($Figures) {
        uv run --no-sync python scripts/generate_figures.py
        if ($LASTEXITCODE -ne 0) { throw 'Figure generation failed.' }
    }
    latexmk -lualatex -interaction=nonstopmode -halt-on-error -file-line-error '-outdir=.build' main.tex
    if ($LASTEXITCODE -ne 0) { throw 'LuaLaTeX compilation failed.' }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '.build/main.pdf') -Destination (Join-Path $PSScriptRoot 'main.pdf') -Force
} finally {
    Pop-Location
}
