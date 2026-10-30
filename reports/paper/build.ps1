param([switch]$Figures, [switch]$Validate)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if ($Figures) {
        uv run --no-sync python scripts/prepare.py
        if ($LASTEXITCODE -ne 0) { throw 'Evidence validation or figure generation failed.' }
    }
    latexmk -lualatex -interaction=nonstopmode -halt-on-error -file-line-error '-outdir=.build' main.tex
    if ($LASTEXITCODE -ne 0) { throw 'LuaLaTeX compilation failed.' }
    Copy-Item -LiteralPath '.build/main.pdf' -Destination 'main.pdf' -Force
    if ($Validate) {
        uv run --no-project --with pymupdf python scripts/validate_document.py
        if ($LASTEXITCODE -ne 0) { throw 'Document validation failed.' }
    }
} finally {
    Pop-Location
}
