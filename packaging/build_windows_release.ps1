param(
    [string]$Version = ""
)
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
if (-not $Version) {
    $Version = (Get-Content -LiteralPath (Join-Path $Root 'VERSION') -Raw).Trim()
}
if (-not $Version) { throw 'VERSION is empty.' }

$Dist = Join-Path $Root 'dist'
$Stage = Join-Path $Dist 'MangaHDTransferStudio-Windows'
$Archive = Join-Path $Dist ("MangaHDTransfer_{0}_Windows_x64.zip" -f $Version)
$TarPath = Join-Path $Dist '_tracked-source.tar'

Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $Archive, $TarPath -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Dist, $Stage | Out-Null

# Privacy boundary: package only files tracked by the exact Git commit.
Push-Location $Root
try {
    git archive --format=tar HEAD -o $TarPath
    if ($LASTEXITCODE -ne 0) { throw 'git archive failed.' }
    tar.exe -xf $TarPath -C $Stage
    if ($LASTEXITCODE -ne 0) { throw 'tar extraction failed.' }
} finally {
    Pop-Location
}
Remove-Item -LiteralPath $TarPath -Force -ErrorAction SilentlyContinue

foreach ($relative in @('.github', 'tests', '.gitignore')) {
    Remove-Item -LiteralPath (Join-Path $Stage $relative) -Recurse -Force -ErrorAction SilentlyContinue
}

@"
Manga HD Transfer Studio $Version - Windows x64
================================================

1. Extract this ZIP to a normal writable folder.
2. Double-click the Windows launcher batch file in the package root.
3. The launcher uses a compatible Python 3.11-3.13 installation if available;
   otherwise it downloads a verified standalone Python runtime.
4. Startup installs only the main GUI/runtime dependencies.
5. OCR/ML dependencies and model weights are never bundled and remain on-demand
   through the application's existing confirmation flow.

Privacy: this package is generated from git archive in a clean GitHub Actions
checkout. It contains no developer cache, .env file, credentials, model cache,
logs, databases, user manga, output workspace, or local virtual environment.
"@ | Set-Content -LiteralPath (Join-Path $Stage 'RELEASE-README.txt') -Encoding UTF8

Compress-Archive -Path (Join-Path $Stage '*') -DestinationPath $Archive -CompressionLevel Optimal
if (-not (Test-Path -LiteralPath $Archive)) { throw 'Windows release archive was not created.' }
Write-Host "Created $Archive"
