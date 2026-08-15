#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${VERSION:-$(tr -d '[:space:]' < "$ROOT/VERSION")}" 
DIST="$ROOT/dist"
STAGE="$DIST/MangaHDTransferStudio-Linux"
ARCHIVE="$DIST/MangaHDTransfer_${VERSION}_Linux_x86_64.tar.gz"
TAR="$DIST/_tracked-source.tar"

rm -rf "$STAGE" "$ARCHIVE" "$TAR"
mkdir -p "$STAGE"
cd "$ROOT"
git archive --format=tar HEAD -o "$TAR"
tar -xf "$TAR" -C "$STAGE"
rm -f "$TAR"
rm -rf "$STAGE/.github" "$STAGE/tests" "$STAGE/.gitignore"
chmod +x "$STAGE/启动Linux.sh" "$STAGE/packaging/launcher_linux.sh" || true

for required in LICENSE THIRD_PARTY_NOTICES.md REFERENCES.md CITATION.cff; do
  test -f "$STAGE/$required" || {
    echo "Required license/attribution file missing from release stage: $required" >&2
    exit 1
  }
done

cat > "$STAGE/RELEASE-README.txt" <<TXT
Manga HD Transfer Studio ${VERSION} - Linux x86_64
=================================================

1. Extract the archive to a normal writable directory.
2. Run: bash ./启动Linux.sh
3. A compatible Python 3.11-3.13 is used when available. Otherwise a verified
   standalone Python runtime is downloaded to the user's data directory.
4. Only main GUI/runtime dependencies are installed at startup.
5. OCR/ML dependencies and model weights remain on-demand and user-confirmed.

License and attribution:
- Manga HD Transfer Studio original code/documentation: MIT (LICENSE).
- Third-party software keeps its own license; see THIRD_PARTY_NOTICES.md.
- Implementation/research references: REFERENCES.md.
- Repository citation metadata: CITATION.cff.

PySide6 may require common desktop/XCB libraries supplied by your distribution.
The GitHub release workflow validates the package on Ubuntu x86_64.

Privacy: the package is created from git archive in a clean GitHub Actions
checkout and excludes developer caches, credentials, .env files, model weights,
logs, databases, user manga, generated outputs and local virtual environments.
TXT

tar -C "$DIST" -czf "$ARCHIVE" "$(basename "$STAGE")"
echo "Created $ARCHIVE"
