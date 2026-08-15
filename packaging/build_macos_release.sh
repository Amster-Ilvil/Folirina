#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${VERSION:-$(tr -d '[:space:]' < "$ROOT/VERSION")}" 
DIST="$ROOT/dist"
STAGE="$DIST/MangaHDTransferStudio-macOS"
APP="$STAGE/Manga HD Transfer Studio.app"
CONTENTS="$APP/Contents"
APP_SOURCE="$CONTENTS/Resources/app"
ZIP="$DIST/MangaHDTransfer_${VERSION}_macOS_universal.zip"
DMG="$DIST/MangaHDTransfer_${VERSION}_macOS_universal.dmg"
TAR="$DIST/_tracked-source.tar"

rm -rf "$STAGE" "$ZIP" "$DMG" "$TAR"
mkdir -p "$CONTENTS/MacOS" "$APP_SOURCE"

# Privacy boundary: the embedded app source comes only from tracked Git files.
cd "$ROOT"
git archive --format=tar HEAD -o "$TAR"
tar -xf "$TAR" -C "$APP_SOURCE"
rm -f "$TAR"
rm -rf "$APP_SOURCE/.github" "$APP_SOURCE/tests" "$APP_SOURCE/.gitignore" "$APP_SOURCE/packaging"

# License boundary: all public macOS artifacts must expose the project license,
# third-party notices, implementation/research references, and citation metadata.
for required in LICENSE THIRD_PARTY_NOTICES.md REFERENCES.md CITATION.cff; do
  test -f "$APP_SOURCE/$required" || {
    echo "Required license/attribution file missing from release stage: $required" >&2
    exit 1
  }
  cp "$APP_SOURCE/$required" "$STAGE/$required"
done

cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>zh_CN</string>
  <key>CFBundleDisplayName</key><string>Manga HD Transfer Studio</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIdentifier</key><string>org.mangahd.transferstudio</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>Manga HD Transfer Studio</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
printf 'APPL????' > "$CONTENTS/PkgInfo"
cp "$ROOT/packaging/launcher_macos.zsh" "$CONTENTS/MacOS/launcher"
chmod +x "$CONTENTS/MacOS/launcher"

cat > "$STAGE/RELEASE-README.txt" <<TXT
Manga HD Transfer Studio ${VERSION} - macOS universal source/runtime shell
=======================================================================

- Supports Apple Silicon and Intel Macs. The launcher downloads a verified Python
  runtime matching the current Mac architecture on first launch.
- Startup installs only the main GUI/runtime dependencies.
- OCR/ML dependencies and model weights remain on-demand and user-confirmed.
- Apple Live Text helper source is included; when needed, the existing application
  flow can build it with Xcode Command Line Tools on supported macOS versions.

License and attribution:
- Manga HD Transfer Studio original code/documentation: MIT (LICENSE).
- Third-party software keeps its own license; see THIRD_PARTY_NOTICES.md.
- Implementation/research references: REFERENCES.md.
- Repository citation metadata: CITATION.cff.

Privacy: this application is assembled from git archive in a clean GitHub Actions
checkout. No developer cache, credentials, .env file, model cache, logs, user manga,
output workspace, database, or local virtual environment is included.
TXT

codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
plutil -lint "$CONTENTS/Info.plist"

/usr/bin/ditto -c -k --keepParent "$STAGE" "$ZIP"

DMGROOT="$DIST/_dmgroot"
rm -rf "$DMGROOT"
mkdir -p "$DMGROOT"
cp -R "$APP" "$DMGROOT/"
cp "$STAGE/RELEASE-README.txt" "$DMGROOT/"
for required in LICENSE THIRD_PARTY_NOTICES.md REFERENCES.md CITATION.cff; do
  cp "$STAGE/$required" "$DMGROOT/$required"
done
hdiutil create -volname "Manga HD Transfer Studio ${VERSION}" -srcfolder "$DMGROOT" -ov -format UDZO "$DMG" >/dev/null
hdiutil verify "$DMG" >/dev/null
rm -rf "$DMGROOT"

echo "Created $ZIP"
echo "Created $DMG"
