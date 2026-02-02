#!/usr/bin/env bash
# fetch.sh — Download external data sources for HCP
#
# Downloads go into sources/data/ (gitignored).
# Run from the repository root: ./sources/fetch.sh
#
# For data already in /usr/share/databases/reference/, we symlink
# rather than duplicate.

set -euo pipefail

DATA_DIR="$(dirname "$0")/data"
REFERENCE_DIR="/usr/share/databases/reference"

mkdir -p "$DATA_DIR/unicode"
mkdir -p "$DATA_DIR/encodings"

echo "HCP data fetch script"
echo "Target directory: $DATA_DIR"
echo ""

# --- Symlink reference data ---

if [ -d "$REFERENCE_DIR/kaikki" ]; then
    if [ ! -e "$DATA_DIR/kaikki" ]; then
        ln -s "$REFERENCE_DIR/kaikki" "$DATA_DIR/kaikki"
        echo "Linked: kaikki -> $REFERENCE_DIR/kaikki"
    else
        echo "Exists: kaikki"
    fi
fi

# --- Unicode data ---

UNICODE_BASE="https://www.unicode.org/Public/UCD/latest/ucd"

fetch_unicode() {
    local file="$1"
    if [ ! -f "$DATA_DIR/unicode/$file" ]; then
        echo "Fetching: unicode/$file"
        curl -sL "$UNICODE_BASE/$file" -o "$DATA_DIR/unicode/$file"
    else
        echo "Exists: unicode/$file"
    fi
}

# Core Unicode data
fetch_unicode "UnicodeData.txt"          # All codepoints: name, category, bidi, decomposition
fetch_unicode "Blocks.txt"               # Block ranges (Basic Latin, Cyrillic, CJK, etc.)
fetch_unicode "Scripts.txt"              # Script assignments per codepoint
fetch_unicode "PropertyValueAliases.txt" # Category/property name mappings

# --- Legacy encoding tables (from Unicode mapping files) ---

MAPPING_BASE="https://www.unicode.org/Public/MAPPINGS"

fetch_mapping() {
    local path="$1"
    local file
    file="$(basename "$path")"
    if [ ! -f "$DATA_DIR/encodings/$file" ]; then
        echo "Fetching: encodings/$file"
        curl -sL "$MAPPING_BASE/$path" -o "$DATA_DIR/encodings/$file"
    else
        echo "Exists: encodings/$file"
    fi
}

# ISO 8859 series
for i in 1 2 3 4 5 6 7 8 9 10 13 14 15 16; do
    fetch_mapping "ISO8859/8859-${i}.TXT"
done

# Windows code pages
fetch_mapping "VENDORS/MICSFT/WINDOWS/CP1250.TXT"
fetch_mapping "VENDORS/MICSFT/WINDOWS/CP1251.TXT"
fetch_mapping "VENDORS/MICSFT/WINDOWS/CP1252.TXT"
fetch_mapping "VENDORS/MICSFT/WINDOWS/CP1253.TXT"
fetch_mapping "VENDORS/MICSFT/WINDOWS/CP1254.TXT"
fetch_mapping "VENDORS/MICSFT/WINDOWS/CP1255.TXT"
fetch_mapping "VENDORS/MICSFT/WINDOWS/CP1256.TXT"
fetch_mapping "VENDORS/MICSFT/WINDOWS/CP1257.TXT"
fetch_mapping "VENDORS/MICSFT/WINDOWS/CP1258.TXT"

# EBCDIC (IBM)
fetch_mapping "VENDORS/MICSFT/EBCDIC/CP037.TXT"
fetch_mapping "VENDORS/MICSFT/EBCDIC/CP500.TXT"
fetch_mapping "VENDORS/MICSFT/EBCDIC/CP875.TXT"
fetch_mapping "VENDORS/MICSFT/EBCDIC/CP1026.TXT"

# KOI8
fetch_mapping "VENDORS/MISC/KOI8-R.TXT"

echo ""
echo "Done. See sources/README.md for source descriptions."
