#!/usr/bin/env bash
# Build the passkey-protected dataset archive used by the Colab notebooks.
#
# Usage (from the repository root, in Git Bash on Windows or a Linux/macOS shell):
#   bash scripts/encrypt_dataset.sh [output_file]
#
# The archive contains data/5_355/raw/* and patient_info_anonymized.csv renamed to
# patient_info.csv: real patient IDs are never included.
# Upload the output file to Google Drive, share it as "anyone with the link", and put
# its file ID in DATA_FILE_ID in the Colab notebooks. Give the passkey to the students
# outside the repository.
#
# The encryption parameters must match the decryption cell of the Colab notebooks.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_DIR/data/5_355"
OUTPUT="${1:-$REPO_DIR/tmi-dataset-5_355.tar.gz.enc}"

for path in "$DATA_DIR/raw" "$DATA_DIR/patient_info_anonymized.csv"; do
    if [ ! -e "$path" ]; then
        echo "Missing $path" >&2
        exit 1
    fi
done

# Passkey: prompt without echo (TMI_PASSKEY can be set for non-interactive use)
if [ -z "${TMI_PASSKEY:-}" ]; then
    read -r -s -p "Passkey: " TMI_PASSKEY; echo
    read -r -s -p "Repeat passkey: " passkey_check; echo
    if [ "$TMI_PASSKEY" != "$passkey_check" ]; then
        echo "The passkeys do not match." >&2
        exit 1
    fi
fi
if [ "${#TMI_PASSKEY}" -lt 12 ]; then
    echo "Use a passkey of at least 12 characters." >&2
    exit 1
fi
export TMI_PASSKEY

STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT

mkdir -p "$STAGING_DIR/raw"
cp "$DATA_DIR"/raw/* "$STAGING_DIR/raw/"
cp "$DATA_DIR/patient_info_anonymized.csv" "$STAGING_DIR/patient_info.csv"

echo "Creating the encrypted archive..."
tar -czf - -C "$STAGING_DIR" patient_info.csv raw \
    | openssl enc -aes-256-cbc -pbkdf2 -iter 600000 -md sha256 -salt \
        -pass env:TMI_PASSKEY -out "$OUTPUT"

echo "Done: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
