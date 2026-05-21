#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="./data"
DATASET_ID="jessicali9530/celeba-dataset"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI is required. Install it with: pip install kaggle"
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required. Install it with your package manager."
  exit 1
fi

mkdir -p "$DATA_DIR"

shopt -s nullglob
existing_images=("$DATA_DIR"/*.jpg)

if (( ${#existing_images[@]} > 0 )); then
  echo "CelebA images already exist in $DATA_DIR; skipping download."
  exit 0
fi

# If a zip archive already exists in the data dir, don't re-download it.
ZIP_FILE="$(find "$DATA_DIR" -maxdepth 1 -name '*.zip' | head -n 1 || true)"
if [[ -n "${ZIP_FILE:-}" ]]; then
  echo "Found existing archive: $ZIP_FILE — will not re-download."
else
  echo "Downloading CelebA into $DATA_DIR ..."
  kaggle datasets download -d "$DATASET_ID" -p "$DATA_DIR" --force
  ZIP_FILE="$DATA_DIR/celeba-dataset.zip"
fi

if [[ -z "${ZIP_FILE:-}" || ! -f "$ZIP_FILE" ]]; then
  echo "Could not find the CelebA zip file in $DATA_DIR."
  exit 1
fi

echo "Extracting archive..."
unzip -q "$ZIP_FILE" -d "$DATA_DIR"

SOURCE_DIR="$(find "$DATA_DIR" -type d -name 'img_align_celeba' | head -n 1)"
if [[ -z "${SOURCE_DIR:-}" ]]; then
  echo "Could not find img_align_celeba in the extracted archive."
  exit 1
fi

DATA_DIR_REAL="$(realpath "$DATA_DIR")"
SOURCE_DIR_REAL="$(realpath "$SOURCE_DIR")"

echo "Copying JPG files into $DATA_DIR ..."
if [[ "$SOURCE_DIR_REAL" == "$DATA_DIR_REAL" ]]; then
  echo "Archive extracted directly into $DATA_DIR; files are already in place. Skipping copy."
else
  # copy images so extracted folder and zip are preserved
  find "$SOURCE_DIR" -maxdepth 1 -type f -name '*.jpg' -print0 | while IFS= read -r -d '' image_file; do
    cp -n "$image_file" "$DATA_DIR/"
  done
fi

echo "Done. CelebA images are available in: $DATA_DIR"