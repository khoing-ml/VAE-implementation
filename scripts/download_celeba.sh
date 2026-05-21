#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="${1:-data}"
DATASET_ID="jessicali9530/celeba-dataset"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}

trap cleanup EXIT

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI is required. Install it with: pip install kaggle"
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required. Install it with your package manager."
  exit 1
fi

mkdir -p "$DATA_DIR"

echo "Downloading CelebA to temporary directory..."
kaggle datasets download -d "$DATASET_ID" -p "$TMP_DIR" --force

ZIP_FILE="$TMP_DIR/celeba-dataset.zip"
if [[ ! -f "$ZIP_FILE" ]]; then
  ZIP_FILE="$(find "$TMP_DIR" -maxdepth 1 -name '*.zip' | head -n 1)"
fi

if [[ -z "${ZIP_FILE:-}" || ! -f "$ZIP_FILE" ]]; then
  echo "Could not find the downloaded CelebA zip file."
  exit 1
fi

echo "Extracting archive..."
unzip -q "$ZIP_FILE" -d "$TMP_DIR/extracted"

SOURCE_DIR="$(find "$TMP_DIR/extracted" -type d -name 'img_align_celeba' | head -n 1)"
if [[ -z "${SOURCE_DIR:-}" ]]; then
  echo "Could not find img_align_celeba in the extracted archive."
  exit 1
fi

SOURCE_DIR="$(find "$SOURCE_DIR" -type d -name 'img_align_celeba' | head -n 1)"
if [[ -z "${SOURCE_DIR:-}" ]]; then
  echo "Could not find the nested img_align_celeba image folder."
  exit 1
fi

echo "Copying JPG files into $DATA_DIR ..."
find "$SOURCE_DIR" -maxdepth 1 -type f -name '*.jpg' -print0 | while IFS= read -r -d '' image_file; do
  mv "$image_file" "$DATA_DIR/"
done

echo "Done. CelebA images are available in: $DATA_DIR"