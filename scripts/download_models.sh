#!/usr/bin/env bash
set -euo pipefail

: "${SD15_DIR:=/models/sd15}"
: "${FLUX_DIR:=/models/flux-schnell}"
: "${SD15_REPO:=runwayml/stable-diffusion-v1-5}"
: "${FLUX_REPO:=black-forest-labs/FLUX.1-schnell}"

mkdir -p "$SD15_DIR" "$FLUX_DIR"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Logging into Hugging Face CLI..."
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

echo "Downloading SD1.5 to $SD15_DIR ..."
huggingface-cli download "$SD15_REPO" \
  --local-dir "$SD15_DIR" --local-dir-use-symlinks False

echo "Downloading FLUX.1-schnell to $FLUX_DIR ..."
huggingface-cli download "$FLUX_REPO" \
  --local-dir "$FLUX_DIR" --local-dir-use-symlinks False

echo "All models downloaded. You can set HF_HUB_OFFLINE=1 in future runs."
