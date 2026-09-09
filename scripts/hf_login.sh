#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <HUGGINGFACE_TOKEN>"
  echo "Example: $0 ghp_..."
  exit 1
fi

TOKEN="$1"
mkdir -p "$HOME/.huggingface"
echo "$TOKEN" > "$HOME/.huggingface/token"
chmod 600 "$HOME/.huggingface/token"

echo "Saved Hugging Face token to $HOME/.huggingface/token"
echo "Now run: hf auth whoami"