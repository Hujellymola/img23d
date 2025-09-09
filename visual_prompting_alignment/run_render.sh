#!/usr/bin/env zsh
set -e
export BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
MODEL_DIR="$HOME/RCI/Misc/img23d/input_glb"
BLENDER_OUT_BASE="$HOME/RCI/Misc/img23d/render_output"

for MODEL_PATH in "$MODEL_DIR"/*.glb; do
  MODEL_BASENAME=$(basename "$MODEL_PATH")
  MODEL_NAME="${MODEL_BASENAME%.*}"
  BLENDER_OUT="${BLENDER_OUT_BASE}/${MODEL_NAME}"
  echo "Processing $MODEL_NAME ..."
  $BLENDER -b -P render_views_final.py -- --model "$MODEL_PATH" --out "$BLENDER_OUT"
done
