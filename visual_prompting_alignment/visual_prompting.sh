BLENDER_OUT="$HOME/UAD/output/test_render/pitcher/${MODEL_NAME}"
CLUSTER_OUT="$HOME/UAD/result/single_pca/${MODEL_NAME}"
OVERLAY_OUT="$HOME/UAD/result/overlay_test/${MODEL_NAME}"
ACTION_NAME="pour"
python new_api.py \
   --rgb_path "$BLENDER_OUT" \
   --overlay_root "$OVERLAY_OUT" \
   --action "$ACTION_NAME" \
   --output_path "$CLUSTER_OUT"