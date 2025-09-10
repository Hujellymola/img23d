MODEL_PATH="$HOME/Downloads/selected_aligned_glb/pitcher/pitcher_928_4647894.glb"
MODEL_BASENAME=$(basename "$MODEL_PATH")
MODEL_NAME="${MODEL_BASENAME%.*}"
BLENDER_OUT="$HOME/UAD/output/test_render/pitcher/${MODEL_NAME}"
CLUSTER_OUT="$HOME/UAD/result/single_pca/${MODEL_NAME}"
python visualize.py \
    --input "$BLENDER_OUT" \
    --output "$CLUSTER_OUT"