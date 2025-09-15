MODEL_NAME="robot_arm"
BLENDER_OUT="$HOME/projects/img23d/img23d/output/render_output/${MODEL_NAME}"
OVERLAY_OUT="$HOME/projects/img23d/img23d/output/overlay_output/${MODEL_NAME}"
ACTION_NAME="drill"
# ROLE_NAME="receiver"  # for "hold", "grasp", "lift"
ROLE_NAME="actor"      # for "pour", "cut", "scoop"
OUTPUT="$HOME/projects/img23d/img23d/output/visual_prompting_output/${MODEL_NAME}/${ACTION_NAME}_${ROLE_NAME}"
python new_api.py \
   --rgb_path "$BLENDER_OUT" \
   --overlay_root "$OVERLAY_OUT" \
   --action "$ACTION_NAME" \
   --role "$ROLE_NAME" \
   --output_path "$OUTPUT"