
OBJECT_CLASS="old_swiss_knife"
CLUSTERING_DIR="/home/stud/bbo/projects/img23d/img23d/output/clustering_output/$OBJECT_CLASS"
OVERLAY_DIR="/home/stud/bbo/projects/img23d/img23d/output/overlay_output/$OBJECT_CLASS"
OUTPUT_DIR="/home/stud/bbo/projects/img23d/img23d/output/scene_graph_output/$OBJECT_CLASS"
RGB_DIR="/home/stud/bbo/projects/img23d/img23d/output/render_output/$OBJECT_CLASS"
NUM_CLUSTERS=06

echo "Processing object: $OBJECT_CLASS"

cluster_ply=$(find "$CLUSTERING_DIR" -name "*_${NUM_CLUSTERS}.ply" -print -quit)
cluster_npy=$(find "$CLUSTERING_DIR" -name "*_${NUM_CLUSTERS}.npy" -print -quit)

python kinematic_infer_hier.py \
    --points_ply "$cluster_ply" \
    --labels_npy "$cluster_npy" \
    --overlay_dir "$OVERLAY_DIR" \
    --rgb_dir "$RGB_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model gpt-4o