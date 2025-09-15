MODEL_NAME="paper_drawer"
CLUSTERS=05
BLENDER_OUT="$HOME/projects/img23d/img23d/output/render_output/${MODEL_NAME}"
CLUSTER_PROJECTION_OUT="$HOME/projects/img23d/img23d/output/projection_output/${MODEL_NAME}"
ORIG_POINTS_PLY="$HOME/projects/img23d/img23d/output/clustering_output/${MODEL_NAME}/${MODEL_NAME}_*_${CLUSTERS}.ply"
CLUSTERING_NPY="$HOME/projects/img23d/img23d/output/clustering_output/${MODEL_NAME}/${MODEL_NAME}_*_${CLUSTERS}.npy"

python sample_color_from_cluster.py \
    --input $BLENDER_OUT \
    --output $CLUSTER_PROJECTION_OUT \
    --points_ply $ORIG_POINTS_PLY \
    --clustering_npy $CLUSTERING_NPY \
    --n_views 8