MODEL_NAME="robot_arm"
BLENDER_OUT="$HOME/projects/img23d/img23d/output/render_output/${MODEL_NAME}"
CLUSTER_PROJECTION_OUT="$HOME/projects/img23d/img23d/output/projection_output/${MODEL_NAME}"
ORIG_POINTS_PLY="$HOME/projects/img23d/dependencies/PartField/exp_results/clustering/splat/ply/${MODEL_NAME}/sample_pointcloud_0_06.ply"
CLUSTERING_NPY="$HOME/projects/img23d/dependencies/PartField/exp_results/clustering/splat/cluster_out/${MODEL_NAME}/sample_pointcloud_0_06.npy"

python sample_color_from_cluster.py \
    --input $BLENDER_OUT \
    --output $CLUSTER_PROJECTION_OUT \
    --points_ply $ORIG_POINTS_PLY \
    --clustering_npy $CLUSTERING_NPY \
    --n_views 8