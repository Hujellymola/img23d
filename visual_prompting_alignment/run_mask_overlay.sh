MODEL_NAME="robot_arm"
BLENDER_OUT="$HOME/projects/img23d/img23d/output/render_output/${MODEL_NAME}"
CLUSTER_PROJECTION_OUT="$HOME/projects/img23d/img23d/output/projection_output/${MODEL_NAME}"
OVERLAY_OUT="$HOME/projects/img23d/img23d/output/overlay_output/${MODEL_NAME}"
CLUSTERING_NPY="$HOME/projects/img23d/dependencies/PartField/exp_results/clustering/splat/cluster_out/${MODEL_NAME}/sample_pointcloud_0_06.npy"

python test_overlay.py \
    --rgb_dir $BLENDER_OUT \
    --cluster_dir $CLUSTER_PROJECTION_OUT \
    --output_path $OVERLAY_OUT \
    --label_path $CLUSTERING_NPY