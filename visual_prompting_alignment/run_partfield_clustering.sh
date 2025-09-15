
DATA_DIR="/home/stud/bbo/projects/img23d/data"
PARTFIELD_DIR="/home/stud/bbo/projects/img23d/dependencies/PartField"
OUTPUT_DIR="/home/stud/bbo/projects/img23d/img23d/output/clustering_output"
MAX_CLUSTERS=10
sample_name="stapler"



cd "$PARTFIELD_DIR" || { echo "PartField directory not found! Exiting."; exit 1; }



# ################# for mesh samples ###################
## ---------- feature extraction -----------
# python partfield_inference.py \
# -c configs/final/demo.yaml \
# --opts continue_ckpt model/model_objaverse.ckpt \
# result_name partfield_features/objaverse \
# dataset.data_path data/objaverse_samples
## ----------- part clustering -----------
# python run_part_clustering.py \
# --root exp_results/partfield_features/objaverse \
# --dump_dir exp_results/clustering/objaverse \
# --source_dir data/objaverse_samples \
# --use_agglo True \
# --max_num_clusters 20 \
# --option 0

# ################### for splat samples ####################
# ---------- feature extraction -----------
python partfield_inference.py \
    -c configs/final/demo.yaml \
    --opts continue_ckpt model/model_objaverse.ckpt \
    result_name partfield_features/splat \
    dataset.data_path $DATA_DIR/pcd \
    is_pc True
# ---------- part clustering -----------
python run_part_clustering.py \
    --root exp_results/partfield_features/splat \
    --dump_dir exp_results/clustering/splat \
    --source_dir $DATA_DIR/pcd \
    --max_num_clusters $MAX_CLUSTERS \
    --is_pc True

# ------------------- organize results --------------------
for pcd in $DATA_DIR/pcd/*.ply; do
    if [ -f "$pcd" ]; then
        obj_name=$(basename "$pcd" .ply)
        mkdir -p "$OUTPUT_DIR/$obj_name"

        cp exp_results/clustering/splat/cluster_out/${obj_name}_*.npy "$OUTPUT_DIR/$obj_name/"
        cp exp_results/clustering/splat/ply/${obj_name}_*.ply "$OUTPUT_DIR/$obj_name/"

    fi
done