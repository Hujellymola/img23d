#!/bin/bash
# filepath: /home/stud/bbo/projects/img23d/img23d/visual_prompting_alignment/run_partfield_pipeline.sh

# Configuration
DATA_DIR="/home/stud/bbo/projects/img23d/data/pcd"
PARTFIELD_DIR="/home/stud/bbo/projects/img23d/dependencies/PartField"
OUTPUT_DIR="/home/stud/bbo/projects/img23d/img23d/output/clustering_output"
MAX_CLUSTERS=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PartField Feature Extraction & Clustering"
echo "=========================================="

# Check dependencies
if [ ! -d "$PARTFIELD_DIR" ]; then
    echo -e "${RED}Error: PartField directory not found at $PARTFIELD_DIR${NC}"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found at $DATA_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to PartField directory
cd "$PARTFIELD_DIR"

# Process each object class
for pcd_dir in "$DATA_DIR"/*_pcd; do
    if [ -d "$pcd_dir" ]; then
        # Extract object class name (remove _pcd suffix)
        object_class=$(basename "$pcd_dir" | sed 's/_pcd$//')
        
        echo -e "${YELLOW}Processing object class: $object_class${NC}"
        
        # Step 1: Feature extraction
        echo "  → Running feature extraction..."
        python partfield_inference.py \
            -c configs/final/demo.yaml \
            --opts \
            continue_ckpt model/model_objaverse.ckpt \
            result_name partfield_features/splat/$object_class \
            dataset.data_path "$pcd_dir" \
            is_pc True
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}  ✗ Feature extraction failed for $object_class${NC}"
            continue
        fi
        
        # Step 2: Clustering
        echo "  → Running clustering..."
        echo "=========================================="
        python run_part_clustering.py \
            --root exp_results/partfield_features/splat/$object_class \
            --dump_dir exp_results/clustering/splat/$object_class \
            --source_dir "$pcd_dir" \
            --max_num_clusters $MAX_CLUSTERS \
            --is_pc True
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}  ✗ Clustering failed for $object_class${NC}"
            continue
        fi
        
        # Step 3: Organize results
        echo "  → Organizing results..."
        echo "=========================================="
        class_output_dir="$OUTPUT_DIR/$object_class"
        mkdir -p "$class_output_dir"
        
        # Process clustering results
        cluster_dir="exp_results/clustering/splat/$object_class"
        ply_dir="$cluster_dir/ply"
        npy_dir="$cluster_dir/cluster_out"
        
        if [ -d "$ply_dir" ] && [ -d "$npy_dir" ]; then
            # Group files by object name
            for ply_file in "$ply_dir"/*.ply; do
                if [ -f "$ply_file" ]; then
                    filename=$(basename "$ply_file" .ply)
                    # Extract object name (remove _pointcloud_0_XX suffix)
                    object_name=$(echo "$filename" | sed 's/_pointcloud_0_[0-9][0-9]$//')
                    cluster_num=$(echo "$filename" | grep -o '[0-9][0-9]$')
                    
                    # Create object directory
                    object_output_dir="$class_output_dir/$object_name"
                    mkdir -p "$object_output_dir"
                    
                    # Copy files with cleaner names
                    cp "$ply_file" "$object_output_dir/cluster_${cluster_num}.ply"
                    
                    # Copy corresponding npy file
                    npy_file="$npy_dir/${filename}.npy"
                    if [ -f "$npy_file" ]; then
                        cp "$npy_file" "$object_output_dir/cluster_${cluster_num}.npy"
                    fi
                fi
            done
        fi
        
        echo -e "${GREEN}  ✓ Completed $object_class${NC}"
        echo "    Results saved to: $class_output_dir"
        
        # Show summary
        object_count=$(find "$class_output_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "    Objects processed: $object_count"
        echo "=========================================="
        echo "=========================================="
    fi
done

echo "=========================================="
echo "Pipeline completed!"
echo "Results organized in: $OUTPUT_DIR"
echo ""

# Show final summary
echo "Summary of processed object classes:"
for class_dir in "$OUTPUT_DIR"/*; do
    if [ -d "$class_dir" ]; then
        class_name=$(basename "$class_dir")
        object_count=$(find "$class_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
        total_files=$(find "$class_dir" -name "*.ply" | wc -l)
        echo "  $class_name: $object_count objects, $total_files cluster files"
    fi
done

echo ""
echo "Done!"