#!/bin/bash
# Set the base paths
DATA_DIR="/home/stud/bbo/projects/img23d/data"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at $DATA_DIR"
    exit 1
fi

echo "Starting automated GLB subsampling process..."
echo "Data directory: $DATA_DIR"
echo "============================================"

# Counter for processed directories
processed_count=0
total_count=0

# Find all directories that contain GLB files
for object_dir in "$DATA_DIR/glb"/*; do
    if [ -d "$object_dir" ]; then
        object_name=$(basename "$object_dir")
        
        # Check if directory contains GLB files
        glb_count=$(find "$object_dir" -name "*.glb" | wc -l)
        
        if [ "$glb_count" -gt 0 ]; then
            total_count=$((total_count + 1))
            output_dir="${DATA_DIR}/pcd/${object_name}_pcd"
            
            echo "Processing object class: $object_name"
            echo "Found $glb_count GLB files in $object_dir"
            echo "Output directory: $output_dir"
            
            # Run the subsampling script
            python3 glb_subsample.py \
            "$object_dir" \
            "$output_dir" \
                --output_format pointcloud \
                --n_points 10000 \
                --max_faces 50000 \
                --seed 42
            
            if [ $? -eq 0 ]; then
                processed_count=$((processed_count + 1))
                echo "✓ Successfully processed $object_name"
            else
                echo "✗ Error processing $object_name"
            fi
            
            echo "--------------------------------------------"
        else
            echo "Skipping $object_name (no GLB files found)"
        fi
    fi
done

echo "============================================"
echo "Processing complete!"
echo "Processed $processed_count out of $total_count object classes"

# List all created pointcloud directories
echo ""
echo "Created pointcloud directories:"
for pcd_dir in "$DATA_DIR"/*_pcd; do
    if [ -d "$pcd_dir" ]; then
        pcd_count=$(find "$pcd_dir" -name "*.ply" | wc -l)
        echo "  $(basename "$pcd_dir"): $pcd_count PLY files"
    fi
done

echo ""
echo "Done!"