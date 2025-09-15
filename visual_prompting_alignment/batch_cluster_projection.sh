#!/bin/bash
# filepath: /home/stud/bbo/projects/img23d/img23d/visual_prompting_alignment/batch_cluster_projection.sh

# Configuration
CLUSTERING_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/clustering_output"
RENDER_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/render_output"
PROJECTION_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/projection_output"
TARGET_CLUSTER="05"  # Only process cluster_05 results
N_VIEWS=8

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "=============================================="
echo "Batch Cluster Projection Pipeline"
echo "Target cluster: $TARGET_CLUSTER"
echo "=============================================="

mkdir -p "$PROJECTION_OUTPUT"

processed_count=0
total_count=0

# Process each object class
for class_dir in "$CLUSTERING_OUTPUT"/*; do
    if [ -d "$class_dir" ]; then
        object_class=$(basename "$class_dir")
        echo -e "${YELLOW}Processing class: $object_class${NC}"
        
        # Process each object in the class
        for object_dir in "$class_dir"/*; do
            if [ -d "$object_dir" ]; then
                object_name=$(basename "$object_dir")
                
                # Check if render data exists
                render_dir="$RENDER_OUTPUT/$object_class/$object_name"
                if [ ! -d "$render_dir" ]; then
                    echo "  Skipping $object_name (no render data)"
                    continue
                fi
                
                # Check if target cluster files exist
                cluster_ply="$object_dir/cluster_${TARGET_CLUSTER}.ply"
                cluster_npy="$object_dir/cluster_${TARGET_CLUSTER}.npy"
                
                if [ ! -f "$cluster_ply" ] || [ ! -f "$cluster_npy" ]; then
                    echo "  Skipping $object_name (missing cluster_${TARGET_CLUSTER} files)"
                    continue
                fi
                
                total_count=$((total_count + 1))
                echo -e "${BLUE}  Processing object: $object_name${NC}"
                
                # Set output directory
                projection_dir="$PROJECTION_OUTPUT/$object_class/$object_name"
                
                echo "    → Projecting cluster $TARGET_CLUSTER"
                python sample_color_from_cluster.py \
                    --input "$render_dir" \
                    --output "$projection_dir" \
                    --points_ply "$cluster_ply" \
                    --clustering_npy "$cluster_npy" \
                    --n_views $N_VIEWS
                
                if [ $? -eq 0 ]; then
                    processed_count=$((processed_count + 1))
                    echo -e "    ${GREEN}✓ Cluster projection completed${NC}"
                else
                    echo -e "    ${RED}✗ Failed to project cluster${NC}"
                fi
            fi
        done
        echo ""
    fi
done

echo "=============================================="
echo "Batch cluster projection completed!"
echo "Processed: $processed_count/$total_count objects"
echo "Results saved in: $PROJECTION_OUTPUT"
echo "=============================================="