#!/bin/bash
# filepath: /home/stud/bbo/projects/img23d/img23d/visual_prompting_alignment/batch_create_overlay.sh

# Configuration
RENDER_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/render_output"
PROJECTION_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/projection_output"
CLUSTERING_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/clustering_output"
OVERLAY_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/overlay_output"
TARGET_CLUSTER="05"  # Match the cluster number from projection

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "=============================================="
echo "Batch Overlay Creation Pipeline"
echo "Target cluster: $TARGET_CLUSTER"
echo "=============================================="

mkdir -p "$OVERLAY_OUTPUT"

processed_count=0
total_count=0

# Process each object class
for class_dir in "$PROJECTION_OUTPUT"/*; do
    if [ -d "$class_dir" ]; then
        object_class=$(basename "$class_dir")
        echo -e "${YELLOW}Processing class: $object_class${NC}"
        
        # Process each object in the class
        for object_dir in "$class_dir"/*; do
            if [ -d "$object_dir" ]; then
                object_name=$(basename "$object_dir")
                
                # Check required directories
                rgb_dir="$RENDER_OUTPUT/$object_class/$object_name"
                cluster_dir="$object_dir"
                label_file="$CLUSTERING_OUTPUT/$object_class/$object_name/cluster_${TARGET_CLUSTER}.npy"
                
                if [ ! -d "$rgb_dir" ]; then
                    echo "  Skipping $object_name (no RGB data)"
                    continue
                fi
                
                if [ ! -f "$label_file" ]; then
                    echo "  Skipping $object_name (no cluster labels)"
                    continue
                fi
                
                total_count=$((total_count + 1))
                echo -e "${BLUE}  Processing object: $object_name${NC}"
                
                # Set output directory
                overlay_dir="$OVERLAY_OUTPUT/$object_class/$object_name"
                
                echo "    → Creating overlay images"
                python test_overlay.py \
                    --rgb_dir "$rgb_dir" \
                    --cluster_dir "$cluster_dir" \
                    --output_path "$overlay_dir" \
                    --label_path "$label_file"
                
                if [ $? -eq 0 ]; then
                    processed_count=$((processed_count + 1))
                    echo -e "    ${GREEN}✓ Overlay creation completed${NC}"
                else
                    echo -e "    ${RED}✗ Failed to create overlay${NC}"
                fi
            fi
        done
        echo ""
    fi
done

echo "=============================================="
echo "Batch overlay creation completed!"
echo "Processed: $processed_count/$total_count objects"
echo "Results saved in: $OVERLAY_OUTPUT"
echo "=============================================="