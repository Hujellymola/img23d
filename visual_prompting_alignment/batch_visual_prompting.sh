#!/bin/bash
# filepath: /home/stud/bbo/projects/img23d/img23d/visual_prompting_alignment/batch_visual_prompting.sh

# Configuration
RENDER_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/render_output"
OVERLAY_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/overlay_output"
CLUSTERING_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/clustering_output"
VP_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/visual_prompting_output"
CSV_OUTPUT="/home/stud/bbo/projects/img23d/img23d/output/csv_results"

# Define specific action-role combinations for containers
declare -a ACTION_ROLE_PAIRS=("pour:actor" "grasp:receiver")

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "=============================================="
echo "Batch Visual Prompting Pipeline (Containers)"
echo "Testing: pour-actor, grasp-receiver"
echo "=============================================="

mkdir -p "$VP_OUTPUT"
mkdir -p "$CSV_OUTPUT"

# Process each action-role combination
for pair in "${ACTION_ROLE_PAIRS[@]}"; do
    IFS=':' read -r action role <<< "$pair"
    
    echo -e "${CYAN}Processing action: $action, role: $role${NC}"
    
    csv_file="$CSV_OUTPUT/${action}_${role}_results.csv"
    
    # Create CSV header
    echo "class,object_name,part_idx,result" > "$csv_file"
    
    processed_count=0
    total_count=0
    
    # Process each object class
    for class_dir in "$OVERLAY_OUTPUT"/*; do
        if [ -d "$class_dir" ]; then
            object_class=$(basename "$class_dir")
            echo -e "${YELLOW}  Processing class: $object_class${NC}"
            
            # Process each object in the class
            for object_dir in "$class_dir"/*; do
                if [ -d "$object_dir" ]; then
                    object_name=$(basename "$object_dir")
                    
                    # Check if RGB data exists
                    rgb_dir="$RENDER_OUTPUT/$object_class/$object_name"
                    if [ ! -d "$rgb_dir" ]; then
                        continue
                    fi
                    
                    total_count=$((total_count + 1))
                    echo -e "${BLUE}    Processing object: $object_name${NC}"
                    
                    # Set output directory for this specific query
                    vp_output_dir="$VP_OUTPUT/$object_class/${object_name}/${action}_${role}"
                    mkdir -p "$vp_output_dir"
                    
                    echo "      → Running visual prompting for $action-$role"
                    python new_api.py \
                        --rgb_path "$rgb_dir" \
                        --overlay_root "$object_dir" \
                        --action "$action" \
                        --role "$role" \
                        --output_path "$vp_output_dir"
                    
                    if [ $? -eq 0 ] && [ -f "$vp_output_dir/result.json" ]; then
                        processed_count=$((processed_count + 1))
                        
                        # Parse results and add to CSV
                        python -c "
import json
import sys

# Load the results
with open('$vp_output_dir/result.json', 'r') as f:
    results = json.load(f)

# Write to CSV
for idx, result in enumerate(results):
    print(f'$object_class,$object_name,{idx},{str(result).lower()}')
" >> "$csv_file"
                        
                        echo -e "      ${GREEN}✓ Visual prompting completed${NC}"
                    else
                        echo -e "      ${RED}✗ Failed visual prompting${NC}"
                    fi
                fi
            done
        fi
    done
    
    echo -e "${CYAN}Completed $action-$role: $processed_count/$total_count objects${NC}"
    echo -e "${CYAN}Results saved to: $csv_file${NC}"
    echo ""
done

echo "=============================================="
echo "Batch visual prompting completed!"
echo "CSV results saved in: $CSV_OUTPUT"

# Show summary of generated CSV files
echo ""
echo "Generated CSV files:"
for csv_file in "$CSV_OUTPUT"/*.csv; do
    if [ -f "$csv_file" ]; then
        filename=$(basename "$csv_file")
        line_count=$(($(wc -l < "$csv_file") - 1))  # Subtract header
        echo "  $filename: $line_count entries"
    fi
done

echo ""
echo "Container evaluation completed!"
echo "Results:"
echo "  - pour_actor_results.csv: Parts that emit liquid (spouts, lips, nozzles)"
echo "  - grasp_receiver_results.csv: Parts that receive grip (handles, gripping areas)"