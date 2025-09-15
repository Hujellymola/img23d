#!/bin/bash
# filepath: /home/stud/bbo/projects/img23d/img23d/visual_prompting_alignment/run_kinematic_analysis.sh

OBJECT_CLASS="old_swiss_knife"
CLUSTERING_DIR="/home/stud/bbo/projects/img23d/img23d/output/clustering_output/$OBJECT_CLASS"
OVERLAY_DIR="/home/stud/bbo/projects/img23d/img23d/output/overlay_output/$OBJECT_CLASS"
OUTPUT_DIR="/home/stud/bbo/projects/img23d/img23d/output/kinematic_graphs/$OBJECT_CLASS"
RGB_DIR="/home/stud/bbo/projects/img23d/img23d/output/render_output/$OBJECT_CLASS"
NUM_CLUSTERS=06

echo "=============================================="
echo "Kinematic Scene Graph Analysis"
echo "Object class: $OBJECT_CLASS"
echo "=============================================="

# Check if the class directory structure exists
if [ ! -d "$CLUSTERING_DIR" ]; then
    echo "❌ Error: Clustering directory not found: $CLUSTERING_DIR"
    exit 1
fi

if [ ! -d "$OVERLAY_DIR" ]; then
    echo "❌ Error: Overlay directory not found: $OVERLAY_DIR"
    exit 1
fi

# Function to process a single object
process_object() {
    local object_path="$1"
    local object_name="$2"
    local overlay_path="$3"
    local output_path="$4"
    
    echo "Processing object: $object_name"
    
    cluster_ply=$(find "$object_path" -name "*_${NUM_CLUSTERS}.ply" -print -quit)
    cluster_npy=$(find "$object_path" -name "*_${NUM_CLUSTERS}.npy" -print -quit)

    if [ -z "$cluster_ply" ] || [ -z "$cluster_npy" ]; then
        echo "  Skipping $object_name (missing *_${NUM_CLUSTERS} files)"
        return 1
    fi
    
    if [ ! -d "$overlay_path" ]; then
        echo "  Skipping $object_name (no overlay data at $overlay_path)"
        return 1
    fi
    
    echo "  → Running kinematic analysis..."
    python infer_kinematic_constraints.py \
        --points_ply "$cluster_ply" \
        --labels_npy "$cluster_npy" \
        --overlay_dir "$overlay_path" \
        --rgb_dir "$rgb_dir" \
        --output_dir "$output_path" \
        # --use_rgb \
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Completed $object_name"
        return 0
    else
        echo "  ❌ Failed $object_name"
        return 1
    fi
}

# Modify the check for files in root
cluster_files_in_root=$(find "$CLUSTERING_DIR" -maxdepth 1 -name "*_${NUM_CLUSTERS}.ply" | wc -l)
subdirs_count=$(find "$CLUSTERING_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)


if [ "$cluster_files_in_root" -gt 0 ]; then
    # Case 1: Standalone object (class IS the object)
    echo "Detected standalone object structure"
    echo ""
    
    process_object "$CLUSTERING_DIR" "$OBJECT_CLASS" "$OVERLAY_DIR" "$OUTPUT_DIR"
    
elif [ "$subdirs_count" -gt 0 ]; then
    # Case 2: Multiple objects within the class
    echo "Detected multi-object class structure"
    echo "Found $subdirs_count objects in class $OBJECT_CLASS"
    echo ""
    
    # Process each object in the class
    processed_count=0
    total_count=0
    
    for object_dir in "$CLUSTERING_DIR"/*; do
        if [ -d "$object_dir" ]; then
            object_name=$(basename "$object_dir")
            overlay_obj_dir="$OVERLAY_DIR/$object_name"
            output_obj_dir="$OUTPUT_DIR/$object_name"
            
            total_count=$((total_count + 1))
            
            if process_object "$object_dir" "$object_name" "$overlay_obj_dir" "$output_obj_dir"; then
                processed_count=$((processed_count + 1))
            fi
            echo ""
        fi
    done
    
    echo "Processed $processed_count/$total_count objects"
    
else
    echo "❌ Error: No cluster files or subdirectories found in $CLUSTERING_DIR"
    echo "Expected either:"
    echo "  - cluster_*.ply files directly in $CLUSTERING_DIR (standalone object)"
    echo "  - subdirectories containing cluster_*.ply files (multi-object class)"
    exit 1
fi

echo "=============================================="
echo "Kinematic analysis completed!"
echo "Results saved in: $OUTPUT_DIR"

# Show summary
echo ""
echo "Summary:"
if [ -f "$OUTPUT_DIR/kinematic_scene_graph.json" ]; then
    # Standalone object case
    echo "  Generated: $OUTPUT_DIR/kinematic_scene_graph.json"
else
    # Multi-object case
    graph_count=$(find "$OUTPUT_DIR" -name "kinematic_scene_graph.json" 2>/dev/null | wc -l)
    echo "  Generated $graph_count scene graph files in $OUTPUT_DIR"
fi