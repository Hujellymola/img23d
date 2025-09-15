import os
import json
import base64
import argparse
import itertools
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

def load_partfield_data(points_ply_path, labels_npy_path):
    """Load 3D points and clustering labels from PartField outputs."""
    # Load point cloud
    pcd = o3d.io.read_point_cloud(points_ply_path)
    points = np.asarray(pcd.points)
    
    # Load clustering labels
    labels = np.load(labels_npy_path).astype(int)
    
    return points, labels

def analyze_part_geometry(points, labels):
    """Analyze geometric properties of each part."""
    unique_labels = np.unique(labels)
    part_info = {}
    
    for label in unique_labels:
        label_py = int(label)
        mask = (labels.ravel() == label)
        part_points = points[mask]
        
        if len(part_points) < 10:  # Skip tiny parts
            continue
            
        # Compute geometric properties
        centroid = np.mean(part_points, axis=0)
        
        # Principal component analysis for orientation
        centered = part_points - centroid
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (largest first)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Compute bounding box
        bbox_min = np.min(part_points, axis=0)
        bbox_max = np.max(part_points, axis=0)
        bbox_size = bbox_max - bbox_min
        
        # Aspect ratios for shape analysis
        sorted_dims = np.sort(bbox_size)[::-1]
        aspect_ratio_1 = sorted_dims[0] / (sorted_dims[1] + 1e-8)
        aspect_ratio_2 = sorted_dims[1] / (sorted_dims[2] + 1e-8)
        
        part_info[label_py] = {
            'centroid': centroid.tolist(),
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist(),
            'bbox_size': bbox_size.tolist(),
            'volume_approx': float(np.prod(bbox_size)),
            'point_count': len(part_points),
            'principal_axes': eigenvecs.T.tolist(),
            'eigenvalues': eigenvals.tolist(),
            'aspect_ratios': [float(aspect_ratio_1), float(aspect_ratio_2)],
            'elongation': float(aspect_ratio_1),  # How elongated the part is
            'flatness': float(aspect_ratio_2)     # How flat the part is
        }
    
    return part_info

def compute_spatial_relationships(part_info):
    """Compute spatial relationships between parts."""
    relationships = []
    
    labels = list(part_info.keys())
    for i, j in itertools.combinations(labels, 2):
        part_i = part_info[i]
        part_j = part_info[j]
        
        # Distance between centroids
        cent_i = np.array(part_i['centroid'])
        cent_j = np.array(part_j['centroid'])
        distance = np.linalg.norm(cent_j - cent_i)
        
        # Relative size
        size_ratio = part_i['volume_approx'] / (part_j['volume_approx'] + 1e-8)
        
        # Check if parts are adjacent (simple proximity test)
        bbox_i_size = np.array(part_i['bbox_size'])
        bbox_j_size = np.array(part_j['bbox_size'])
        typical_size = (np.mean(bbox_i_size) + np.mean(bbox_j_size)) / 2
        is_adjacent = distance < typical_size * 1.5
        
        # Alignment analysis - check if principal axes are aligned
        axes_i = np.array(part_i['principal_axes'])
        axes_j = np.array(part_j['principal_axes'])
        
        # Find best axis alignment
        max_alignment = 0
        for axis_i in axes_i:
            for axis_j in axes_j:
                alignment = abs(np.dot(axis_i, axis_j))
                max_alignment = max(max_alignment, alignment)
        
        relationships.append({
            'part_i': int(i),
            'part_j': int(j),
            'distance': float(distance),
            'size_ratio': float(size_ratio),
            'is_adjacent': bool(is_adjacent),
            'axis_alignment': float(max_alignment),
            'part_i_elongation': float(part_i['elongation']),
            'part_j_elongation': float(part_j['elongation'])
        })
    
    return relationships

def b64img(path):
    """Encode image to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def select_best_views(overlay_dir, part_i, part_j, n_views=8):
    """Select the best views showing both parts clearly."""
    selected_images = []
    
    # For now, select first few views (could be improved with visibility analysis)
    for vid in range(min(n_views, 4)):  # Use first 4 views
        overlay_i_path = os.path.join(overlay_dir, f"index_{part_i}", f"overlay_{vid:02d}.png")
        overlay_j_path = os.path.join(overlay_dir, f"index_{part_j}", f"overlay_{vid:02d}.png")
        
        if os.path.exists(overlay_i_path) and os.path.exists(overlay_j_path):
            selected_images.extend([overlay_i_path, overlay_j_path])
    
    return selected_images

KINEMATIC_ANALYSIS_PROMPT = """You are a robotics expert analyzing spatial relationships between parts of a robotic object.

You are given multiple overlay images showing TWO highlighted parts from different viewpoints. Each pair of consecutive images shows the same two parts from the same viewpoint.

Your task: Determine the most likely KINEMATIC JOINT TYPE between these two parts based on their spatial arrangement, shape, and typical robotic joint configurations.

Joint Types:
1. **revolute**: One part rotates around a fixed axis relative to the other (hinge joint)
2. **prismatic**: One part slides along a fixed axis relative to the other (linear actuator)  
3. **fixed**: Parts are rigidly connected with no relative motion
4. **rigid_bundle**: Parts move together as a single rigid body
5. **ball**: Spherical joint allowing rotation in multiple axes
6. **cylindrical**: Combination of rotation and translation along same axis

Consider:
- Shape characteristics (elongated parts often indicate revolute joints)
- Connection points and interfaces between parts
- Typical robot joint configurations
- Size relationships between parts

Spatial Analysis Data:
{spatial_data}

Output EXACTLY this JSON format:
{{
  "joint_type": "revolute|prismatic|fixed|rigid_bundle|ball|cylindrical",
  "confidence": 0.85,
  "reasoning": "Brief explanation of visual and geometric evidence",
  "joint_axis_hint": [ax, ay, az],
  "connection_point": [x, y, z]
}}

Provide only the JSON, no additional text."""

def analyze_joint_with_vlm(client, model, obj_class, part_i, part_j, spatial_data, overlay_images):
    """Use VLM to analyze joint type between two parts."""
    
    content = [
        {
            "type": "text", 
            "text": KINEMATIC_ANALYSIS_PROMPT.format(spatial_data=json.dumps(spatial_data, indent=2))
        },
        {
            "type": "text",
            "text": f"Object: {obj_class}\nAnalyzing joint between part_{part_i} and part_{part_j}"
        }
    ]
    
    # Add overlay images
    for img_path in overlay_images:
        if os.path.exists(img_path):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64img(img_path)}"}
            })
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error in VLM analysis for parts {part_i}-{part_j}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Infer kinematic scene graph from PartField clustering")
    parser.add_argument("--points_ply", required=True, help="Path to clustered point cloud PLY file")
    parser.add_argument("--labels_npy", required=True, help="Path to clustering labels NPY file")
    parser.add_argument("--overlay_dir", required=True, help="Directory with overlay visualizations")
    parser.add_argument("--object_class", default="", help="Object class for context")
    parser.add_argument("--output_dir", required=True, help="Output directory for scene graph")
    parser.add_argument("--model", default="gpt-4o", help="VLM model to use")
    
    args = parser.parse_args()
    
    # Load environment
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(f"OPENAI_API_KEY not found at {env_path}")
    
    client = OpenAI()
    
    # Load PartField data
    print("Loading PartField clustering results...")
    points, labels = load_partfield_data(
        os.path.expanduser(args.points_ply),
        os.path.expanduser(args.labels_npy)
    )
    
    # Analyze part geometry
    print("Analyzing part geometry...")
    part_info = analyze_part_geometry(points, labels)
    print(f"Found {len(part_info)} valid parts")
    
    # Compute spatial relationships
    print("Computing spatial relationships...")
    relationships = compute_spatial_relationships(part_info)
    
    # Analyze joints with VLM
    print("Analyzing joints with VLM...")
    edges = []
    
    for rel in relationships:
        part_i, part_j = rel['part_i'], rel['part_j']
        
        if not rel['is_adjacent']:
            print(f"Parts {part_i}-{part_j}: Too distant, skipping")
            continue
            
        print(f"Analyzing joint: part_{part_i} <-> part_{part_j}")
        
        # Get overlay images
        overlay_images = select_best_views(args.overlay_dir, part_i, part_j)
        
        if len(overlay_images) < 2:
            print(f"  Insufficient overlay images, skipping")
            continue
        
        # Analyze with VLM
        vlm_result = analyze_joint_with_vlm(
            client, args.model, args.object_class, 
            part_i, part_j, rel, overlay_images
        )
        
        if vlm_result:
            edge = {
                "source": f"part_{part_i}",
                "target": f"part_{part_j}",
                "type": vlm_result.get("joint_type", "fixed"),
                "confidence": vlm_result.get("confidence", 0.5),
                "reasoning": vlm_result.get("reasoning", ""),
                "spatial_metrics": rel,
                "overlay_paths": {
                    "source": os.path.join(args.overlay_dir, f"index_{part_i}"),
                    "target": os.path.join(args.overlay_dir, f"index_{part_j}")
                }
            }
            
            # Add joint-specific parameters
            if "joint_axis_hint" in vlm_result:
                edge["axis_hint"] = vlm_result["joint_axis_hint"]
            if "connection_point" in vlm_result:
                edge["connection_point"] = vlm_result["connection_point"]
                
            edges.append(edge)
            print(f"  -> {edge['type']} (confidence: {edge['confidence']:.2f})")
        else:
            print(f"  -> VLM analysis failed")
    
    # Build scene graph
    nodes = []
    for part_id in part_info.keys():
        node = {
            "id": f"part_{part_id}",
            "index": int(part_id),
            "geometry": part_info[part_id],
            "overlay_path": os.path.join(args.overlay_dir, f"index_{part_id}")
        }
        nodes.append(node)
    
    scene_graph = {
        "object_class": args.object_class,
        "analysis_type": "kinematic_spatial",
        "points_source": args.points_ply,
        "labels_source": args.labels_npy,
        "overlay_dir": args.overlay_dir,
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "total_parts": len(nodes),
            "total_joints": len(edges),
            "joint_types": {edge["type"]: 1 for edge in edges}
        }
    }
    
    # Count joint types
    joint_counts = {}
    for edge in edges:
        joint_type = edge["type"]
        joint_counts[joint_type] = joint_counts.get(joint_type, 0) + 1
    scene_graph["statistics"]["joint_types"] = joint_counts
    
    # Save results
    os.makedirs(os.path.expanduser(args.output_dir), exist_ok=True)
    output_file = os.path.join(os.path.expanduser(args.output_dir), "kinematic_scene_graph.json")
    
    with open(output_file, "w") as f:
        json.dump(scene_graph, f, indent=2)
    
    print(f"\n✅ Kinematic scene graph saved to: {output_file}")
    print(f"   Parts: {len(nodes)}")
    print(f"   Joints: {len(edges)}")
    print(f"   Joint types: {joint_counts}")

if __name__ == "__main__":
    main()