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
    pcd = o3d.io.read_point_cloud(points_ply_path)
    points = np.asarray(pcd.points)
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
            'elongation': float(aspect_ratio_1),
            'flatness': float(aspect_ratio_2)
        }
    
    return part_info

def compute_contact_relationships(part_info, contact_threshold_factor=1.2):
    """Determine which parts are in contact based on proximity."""
    contacts = []
    labels = list(part_info.keys())
    
    for i, j in itertools.combinations(labels, 2):
        part_i = part_info[i]
        part_j = part_info[j]
        
        # Distance between centroids
        cent_i = np.array(part_i['centroid'])
        cent_j = np.array(part_j['centroid'])
        distance = np.linalg.norm(cent_j - cent_i)
        
        # Estimate contact threshold based on part sizes
        size_i = np.mean(part_i['bbox_size'])
        size_j = np.mean(part_j['bbox_size'])
        contact_threshold = (size_i + size_j) * contact_threshold_factor
        
        is_contact = distance < contact_threshold
        
        if is_contact:
            contacts.append({
                'part_i': i,
                'part_j': j,
                'distance': float(distance),
                'contact_strength': float(contact_threshold / (distance + 1e-8))
            })
    
    return contacts

def b64img(path):
    """Encode image to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_overlay_images(overlay_dir, part_indices, n_views=8):
    """Get all overlay images for specified parts."""
    overlay_images = []
    
    for part_idx in part_indices:
        for vid in range(n_views):
            overlay_path = os.path.join(overlay_dir, f"index_{part_idx}", f"overlay_{vid:02d}.png")
            if os.path.exists(overlay_path):
                overlay_images.append(overlay_path)
    
    return overlay_images

def get_rgb_images(rgb_dir, n_views=8):
    """Get RGB reference images."""
    rgb_images = []
    for vid in range(n_views):
        rgb_path = os.path.join(rgb_dir, f"rgb_{vid:02d}.png")
        if os.path.exists(rgb_path):
            rgb_images.append(rgb_path)
    return rgb_images

# Updated prompts for hierarchical analysis
OBJECT_UNDERSTANDING_PROMPT = """You are a robotics expert analyzing a 3D object that has been segmented into parts.

You are given:
1. RGB images showing the complete object from 8 different viewpoints
2. Part segmentation data showing {num_parts} distinct parts
3. Spatial relationships between parts

Your task: Understand the object type and create a hierarchical grouping of parts.

Object Analysis Data:
- Number of parts: {num_parts}
- Part spatial data: {spatial_summary}
- Contact relationships: {contact_summary}

Instructions:
1. Identify what the given common object is
2. Group parts that should move together as rigid bodies (rigid_bundles)
3. Consider typical kinematic constraints for this object type

Output EXACTLY this JSON format:
{{
  "object_type": "object_class_name",
  "confidence": 0.0~1.0,
  "reasoning": "Brief explanation of object identification",
  "rigid_groups": [
    {{
      "group_id": "handle",
      "part_indices": [1, 2, 3],
      "description": "Handle assembly that moves as one piece"
    }},
    {{
      "group_id": "blade", 
      "part_indices": [0],
      "description": "Main cutting blade"
    }}
  ]
}}

Provide only the JSON, no additional text."""

KINEMATIC_ANALYSIS_PROMPT = """You are a robotics expert analyzing kinematic joints between rigid groups of an object.

Object Type: {object_type}
You are analyzing the joint between:
- Group 1 ({group1_id}): parts {group1_parts}
- Group 2 ({group2_id}): parts {group2_parts}

You are given:
- RGB images showing the complete object
- Overlay images highlighting each group from multiple viewpoints
- Spatial relationship data

Joint Types:
1. **revolute**: One group rotates around a fixed axis (hinge joint)
2. **prismatic**: One group slides along a fixed axis (linear actuator)  
3. **fixed**: Groups are rigidly connected with no relative motion
4. **ball**: Spherical joint allowing rotation in multiple axes
5. **cylindrical**: Combination of rotation and translation along same axis

Consider the typical kinematic behavior of {object_type} objects.

Spatial Data: {spatial_data}

Output EXACTLY this JSON format:
{{
  "joint_type": "revolute|prismatic|fixed|ball|cylindrical",
  "confidence": 0.85,
  "reasoning": "Brief explanation based on object type and visual evidence",
  "joint_axis_hint": [ax, ay, az],
  "connection_point": [x, y, z]
}}

Provide only the JSON, no additional text."""

def analyze_object_and_create_hierarchy(client, model, part_info, contacts, rgb_images):
    """Use VLM to understand object type and create part hierarchy."""
    
    # Prepare summary data
    num_parts = len(part_info)
    spatial_summary = {
        "part_sizes": [p['volume_approx'] for p in part_info.values()],
        "elongations": [p['elongation'] for p in part_info.values()],
        "total_parts": num_parts
    }
    contact_summary = {
        "total_contacts": len(contacts),
        "contact_pairs": [(c['part_i'], c['part_j']) for c in contacts]
    }
    
    content = [
        {
            "type": "text", 
            "text": OBJECT_UNDERSTANDING_PROMPT.format(
                num_parts=num_parts,
                spatial_summary=json.dumps(spatial_summary, indent=2),
                contact_summary=json.dumps(contact_summary, indent=2)
            )
        }
    ]
    
    # Add RGB images for object context
    for img_path in rgb_images[:4]:  # Use first 4 views for context
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
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error in object analysis: {e}")
        return None

def analyze_group_joint(client, model, object_type, group1, group2, spatial_data, rgb_images, overlay_images):
    """Use VLM to analyze joint type between two rigid groups."""
    
    content = [
        {
            "type": "text", 
            "text": KINEMATIC_ANALYSIS_PROMPT.format(
                object_type=object_type,
                group1_id=group1['group_id'],
                group1_parts=group1['part_indices'],
                group2_id=group2['group_id'], 
                group2_parts=group2['part_indices'],
                spatial_data=json.dumps(spatial_data, indent=2)
            )
        }
    ]
    
    # Add RGB images for context (first 4 views)
    for img_path in rgb_images[:4]:
        if os.path.exists(img_path):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64img(img_path)}"}
            })
    
    # Add overlay images for both groups (all 8 views)
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
        
        # Extract JSON from response
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error in joint analysis for groups {group1['group_id']}-{group2['group_id']}: {e}")
        return None

def compute_group_spatial_relationship(group1, group2, part_info):
    """Compute spatial relationship between two groups."""
    # Get centroids of all parts in each group
    centroids1 = [part_info[i]['centroid'] for i in group1['part_indices']]
    centroids2 = [part_info[i]['centroid'] for i in group2['part_indices']]
    
    # Group centroids
    group1_center = np.mean(centroids1, axis=0)
    group2_center = np.mean(centroids2, axis=0)
    
    # Distance between groups
    distance = np.linalg.norm(group2_center - group1_center)
    
    # Size estimates
    sizes1 = [part_info[i]['volume_approx'] for i in group1['part_indices']]
    sizes2 = [part_info[i]['volume_approx'] for i in group2['part_indices']]
    
    return {
        'distance': float(distance),
        'group1_center': group1_center.tolist(),
        'group2_center': group2_center.tolist(),
        'group1_total_volume': float(sum(sizes1)),
        'group2_total_volume': float(sum(sizes2)),
        'size_ratio': float(sum(sizes1) / (sum(sizes2) + 1e-8))
    }

def check_group_contact(group1, group2, contacts):
    """Check if two groups are in contact based on their constituent parts."""
    group1_parts = set(group1['part_indices'])
    group2_parts = set(group2['part_indices'])
    
    # Check if any parts from group1 contact any parts from group2
    for contact in contacts:
        part_i, part_j = contact['part_i'], contact['part_j']
        if (part_i in group1_parts and part_j in group2_parts) or \
           (part_i in group2_parts and part_j in group1_parts):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Infer hierarchical kinematic scene graph from PartField clustering")
    parser.add_argument("--points_ply", required=True, help="Path to clustered point cloud PLY file")
    parser.add_argument("--labels_npy", required=True, help="Path to clustering labels NPY file")
    parser.add_argument("--overlay_dir", required=True, help="Directory with overlay visualizations")
    parser.add_argument("--rgb_dir", required=True, help="Directory with RGB reference images")
    parser.add_argument("--object_class", default="", help="Object class hint (optional)")
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
    
    # Compute contact relationships
    print("Computing contact relationships...")
    contacts = compute_contact_relationships(part_info)
    print(f"Found {len(contacts)} contact relationships")
    
    # Get RGB images for object context
    rgb_images = get_rgb_images(args.rgb_dir)
    print(f"Found {len(rgb_images)} RGB reference images")
    
    # Step 1: Analyze object and create hierarchy
    print("Analyzing object type and creating part hierarchy...")
    hierarchy_result = analyze_object_and_create_hierarchy(
        client, args.model, part_info, contacts, rgb_images
    )
    
    if not hierarchy_result:
        print("Failed to analyze object hierarchy, falling back to individual parts")
        # Fallback: treat each part as its own group
        rigid_groups = [
            {
                "group_id": f"part_{i}",
                "part_indices": [i],
                "description": f"Individual part {i}"
            }
            for i in part_info.keys()
        ]
        object_type = args.object_class or "unknown"
    else:
        rigid_groups = hierarchy_result['rigid_groups']
        object_type = hierarchy_result['object_type']
        print(f"Identified object type: {object_type}")
        print(f"Created {len(rigid_groups)} rigid groups")
    
    # Step 2: Analyze joints between contacting groups
    print("Analyzing joints between rigid groups...")
    edges = []
    
    for i, j in itertools.combinations(range(len(rigid_groups)), 2):
        group1, group2 = rigid_groups[i], rigid_groups[j]
        
        # Check if groups are in contact
        if not check_group_contact(group1, group2, contacts):
            print(f"Groups {group1['group_id']}-{group2['group_id']}: No contact, skipping")
            continue
        
        print(f"Analyzing joint: {group1['group_id']} <-> {group2['group_id']}")
        
        # Get spatial relationship
        spatial_data = compute_group_spatial_relationship(group1, group2, part_info)
        
        # Get overlay images for both groups
        all_part_indices = group1['part_indices'] + group2['part_indices']
        overlay_images = get_overlay_images(args.overlay_dir, all_part_indices)
        
        if len(overlay_images) < 4:  # Need reasonable number of views
            print(f"  Insufficient overlay images ({len(overlay_images)}), skipping")
            continue
        
        # Analyze with VLM
        vlm_result = analyze_group_joint(
            client, args.model, object_type,
            group1, group2, spatial_data, rgb_images, overlay_images
        )
        
        if vlm_result:
            edge = {
                "source": group1['group_id'],
                "target": group2['group_id'],
                "type": vlm_result.get("joint_type", "fixed"),
                "confidence": vlm_result.get("confidence", 0.5),
                "reasoning": vlm_result.get("reasoning", ""),
                "spatial_metrics": spatial_data,
                "source_parts": group1['part_indices'],
                "target_parts": group2['part_indices']
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
    
    # Build hierarchical scene graph
    nodes = []
    
    # Add group nodes
    for group in rigid_groups:
        node = {
            "id": group['group_id'],
            "type": "rigid_group",
            "description": group['description'],
            "part_indices": group['part_indices'],
            "geometry": {
                "parts": {str(idx): part_info[idx] for idx in group['part_indices']}
            }
        }
        nodes.append(node)
    
    # Add individual part nodes as children
    for part_id in part_info.keys():
        node = {
            "id": f"part_{part_id}",
            "type": "part",
            "index": int(part_id),
            "parent_group": next(
                (g['group_id'] for g in rigid_groups if part_id in g['part_indices']), 
                None
            ),
            "geometry": part_info[part_id],
            "overlay_path": os.path.join(args.overlay_dir, f"index_{part_id}")
        }
        nodes.append(node)
    
    scene_graph = {
        "object_type": object_type,
        "analysis_type": "hierarchical_kinematic",
        "hierarchy_analysis": hierarchy_result,
        "points_source": args.points_ply,
        "labels_source": args.labels_npy,
        "overlay_dir": args.overlay_dir,
        "rgb_dir": args.rgb_dir,
        "nodes": nodes,
        "edges": edges,
        "rigid_groups": rigid_groups,
        "contact_relationships": contacts,
        "statistics": {
            "total_parts": len(part_info),
            "total_groups": len(rigid_groups),
            "total_joints": len(edges),
            "total_contacts": len(contacts),
            "joint_types": {}
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
    output_file = os.path.join(os.path.expanduser(args.output_dir), "hierarchical_kinematic_scene_graph.json")
    
    with open(output_file, "w") as f:
        json.dump(scene_graph, f, indent=2)
    
    print(f"\n✅ Hierarchical kinematic scene graph saved to: {output_file}")
    print(f"   Object type: {object_type}")
    print(f"   Parts: {len(part_info)} -> Groups: {len(rigid_groups)}")
    print(f"   Contacts: {len(contacts)} -> Joints: {len(edges)}")
    print(f"   Joint types: {joint_counts}")

if __name__ == "__main__":
    main()