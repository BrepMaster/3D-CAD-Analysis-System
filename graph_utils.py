# graph_utils.py
import numpy as np
import torch
import dgl
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid

def build_graph(solid, curv_num_u_samples=10, surf_num_u_samples=10, surf_num_v_samples=10):
    """Convert STEP solid to DGL graph"""
    # Build face adjacency graph
    graph = face_adjacency(solid)

    # Compute face UV grid features
    graph_face_feat = []
    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        points = uvgrid(face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples)
        normals = uvgrid(face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples)
        visibility_status = uvgrid(face, method="visibility_status",
                               num_u=surf_num_u_samples, num_v=surf_num_v_samples)
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)

    # Compute edge U grid features
    graph_edge_feat = []
    for edge_idx in graph.edges:
        edge = graph.edges[edge_idx]["edge"]
        if not edge.has_curve():
            continue
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert to DGL graph
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph