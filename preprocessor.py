# preprocessor.py
import torch
from pathlib import Path
from torch import FloatTensor

def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)

def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))
    mask = inp[..., 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)

def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp

def load_one_graph(file_path):
    from dgl.data.utils import load_graphs
    file_path = Path(file_path)
    graph = load_graphs(str(file_path))[0][0]
    sample = {"graph": graph, "filename": file_path.stem}

    sample["graph"].ndata["x"], center, scale = center_and_scale_uvgrid(
        sample["graph"].ndata["x"], return_center_scale=True
    )

    sample["graph"].edata["x"][..., :3] -= center
    sample["graph"].edata["x"][..., :3] *= scale
    sample["graph"].ndata["x"] = sample["graph"].ndata["x"].type(FloatTensor)
    sample["graph"].edata["x"] = sample["graph"].edata["x"].type(FloatTensor)

    return sample