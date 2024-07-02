import sys

sys.path.append("..")

from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
from hesiod import get_out_dir, hcfg, hmain
from torch import Tensor
from torch.utils.data import DataLoader

from data.cloth3d import Cloth3d
from meshudf.meshudf import get_mesh_from_udf
from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from models.dgcnn import Dgcnn
from utils import get_o3d_mesh_from_tensors, progress_bar, random_point_sampling

import open3d as o3d  # isort: skip

# @hmain(
#     base_cfg_dir="../cfg/bases",
#     run_cfg_file=sys.argv[1],
#     parse_cmd_line=False,
#     out_dir_root="../logs",
# )
def main() -> None:
    ckpt_path = Path("/home/hello/drapenet/DrapeNet/logs/chkpt/last_2999.pt")
    ckpt = torch.load(ckpt_path)

    latent_size = 32
    num_points_pcd = 10000
    udf_max_dist = 0.1

    encoder = Dgcnn(latent_size)
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    coords_encoder = CoordsEncoder()

    decoder_cfg = {
        "hidden_dim": 512,
        "num_hidden_layers": 5,
    }
    decoder = CbnDecoder(
        coords_encoder.out_dim,
        latent_size,
        decoder_cfg["hidden_dim"],
        decoder_cfg["num_hidden_layers"],
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.cuda()
    decoder.eval()

    # Load a single point cloud file for processing
    point_cloud_file = Path("/home/hello/drapenet/shirt/udfs/Shirt.npz")  # Adjust the path
    npz_data = np.load(point_cloud_file)
    item_id = "single_point_cloud"  # Provide a unique identifier for the single point cloud

    pcd = torch.tensor(npz_data["pcd"], dtype=torch.float32).unsqueeze(0).cuda()
    pcd = random_point_sampling(pcd, num_points_pcd)

    with torch.no_grad():
        latent_codes = encoder(pcd)

    # Process the single point cloud
    lat = latent_codes.squeeze(0)

    def udf_func(c: Tensor) -> Tensor:
        c = coords_encoder.encode(c.unsqueeze(0))
        p = decoder(c, lat).squeeze(0)
        p = torch.sigmoid(p)
        p = (1 - p) * udf_max_dist
        return p

    v, t = get_mesh_from_udf(
        udf_func,
        coords_range=(-1, 1),
        max_dist=udf_max_dist,
        N=512,
        max_batch=2**16,
        differentiable=False,
    )

    pred_mesh_o3d = get_o3d_mesh_from_tensors(v, t)
    mesh_path = get_out_dir() / f"meshes_single/{item_id}.obj"
    mesh_path.parent.mkdir(exist_ok=True, parents=True)
    o3d.io.write_triangle_mesh(str(mesh_path), pred_mesh_o3d)

if __name__ == "__main__":
    main()
