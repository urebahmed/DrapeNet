import sys

sys.path.append("..")

from pathlib import Path
import numpy as np
import torch
from hesiod import get_out_dir, hcfg, hmain
from torch.utils.data import DataLoader

from data.cloth3d import Cloth3d
from models.dgcnn import Dgcnn
from utils import progress_bar, random_point_sampling

# if len(sys.argv) != 2:
#     print("Usage: python export_codes.py <run_cfg_file>")
#     exit(1)


# @hmain(
#     base_cfg_dir="../cfg/bases",
#     run_cfg_file=sys.argv[1],
#     parse_cmd_line=False,
#     out_dir_root="../logs",
# )
def main() -> None:
    ckpt_path = "/home/hello/drapenet/DrapeNet/logs/chkpt/last_2999.pt"
    ckpt = torch.load(ckpt_path)

    # Get configuration parameters
    num_points_pcd = 10000
    latent_size = 32

    # Initialize the encoder model
    encoder = Dgcnn(latent_size)
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

     # Load the single point cloud file (assuming it's in NPZ format)
    point_cloud_file = Path("/home/hello/drapenet/shirt/udfs/Shirt.npz")
    npz_data = np.load(point_cloud_file)
    pcd = torch.tensor(npz_data["pcd"], dtype=torch.float32)  # Assuming "points" is the key for the point cloud data
    pcd = pcd.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
    pcd = random_point_sampling(pcd, num_points_pcd)  # Assuming the file is a PyTorch tensor


    # Generate latent codes
    with torch.no_grad():
        latent_codes = encoder(pcd)
    # Save the latent codes
    latent_codes = latent_codes.detach().cpu()

    out_dir = Path("/home/hello/drapenet/shirt/codes")
    out_dir.mkdir(exist_ok=True, parents=True)

    latent_codes_path = out_dir / f"{point_cloud_file.stem}_latent.pt"
    torch.save(latent_codes, latent_codes_path)

    mean_latent_code = torch.mean(latent_codes, dim=0)
    torch.save(mean_latent_code, out_dir / f"{point_cloud_file.stem}_mean.pt")


if __name__ == "__main__":
    main()