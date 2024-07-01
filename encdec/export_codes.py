import sys

sys.path.append("..")
from pathlib import Path
import torch
import numpy as np
from hesiod import get_out_dir, hcfg, hmain
from torch.utils.data import DataLoader, Dataset

from models.dgcnn import Dgcnn
from utils import progress_bar, random_point_sampling

if len(sys.argv) != 2:
    print("Usage: python export_codes.py <run_cfg_file>")
    exit(1)

class SingleFileDataset(Dataset):
    def __init__(self, npz_file):
        self.data = np.load(npz_file)
    
    def __len__(self):
        return len(self.data['pcd'])
    
    def __getitem__(self, idx):
        pcd = self.data['pcd'][idx]
        return pcd

# @hmain(
#     base_cfg_dir="../cfg/bases",
#     run_cfg_file=sys.argv[1],
#     parse_cmd_line=False,
#     out_dir_root="../logs",
# )
def main() -> None:
    ckpt_path = '/mnt/d/DrapeNet-main/last_2999.pt'
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

    num_points_pcd = 10000
    latent_size = 32
    encoder = Dgcnn(latent_size)
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cpu()
    encoder.eval()

    top_npz_file = Path("/mnt/d/DrapeNet-main/testing/top.npz")
    bottom_npz_file = Path("/mnt/d/DrapeNet-main/testing/bottom.npz")

    bs = 1

    print(f"Loading datasets: {top_npz_file} and {bottom_npz_file}")

    top_dset = SingleFileDataset(top_npz_file)
    top_loader = DataLoader(top_dset, bs, num_workers=4)
    bottom_dset = SingleFileDataset(bottom_npz_file)
    bottom_loader = DataLoader(bottom_dset, bs, num_workers=4)
    print("All loaded")

    for split, loader in [("top", top_loader), ("bottom", bottom_loader)]:
        all_latent_codes = []

        for batch_idx, batch in enumerate(progress_bar(loader, split)):
            print(f"Processing batch {batch_idx}")
            pcds = batch.cpu()
            pcds = random_point_sampling(pcds, num_points_pcd)

            with torch.no_grad():
                latent_codes = encoder(pcds)

            all_latent_codes.append(latent_codes.detach().cpu())

        all_latent_codes = torch.cat(all_latent_codes, dim=0)
        latent_codes_path = get_out_dir() / f"latent_codes/{split}_all.pt"
        latent_codes_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Saving all codes at {latent_codes_path}")
        torch.save(all_latent_codes, latent_codes_path)

        mean_latent_code = torch.mean(all_latent_codes, dim=0)
        print(f"Saving mean points at {mean_latent_code}")
        torch.save(mean_latent_code, latent_codes_path.parent / f"{split}_mean.pt")
    print("Done Saving All.")

if __name__ == "__main__":
    main()
