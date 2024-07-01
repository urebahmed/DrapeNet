import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

from data.cloth3d import Cloth3d
from models.cbndec import CbnDecoder
from models.coordsenc import CoordsEncoder
from models.dgcnn import Dgcnn
from utils import compute_gradients, progress_bar, random_point_sampling

class EncoderDecoderTrainer:
    def __init__(self) -> None:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)

        train_ids_file = "/lib33/DrapeNet/CLOTH3D/train-ids.pkl"
        dset_category = "top"
        dset_root = "/lib33/DrapeNet/dataset/top"
        train_bs = 4 # Adjust batch size according to your setup
        num_points_pcd = 10000  # Adjust as needed
        latent_size = 32  # Adjust as needed
        max_dist = 0.1  # Adjust maximum distance
        num_points_forward = 20000  # Adjust as needed
        decoder_cfg = {
            "hidden_dim": 512,
            "num_hidden_layers": 5
        }
        lr = 0.0001  # Adjust learning rate
        num_epochs = 3000 # Adjust number of epochs

        self.hcfg_values = {
            "train_ids_file": train_ids_file,
            "dset_category": dset_category,
            "dset_root": dset_root,
            "train_bs": train_bs,
            "num_points_pcd": num_points_pcd,
            "latent_size": latent_size,
            "udf_max_dist": max_dist,
            "num_points_forward": num_points_forward,
            "decoder_cfg": decoder_cfg,
            "lr": lr,
            "num_epochs": num_epochs
        }

        train_dset = Cloth3d(
            Path(train_ids_file),
            Path(dset_root),
            dset_category
        )
        self.train_sampler = DistributedSampler(train_dset)
        self.train_loader = DataLoader(
            train_dset,
            batch_size=train_bs,
            shuffle=False,
            num_workers=8,
            sampler=self.train_sampler,
        )

        encoder = Dgcnn(latent_size).cuda(local_rank)
        self.encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[local_rank])

        self.coords_encoder = CoordsEncoder()

        decoder = CbnDecoder(
            self.coords_encoder.out_dim,
            latent_size,
            decoder_cfg["hidden_dim"],
            decoder_cfg["num_hidden_layers"],
        ).cuda(local_rank)
        self.decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[local_rank])

        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = Adam(params, lr)

        self.epoch = 0
        self.global_step = 0

        self.ckpts_path = Path("./ckpts")

        if self.ckpts_path.exists() and local_rank == 0:
            self.restore_from_last_ckpt()

        self.ckpts_path.mkdir(exist_ok=True)

        if local_rank == 0:
            self.logger = SummaryWriter("./logs")

    def train(self) -> None:
        start_epoch = self.epoch

        for epoch in range(start_epoch, self.hcfg_values["num_epochs"]):
            self.epoch = epoch
            self.encoder.train()
            self.decoder.train()
            self.train_sampler.set_epoch(epoch)

            desc = f"Epoch {epoch}/{self.hcfg_values['num_epochs']}"
            for batch in progress_bar(self.train_loader, desc=desc):
                _, _, pcds, coords, gt_udf, gt_grad = batch
                pcds = pcds.cuda()
                coords = coords.cuda()
                gt_udf = gt_udf.cuda()
                gt_grad = gt_grad.cuda()

                pcds = random_point_sampling(pcds, self.hcfg_values["num_points_pcd"])

                gt_udf = gt_udf / self.hcfg_values["udf_max_dist"]
                gt_udf = 1 - gt_udf
                c_u_g = torch.cat([coords, gt_udf.unsqueeze(-1), gt_grad], dim=-1)

                selected_c_u_g = random_point_sampling(c_u_g, self.hcfg_values["num_points_forward"])
                selected_coords = selected_c_u_g[:, :, :3]
                selected_coords.requires_grad = True
                selected_gt_udf = selected_c_u_g[:, :, 3]
                selected_gt_grad = selected_c_u_g[:, :, 4:]

                latent_codes = self.encoder(pcds)
                coords_encoded = self.coords_encoder.encode(selected_coords)
                pred = self.decoder(coords_encoded, latent_codes)

                udf_loss = F.binary_cross_entropy_with_logits(pred, selected_gt_udf)

                udf_pred = torch.sigmoid(pred)
                udf_pred = 1 - udf_pred
                udf_pred *= self.hcfg_values["udf_max_dist"]
                gradients = compute_gradients(selected_coords, udf_pred)

                grad_loss = F.mse_loss(gradients, selected_gt_grad, reduction="none")
                mask = (selected_gt_udf > 0) & (selected_gt_udf < 1)
                grad_loss = grad_loss[mask].mean()

                self.optimizer.zero_grad()

                loss = udf_loss + 0.1 * grad_loss

                loss.backward()
                self.optimizer.step()

                if self.global_step % 10 == 0 and dist.get_rank() == 0:
                    self.logger.add_scalar(
                        "train/udf_loss",
                        udf_loss.item(),
                        self.global_step,
                    )
                    self.logger.add_scalar(
                        "train/grad_loss",
                        grad_loss.item(),
                        self.global_step,
                    )

                self.global_step += 1

            if epoch % 50 == 49 and dist.get_rank() == 0:
                self.save_ckpt(all=True)

            if dist.get_rank() == 0:
                self.save_ckpt()

    def save_ckpt(self, all: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "encoder": self.encoder.module.state_dict(),
            "decoder": self.decoder.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if all:
            ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)
        else:
            for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
                if "last" in previous_ckpt_path.name:
                    previous_ckpt_path.unlink()

            ckpt_path = self.ckpts_path / f"last_{self.epoch}.pt"
            torch.save(ckpt, ckpt_path)

    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_path.exists():
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "last" in p.name]
            error_msg = "Expected only one last ckpt, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)

            self.encoder.module.load_state_dict(ckpt["encoder"])
            self.decoder.module.load_state_dict(ckpt["decoder"])
            self.optimizer.load_state_dict(ckpt["optimizer"])

if __name__ == "__main__":
    trainer = EncoderDecoderTrainer()
    trainer.train()