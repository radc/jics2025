# main.py
import argparse
import math
import random
import sys

from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import VideoFolder
from compressai.zoo import video_models


def collect_likelihoods_list(likelihoods_list, num_pixels: int):
    bpp_info = defaultdict(int)
    bpp_loss = 0
    for i, frame_likelihoods in enumerate(likelihoods_list):
        frame_bpp = 0
        for label, likelihoods in frame_likelihoods.items():
            label_bpp = 0
            for field, v in likelihoods.items():
                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)
                bpp_loss += bpp
                frame_bpp += bpp
                label_bpp += bpp
                bpp_info[f"bpp_loss.{label}"] += bpp.sum()
                bpp_info[f"bpp_loss.{label}.{i}.{field}"] = bpp.sum()
            bpp_info[f"bpp_loss.{label}.{i}"] = label_bpp.sum()
        bpp_info[f"bpp_loss.{i}"] = frame_bpp.sum()
    return bpp_loss, bpp_info


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss, returns per-sequence vectors."""

    def __init__(self, lmbda=1e-2, return_details: bool = False, bitdepth: int = 8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scale = lambda x: (2 ** bitdepth - 1) ** 2 * x
        self.return_details = return_details

    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
            isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst):
        if (
            not isinstance(lst, (tuple, list))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError("Expected a list of 4D torch.Tensor (or tuples) as input")

    def _get_scaled_distortion(self, x, target):
        C = x.size(1)
        if isinstance(x, torch.Tensor):
            x = x.chunk(C, dim=1)
        if isinstance(target, torch.Tensor):
            target = target.chunk(C, dim=1)
        metrics = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metrics.append(v)
        metrics = torch.stack(metrics)
        distortion = torch.sum(metrics.transpose(1, 0), dim=1) / C
        return self._scale(distortion), distortion  # ambos com shape [batch]

    def forward(self, output, target):
        self._check_tensors_list(target)
        self._check_tensors_list(output["x_hat"])
        _, _, H, W = target[0].size()
        num_frames = len(target)
        num_pixels = H * W * num_frames

        distortions = []
        scaled_list = []
        details = {} if self.return_details else None

        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_d, d = self._get_scaled_distortion(x_hat, x)
            scaled_list.append(scaled_d)
            distortions.append(d)
            if self.return_details:
                details[f"frame{i}.mse_loss"] = d

        # per-sequence (batch) vectors
        per_seq_mse    = torch.stack(distortions).mean(dim=0)  # [batch]
        per_seq_scaled = sum(scaled_list) / num_frames        # [batch]
        per_seq_bpp, bpp_info = collect_likelihoods_list(output.pop("likelihoods"), num_pixels)
        if self.return_details:
            details.update(bpp_info)
        per_seq_loss = self.lmbda * per_seq_scaled + per_seq_bpp  # [batch]

        out = {
            "loss":     per_seq_loss,
            "mse_loss": per_seq_mse,
            "bpp_loss": per_seq_bpp,
        }
        if self.return_details:
            out.update(details)
        return out


def test_epoch(epoch, loader, model, criterion):
    """Run inference and print per-sequence: name, Loss, MSE, PSNR, BPP."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # batch é uma lista de T tensors: [Tensor(batch,C,H,W), …]
            frames = [f.to(device) for f in batch]
            res = criterion(model(frames), frames)
            bs = res["loss"].shape[0]
            for i in range(bs):
                idx = batch_idx * loader.batch_size + i
                seq_path = loader.dataset.sample_folders[idx]
                seq_name = seq_path.name
                loss_i = res["loss"][i].item()
                mse_i  = res["mse_loss"][i].item()
                bpp_i  = res["bpp_loss"][i].item()
                # PSNR assume entrada [0,1]
                psnr_i = 10 * math.log10(1.0 / mse_i) if mse_i > 0 else float("inf")
                print(
                    f"[Epoch {epoch}] Seq {seq_name} (#{idx:03d}) → "
                    f"Loss: {loss_i:.4f}, MSE: {mse_i:.12f}, "
                    f"PSNR: {psnr_i:.4f} dB, BPP: {bpp_i:.12f}"
                )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Video testing script (inference only).")
    parser.add_argument("-m", "--model",
                        default="ssf2020",
                        choices=video_models.keys(),
                        help="Model architecture")
    parser.add_argument("-d", "--dataset",
                        type=str,
                        required=True,
                        help="Root directory of test dataset")
    parser.add_argument("--patch-size",
                        type=int,
                        nargs=2,
                        default=(256, 256),
                        help="Spatial crop size")
    parser.add_argument("--test-batch-size",
                        type=int,
                        default=64,
                        help="Batch size for testing")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--quality",
                        type=int,
                        default=3,
                        help="Model quality level (1–9)")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None,
                        help="Optional path to model checkpoint")
    parser.add_argument("--cuda",
                        action="store_true",
                        help="Enable CUDA acceleration")
    parser.add_argument("--num-frames",
                        type=int,
                        default=5,
                        help="Number of frames per sequence")
    parser.add_argument("--aimet-calibrate",
                        action="store_true",
                        default=False,
                        help="Calibrates the model")
    parser.add_argument("--aimet-load-encodings",
                        action="store_true",
                        default=False,
                        help="Load the model encodings")
    parser.add_argument("--aimet-path-encodings",
                        type=str,
                        default=None,
                        help="Path to save/load AIMET encodings")
    parser.add_argument("--aimet-activation-bw",
                        type=int,
                        default=32,
                        help="AIMET activation bit-width")
    parser.add_argument("--aimet-weight-bw",
                        type=int,
                        default=32,
                        help="AIMET weight bit-width")
    parser.add_argument(
        "--aimet-single-module-calibrate",
        type=str,
        choices=[
            "img_encoder", "img_decoder", "img_hyperprior_encoder",
            "img_hyperprior_decoder_mean", "img_hyperprior_decoder_scale",
            "res_encoder", "res_decoder", "res_hyperprior_encoder",
            "res_hyperprior_decoder_mean", "res_hyperprior_decoder_scale",
            "motion_encoder", "motion_decoder", "motion_hyperprior_encoder",
            "motion_hyperprior_decoder_mean", "motion_hyperprior_decoder_scale",
        ],
        default=None,
        help="AIMET single-module calibration target",
    )
    parser.add_argument(
        "--ignore-sequence-folder",
        required=False,
        action="store_true",
        help="Use custom sequences instead of Vimeo setuplet"
    )

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(args.patch_size),
    ])
    split = "calibrate" if args.aimet_calibrate else "test"
    test_dataset = VideoFolder(
        args.dataset,
        rnd_interval=False,
        rnd_temp_order=False,
        split=split,
        transform=transform_test,
        num_frames=args.num_frames,
        ignore_sequence_folder=args.ignore_sequence_folder
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = video_models[args.model](quality=args.quality, pretrained=True).to(device)
    if args.checkpoint:
        chk = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(chk["state_dict"])
        lmbda = chk.get("loss_lambda", 1e-2)
    else:
        print("Warning: no checkpoint provided; using pretrained weights only.")
        lmbda = 1e-2

    criterion = RateDistortionLoss(lmbda=lmbda, return_details=True)

    if args.aimet_load_encodings:
        net.aimet_set_cfg(encodings_path=args.aimet_path_encodings,
                          weight_bw=args.aimet_weight_bw,
                          activation_bw=args.aimet_activation_bw)
        net.aimet_set_modules(args.aimet_single_module_calibrate)
        net.aimet_load_encodings()

    if args.aimet_calibrate:
        net.aimet_set_cfg(encodings_path=args.aimet_path_encodings,
                          weight_bw=args.aimet_weight_bw,
                          activation_bw=args.aimet_activation_bw)
        net.aimet_set_modules(args.aimet_single_module_calibrate)
        net.aimet_quantsim()
        net.aimet_insert_wrappers()

    print("=== Running test only ===")
    test_epoch(0, test_loader, net, criterion)

    if args.aimet_calibrate:
        net.aimet_pass_calibration_data()


if __name__ == "__main__":
    main(sys.argv[1:])
