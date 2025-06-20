import argparse
import math
import random
import sys

from collections import defaultdict
from typing import List

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
    """Rate-distortion loss with optional detailed outputs."""

    def __init__(self, lmbda=1e-2, return_details: bool = False, bitdepth: int = 8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scale = lambda x: (2**bitdepth - 1) ** 2 * x
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
        channels = x.size(1)
        if isinstance(x, torch.Tensor):
            x = x.chunk(channels, dim=1)
        if isinstance(target, torch.Tensor):
            target = target.chunk(channels, dim=1)

        metrics = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metrics.append(v)
        metrics = torch.stack(metrics)
        distortion = torch.sum(metrics.transpose(1, 0), dim=1) / channels
        scaled = self._scale(distortion)
        return scaled, distortion

    def forward(self, output, target):
        self._check_tensors_list(target)
        self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        num_pixels = H * W * num_frames

        distortions = []
        scaled_list = []
        results = {}
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_d, d = self._get_scaled_distortion(x_hat, x)
            scaled_list.append(scaled_d)
            distortions.append(d)
            if self.return_details:
                results[f"frame{i}.mse_loss"] = d
        results["mse_loss"] = torch.stack(distortions).mean()
        avg_scaled = sum(scaled_list) / num_frames

        bpp_loss, bpp_info = collect_likelihoods_list(output.pop("likelihoods"), num_pixels)
        if self.return_details:
            results.update(bpp_info)
        bpp_loss = bpp_loss.mean()

        total_loss = (self.lmbda * avg_scaled).mean() + bpp_loss
        results.update({"loss": total_loss, "distortion": avg_scaled.mean(), "bpp_loss": bpp_loss})
        return results


def test_epoch(epoch, loader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_mse, total_bpp, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            frames = [f.to(device) for f in batch]
            out = model(frames)
            res = criterion(out, frames)
            total_loss += res["loss"].item()
            total_mse += res["mse_loss"].item()
            total_bpp += res["bpp_loss"].item()
            count += 1
    print(f"[{epoch}] Test results - Loss: {total_loss/count:.4f}, MSE: {total_mse/count:.8f}, BPP: {total_bpp/count:.8f}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Video testing script (inference only).")
    parser.add_argument('-m', '--model', default='ssf2020', choices=video_models.keys(), help='Model architecture')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Root directory of test dataset')
    parser.add_argument('--patch-size', type=int, nargs=2, default=(256, 256), help='Spatial crop size')
    parser.add_argument('--test-batch-size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--quality', type=int, default=3, help='Model quality level (1-10)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional path to model checkpoint')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA acceleration')

    parser.add_argument('--aimet-calibrate', action='store_true', default=False, help='Calibrates the model')
    parser.add_argument('--aimet-load-encodings', action='store_true', default=False, help='Load the model encodings')

    parser.add_argument('--aimet-path-encodings', type=str, default=None, help='Path to save/load aimet encodings')
    parser.add_argument('--aimet-activation-bw', type=int, required=False, default=32)
    parser.add_argument('--aimet-weight-bw', type=int, required=False, default=32)
    
    parser.add_argument('--aimet-single-module-calibrate', type=str, choices=[
        "img_encoder", "img_decoder", "img_hyperprior_encoder", "img_hyperprior_decoder_mean", "img_hyperprior_decoder_scale", "res_encoder", "res_decoder", "res_hyperprior_encoder", "res_hyperprior_decoder_mean", "res_hyperprior_decoder_scale", "motion_encoder", "motion_decoder", "motion_hyperprior_encoder", "motion_hyperprior_decoder_mean", "motion_hyperprior_decoder_scale" 
        ], default=None)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(args.patch_size)
    ])

    if (args.aimet_calibrate) :
        split = 'calibrate'
    else: 
        split = 'test'

    test_dataset = VideoFolder(
        args.dataset,
        rnd_interval=False,
        rnd_temp_order=False,
        split=split,
        transform=transform_test
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == 'cuda')
    )

    # Always use pretrained weights
    net = video_models[args.model](quality=args.quality, pretrained=True).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        lmbda = checkpoint.get('loss_lambda', 1e-2)
    else:
        print('Warning: no checkpoint provided, using pretrained weights only.')
        lmbda = 1e-2

    criterion = RateDistortionLoss(lmbda=lmbda, return_details=True)

    if(args.aimet_load_encodings):
        net.aimet_set_cfg(encodings_path=args.aimet_path_encodings, weight_bw=args.aimet_weight_bw, activation_bw=args.aimet_activation_bw)        
        net.aimet_set_modules(args.aimet_single_module_calibrate)        
        net.aimet_load_encodings()
    
    
    if(args.aimet_calibrate):       
        net.aimet_set_cfg(encodings_path=args.aimet_path_encodings, weight_bw=args.aimet_weight_bw, activation_bw=args.aimet_activation_bw)        
        net.aimet_set_modules(args.aimet_single_module_calibrate)        

        net.aimet_quantsim()
        net.aimet_insert_wrappers()

    print('=== Running test only ===')  
    
    test_epoch(0, test_loader, net, criterion)

    if(args.aimet_calibrate):
        net.aimet_pass_calibration_data()


if __name__ == '__main__':
    main(sys.argv[1:])
