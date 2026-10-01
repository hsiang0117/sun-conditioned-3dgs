#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpips import LPIPS
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    # Stream one pair at a time; the local GPU has 8 GB of memory.
    for fname in sorted(os.listdir(renders_dir)):
        if not fname.lower().endswith(".png"):
            continue
        with Image.open(renders_dir / fname) as rendered, Image.open(gt_dir / fname) as target:
            yield (tf.to_tensor(rendered.convert("RGB")).unsqueeze(0).cuda(),
                   tf.to_tensor(target.convert("RGB")).unsqueeze(0).cuda(), fname)

def evaluate(model_paths):

    lpips_model = LPIPS(net="vgg").eval().cuda()

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in sorted(os.listdir(test_dir)):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                image_names = []

                ssims = []
                psnrs = []
                lpipss = []

                with torch.no_grad():
                    for rendered, target, name in tqdm(readImages(renders_dir, gt_dir), desc="Metric evaluation progress"):
                        ssims.append(ssim(rendered, target).item())
                        psnrs.append(psnr(rendered, target).item())
                        lpipss.append(lpips_model(rendered * 2 - 1, target * 2 - 1).item())
                        image_names.append(name)
                if not image_names:
                    raise ValueError(f"No evaluation images in {renders_dir}")

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as error:
            raise RuntimeError(f"Unable to compute metrics for model {scene_dir}") from error

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
