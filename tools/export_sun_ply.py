"""Bake one continuous sun direction into standard 3DGS SH coefficients."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scene.gaussian_model import GaussianModel
from utils.light_utils import normalized_sun_direction
from utils.system_utils import searchForMaxIteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", "-m", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sun_direction", type=float, nargs=3, required=True,
                        help="OpenGL world direction toward sun; +Y is up")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    iteration = (searchForMaxIteration(str(args.model_path / "point_cloud"))
                 if args.iteration == -1 else args.iteration)
    source = args.model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if source.resolve() == args.output.resolve():
        parser.error("Export output must not overwrite the source conditional PLY")
    sidecar = torch.load(source.with_name("sun_conditioning.pt"), map_location="cpu", weights_only=True)
    model = GaussianModel(sidecar["view_sh_degree"])
    model.load_ply(str(source))
    direction = normalized_sun_direction(args.sun_direction)
    with torch.no_grad():
        model.save_ply(str(args.output.resolve()), sun_direction=direction)
    args.output.with_suffix(".json").write_text(json.dumps({
        "source": str(source.resolve()), "sun_direction": direction.tolist(),
        "active_sh_degree": model.active_sh_degree,
    }, indent=2), encoding="utf-8")
    print(f"Exported {model.get_xyz.shape[0]} Gaussians to {args.output}")


if __name__ == "__main__":
    main()
