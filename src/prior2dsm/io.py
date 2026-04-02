from pathlib import Path
import numpy as np
import torch
import rasterio

def read_raster_bands(path, bands, dtype=np.float32, scale=None):
    path = Path(path)
    with rasterio.open(path) as src:
        arr = src.read(bands).astype(dtype)
        profile = src.profile
    if scale is not None:
        arr = arr / scale
    return torch.from_numpy(arr), profile

def load_example(example_name: str, cfg: dict):
    data_cfg = cfg["data"]
    params = cfg["data_params"]

    rgb, rgb_profile = read_raster_bands(
        Path(data_cfg["rgb_dir"]) / example_name,
        [1, 2, 3],
        scale=255.0,
    )
    gt, _ = read_raster_bands(
        Path(data_cfg["gt_dir"]) / example_name,
        [params["dsm_band"]],
    )
    mask, _ = read_raster_bands(
        Path(data_cfg["mask_dir"]) / example_name,
        [params["mask_band"]],
    )
    rel, _ = read_raster_bands(
        Path(data_cfg["da_dir"]) / example_name,
        [params["da_band"]],
    )
    prior, _ = read_raster_bands(
        Path(data_cfg["prior_dir"]) / example_name,
        [params["prior_band"]],
    )

    return {
        "rgb": rgb,
        "gt": gt.squeeze(0),
        "mask": mask.squeeze(0) > 0,
        "rel": rel.squeeze(0),
        "prior": prior.squeeze(0),
        "profile": rgb_profile,
    }
