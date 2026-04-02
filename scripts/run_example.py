import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import normalize

from prior2dsm.config import load_config
from prior2dsm.io import load_example
from prior2dsm.models.dino_loader import load_dino_with_lora
from prior2dsm.models.lora import get_lora_parameters
from prior2dsm.models.mlp_decoder import MLPDecoder
from prior2dsm.engine.tto_trainer import prepare_patch_targets, run_tto
from prior2dsm.engine.inference import run_dense_inference
from prior2dsm.engine.metrics import mae
from prior2dsm.utils.seed import set_seed
from prior2dsm.utils.visualization import save_summary_figure, save_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(
        "cuda" if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu"
    )

    example_name = cfg["example_name"]
    sample = load_example(example_name, cfg)

    rgb_cpu = sample["rgb"]
    gt_cpu = sample["gt"]
    mask_cpu = sample["mask"]
    rel_cpu = sample["rel"].to(device)
    prior_raw = sample["prior"]

    H, W = gt_cpu.shape
    anchor_mask_cpu = (~mask_cpu) & torch.isfinite(prior_raw) & (prior_raw != 0)

    anchor_mask = anchor_mask_cpu.to(device)
    prior_gpu = prior_raw.to(device)

    mean = tuple(cfg["norm"]["mean"])
    std = tuple(cfg["norm"]["std"])

    rgb_tto = normalize(rgb_cpu.unsqueeze(0), mean, std).to(device)

    dino = load_dino_with_lora(
        repo_path=cfg["model"]["dino_repo"],
        ckpt_path=cfg["model"]["dino_ckpt"],
        device=device,
        lora_r=cfg["model"]["lora_r"],
        lora_alpha=cfg["model"]["lora_alpha"],
    ).train()

    mlp_head = MLPDecoder(in_dim=cfg["model"]["in_dim"]).to(device).train()

    rel_p, prior_p, mask_p = prepare_patch_targets(
        rel_cpu=rel_cpu,
        prior_gpu=prior_gpu,
        anchor_mask=anchor_mask,
        H=H,
        W=W,
        patch_size=cfg["data_params"]["patch_size"],
    )

    params = list(mlp_head.parameters()) + get_lora_parameters(dino)
    optimizer = torch.optim.AdamW(params, lr=cfg["optim"]["lr"])

    print(f"Starting TTO for {cfg['optim']['tto_steps']} steps...")
    loss_hist = run_tto(
        dino=dino,
        mlp_head=mlp_head,
        rgb_tto=rgb_tto,
        rel_p=rel_p,
        prior_p=prior_p,
        mask_p=mask_p,
        optimizer=optimizer,
        tto_steps=cfg["optim"]["tto_steps"],
    )

    print("Running dense inference...")
    final_dsm = run_dense_inference(
        dino=dino,
        mlp_head=mlp_head,
        rgb_cpu=rgb_cpu,
        rel_cpu=rel_cpu,
        mean=mean,
        std=std,
        patch_size=cfg["data_params"]["patch_size"],
        stride=cfg["data_params"]["stride"],
    )

    final_mae = mae(
        pred=final_dsm.detach().cpu().numpy(),
        gt=gt_cpu.numpy(),
        mask=mask_cpu.numpy(),
    )
    print(f"Final MAE: {final_mae:.6f}")

    stem = Path(example_name).stem
    out_root = Path(cfg["output"]["root_dir"])
    save_summary_figure(
        rgb_cpu=rgb_cpu,
        final_dsm=final_dsm.detach().cpu(),
        loss_hist=loss_hist,
        output_path=out_root / "figures" / f"{stem}_summary.png",
    )
    save_metrics(
        {"example_name": example_name, "mae": final_mae},
        out_root / "logs" / f"{stem}_metrics.json",
    )

if __name__ == "__main__":
    main()
