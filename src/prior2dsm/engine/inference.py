import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

def run_dense_inference(
    dino,
    mlp_head,
    rgb_cpu,
    rel_cpu,
    mean,
    std,
    patch_size=16,
    stride=4,
):
    dino.eval()
    mlp_head.eval()

    with torch.no_grad():
        p = patch_size
        rgb_pad = F.pad(rgb_cpu.unsqueeze(0), (p, p, p, p), mode="reflect")
        hp_pad, wp_pad = rgb_pad.shape[-2:]
        H, W = rgb_cpu.shape[-2:]

        device = rel_cpu.device
        sb_acc = torch.zeros((2, hp_pad // stride, wp_pad // stride), device=device)
        cnt_acc = torch.zeros((1, hp_pad // stride, wp_pad // stride), device=device)

        rgb_norm = normalize(rgb_pad, mean, std).to(device)

        for dy in range(0, p, stride):
            for dx in range(0, p, stride):
                hc = ((hp_pad - dy) // p) * p
                wc = ((wp_pad - dx) // p) * p
                if hc <= 0 or wc <= 0:
                    continue

                patch = rgb_norm[:, :, dy:dy + hc, dx:dx + wc]
                tokens = dino.forward_features(patch)["x_norm_patchtokens"].squeeze(0)
                sb_local = mlp_head(tokens).t().reshape(2, hc // p, wc // p)

                sb_acc[
                    :,
                    dy // stride: dy // stride + (hc // p) * (p // stride): p // stride,
                    dx // stride: dx // stride + (wc // p) * (p // stride): p // stride,
                ] += sb_local

                cnt_acc[
                    :,
                    dy // stride: dy // stride + (hc // p) * (p // stride): p // stride,
                    dx // stride: dx // stride + (wc // p) * (p // stride): p // stride,
                ] += 1

        sb_dense = sb_acc / (cnt_acc + 1e-8)
        offset = (p - (p // 2)) // stride + 1
        sb_final = sb_dense[:, offset: offset + (H // stride), offset: offset + (W // stride)]

        sb_hr = F.interpolate(sb_final.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0)
        s_hr, b_hr = sb_hr[0], sb_hr[1]
        final_dsm = s_hr * rel_cpu + b_hr

    return final_dsm
