import torch.nn.functional as F

def prepare_patch_targets(rel_cpu, prior_gpu, anchor_mask, H, W, patch_size):
    Hp, Wp = H // patch_size, W // patch_size
    prior_p = F.interpolate(prior_gpu.view(1, 1, H, W), size=(Hp, Wp), mode="bilinear").flatten()
    rel_p = F.interpolate(rel_cpu.view(1, 1, H, W), size=(Hp, Wp), mode="bilinear").flatten()
    mask_p = F.interpolate(anchor_mask.float().view(1, 1, H, W), size=(Hp, Wp), mode="area").flatten() > 0.5
    return rel_p, prior_p, mask_p

def run_tto(
    dino,
    mlp_head,
    rgb_tto,
    rel_p,
    prior_p,
    mask_p,
    optimizer,
    tto_steps=100,
    log_interval=10,
):
    loss_hist = []

    dino.train()
    mlp_head.train()

    for step in range(tto_steps):
        optimizer.zero_grad(set_to_none=True)

        tokens = dino.forward_features(rgb_tto)["x_norm_patchtokens"].squeeze(0)
        sb = mlp_head(tokens)
        s, b = sb[:, 0], sb[:, 1]

        pred_p = s * rel_p + b
        loss = F.huber_loss(pred_p[mask_p], prior_p[mask_p], delta=1.0)

        loss.backward()
        optimizer.step()

        loss_hist.append(float(loss.item()))
        if step % log_interval == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.6f}")

    return loss_hist
