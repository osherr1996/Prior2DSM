import torch
from .lora import inject_lora

def clean_checkpoint_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        k = k.replace("teacher.backbone.", "")
        cleaned[k] = v
    return cleaned

def load_dino_with_lora(repo_path, ckpt_path, device, lora_r=8, lora_alpha=16.0):
    dino = torch.hub.load(repo_path, "dinov3_vitl16", source="local", pretrained=False)
    raw_sd = torch.load(ckpt_path, map_location="cpu")
    state_dict = clean_checkpoint_keys(raw_sd)
    dino.load_state_dict(state_dict, strict=False)
    dino = inject_lora(dino, r=lora_r, alpha=lora_alpha)
    return dino.to(device)
