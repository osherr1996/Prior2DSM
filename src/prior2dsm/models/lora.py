import numpy as np
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        self.scaling = alpha / r

        self.A = nn.Linear(base_linear.in_features, r, bias=False)
        self.B = nn.Linear(r, base_linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + self.scaling * self.B(self.A(x))

def inject_lora(model, r=8, alpha=16.0):
    for blk in model.modules():
        if hasattr(blk, "attn"):
            if hasattr(blk.attn, "qkv"):
                blk.attn.qkv = LoRALinear(blk.attn.qkv, r=r, alpha=alpha)
            if hasattr(blk.attn, "proj"):
                blk.attn.proj = LoRALinear(blk.attn.proj, r=r, alpha=alpha)
    return model

def get_lora_parameters(model):
    params = []
    for name, param in model.named_parameters():
        if "A.weight" in name or "B.weight" in name:
            params.append(param)
    return params
