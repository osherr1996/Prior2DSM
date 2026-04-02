import torch
from prior2dsm.models.mlp_decoder import MLPDecoder

def test_mlp_decoder_output_shape():
    model = MLPDecoder(in_dim=1024)
    x = torch.randn(7, 1024)
    y = model(x)
    assert y.shape == (7, 2)
