import numpy as np
from prior2dsm.engine.metrics import mae

def test_mae():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    gt = np.array([[1.0, 3.0], [1.0, 5.0]])
    mask = np.array([[True, False], [True, True]])
    value = mae(pred, gt, mask)
    assert abs(value - (0.0 + 2.0 + 1.0) / 3.0) < 1e-8
