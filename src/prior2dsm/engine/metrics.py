import numpy as np

def mae(pred, gt, mask):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    mask = np.asarray(mask).astype(bool)
    return float(np.mean(np.abs(pred[mask] - gt[mask])))
