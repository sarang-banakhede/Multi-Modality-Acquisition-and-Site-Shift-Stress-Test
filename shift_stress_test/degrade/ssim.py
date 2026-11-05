import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity


def fov_mask(lum, threshold=10.0, erode=3):
    mask = binary_erosion(lum > threshold, iterations=erode)
    return mask if mask.sum() > 100 else np.ones_like(mask, dtype=bool)


def masked_ssim(clean_lum, deg_lum, mask):
    _, smap = structural_similarity(clean_lum, deg_lum, data_range=255, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, full=True)
    return float(smap[mask].mean())
