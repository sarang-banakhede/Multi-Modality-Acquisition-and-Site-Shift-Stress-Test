import math

import torch
import torch.nn.functional as F


def augment(x, y, cfg, flags, gen):
    b, dev = x.shape[0], x.device
    ones = torch.ones(b, device=dev)

    def rand():
        return torch.rand(b, device=dev, generator=gen)

    if flags["hflip"]:
        m = rand() < 0.5
        x = torch.where(m[:, None, None, None], x.flip(-1), x)
        y = torch.where(m[:, None, None], y.flip(-1), y)
    if flags["vflip"]:
        m = rand() < 0.5
        x = torch.where(m[:, None, None, None], x.flip(-2), x)
        y = torch.where(m[:, None, None], y.flip(-2), y)
    rot_on = rand() < cfg.p_rotate
    ang = torch.where(rot_on, (rand() * 2 - 1) * flags["rot"], torch.zeros(b, device=dev)) * math.pi / 180.0
    lo, hi = cfg.scale_range
    sc_on = rand() < cfg.p_scale
    sc = torch.where(sc_on, lo + rand() * (hi - lo), ones)
    sel = rot_on | sc_on
    if sel.any():
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.zeros(b, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
        grid = F.affine_grid(theta[sel], (int(sel.sum()),) + tuple(x.shape[1:]), align_corners=False)
        x[sel] = F.grid_sample(x[sel], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        y[sel] = F.grid_sample(y[sel][:, None].float(), grid, mode="nearest", padding_mode="zeros",
                               align_corners=False)[:, 0].to(y.dtype)
    it_on = rand() < cfg.p_intensity
    bright = torch.where(it_on, 1 + (rand() * 2 - 1) * cfg.intensity_range, ones)
    contrast = torch.where(it_on, 1 + (rand() * 2 - 1) * cfg.intensity_range, ones)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast[:, None, None, None] + mean) * bright[:, None, None, None]
    return x.clamp_(0.0, 1.0), y
