import torch
import torch.nn.functional as F

FEATURE_COLUMNS = ["conf_fg", "msp", "msp_fg", "entropy", "band_conf"]


@torch.no_grad()
def confidence_features(logits, band_px=5):
    prob = logits.float().softmax(dim=1)
    pred = prob.argmax(dim=1)
    fg = (pred == 1).float()
    msp_map = prob.max(dim=1).values
    ent = -(prob * prob.clamp_min(1e-8).log()).sum(dim=1)
    area = fg.flatten(1).sum(1)
    safe = area.clamp_min(1.0)
    conf_fg = (prob[:, 1] * fg).flatten(1).sum(1) / safe
    msp_fg = (msp_map * fg).flatten(1).sum(1) / safe
    empty = area == 0
    conf_fg = torch.where(empty, torch.zeros_like(conf_fg), conf_fg)
    msp_fg = torch.where(empty, torch.zeros_like(msp_fg), msp_fg)
    k = band_px
    f = fg[:, None]
    band = (F.max_pool2d(f, k, 1, k // 2) - (-F.max_pool2d(-f, k, 1, k // 2)))[:, 0] > 0
    band_n = band.flatten(1).sum(1)
    band_conf = (msp_map * band).flatten(1).sum(1) / band_n.clamp_min(1.0)
    band_conf = torch.where(band_n == 0, msp_map.flatten(1).mean(1), band_conf)
    feats = {"conf_fg": conf_fg, "msp": msp_map.flatten(1).mean(1), "msp_fg": msp_fg,
             "entropy": ent.flatten(1).mean(1), "band_conf": band_conf}
    return pred.to(torch.uint8), {k_: v.cpu().numpy() for k_, v in feats.items()}
