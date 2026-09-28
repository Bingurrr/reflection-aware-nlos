#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - training entry point for radar point segmentation +
reflective-surface (wall) estimation.

Surface teacher (Y_soft):
    1) use the tm*_2t / tp*_2t points provided by dataset.py
       (these are already warped into the coordinate frame of t)
    2) refine each neighbour frame with a y-shift that maximises overlap
    3) keep only 1st_bounce_surface points (label_id=4 by default)
    4) render them with an anisotropic (elliptical) Gaussian splat whose
       orientation is estimated by kNN

Note: the *_2t keys of the dataset are the versions warped into frame t.
"""

from __future__ import annotations

import os, json, math, random, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WheelDatasetSeg, collate_fn_seg
from model import WheelOccXAttnLSS_Seg


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list):
            out[k] = [t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in v]
        else:
            out[k] = v
    return out


# =========================================================
# BEV helpers (feature construction)
# =========================================================
def meters_to_bev_xy(
    xy_m: torch.Tensor,
    x_min: float, y_min: float,
    x_size: float, y_size: float,
    bev_h: int, bev_w: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    xy_m: (N,2) meters, x:right-left, y:forward
    return: ix, iy (long), valid_mask
    """
    x = xy_m[:, 0]
    y = xy_m[:, 1]
    u = (x - x_min) / max(1e-6, x_size) * (bev_w - 1)
    v = (y - y_min) / max(1e-6, y_size) * (bev_h - 1)
    ix = u.round().long()
    iy = v.round().long()
    m = (ix >= 0) & (ix < bev_w) & (iy >= 0) & (iy < bev_h)
    return ix, iy, m


def build_bev_feature_from_history(
    batch: Dict[str, Any],
    bev_h: int, bev_w: int,
    x_min: float, y_min: float,
    x_size: float, y_size: float,
    use_offsets: List[int],   # e.g., [-3,-2,-1,0]
) -> torch.Tensor:
    """
    Only past frames (warped tmK_2t points) are rendered into the BEV input
    feature bev_r.
    - (B,1,H,W)
    """
    B = len(batch["radar_xy_t"])
    device = batch["image"].device
    bev = torch.zeros((B, 1, bev_h, bev_w), device=device, dtype=torch.float32)

    def get_xy(b: int, off: int) -> Optional[torch.Tensor]:
        if off == 0:
            return batch["radar_xy_t"][b]
        if off < 0:
            k = abs(off)
            key = f"radar_xy_tm{k}_2t"
            if key in batch:
                return batch[key][b]
        return None  # future frames must not be used as input features

    for b in range(B):
        acc = torch.zeros((bev_h, bev_w), device=device, dtype=torch.float32)
        for off in use_offsets:
            xy = get_xy(b, off)
            if xy is None or xy.numel() == 0:
                continue
            ix, iy, m = meters_to_bev_xy(xy, x_min, y_min, x_size, y_size, bev_h, bev_w)
            if m.any():
                acc.index_put_((iy[m], ix[m]), torch.ones_like(ix[m], dtype=torch.float32), accumulate=True)

        if acc.max() > 0:
            acc = (acc / acc.max()).clamp(0, 1)
        bev[b, 0] = acc

    return bev


# =========================================================
# Teacher (soft surface label) - anisotropic splat
# =========================================================
def refine_overlap_y_numpy(
    src_xy: np.ndarray,
    base_xy: np.ndarray,
    dy_max: float = 0.5,
    dy_step: float = 0.02,
    radius: float = 0.4,
    max_points: int = 600,
) -> np.ndarray:
    """
    Translate the neighbour points (src_xy) along y so that the number of
    matches within `radius` of base_xy is maximised. No point is dropped; the
    same shift is applied to all of them.
    """
    if src_xy.size == 0 or base_xy.size == 0:
        return src_xy

    # downsample for the overlap score only; the shift is applied to all points
    src_for_score = src_xy
    base_for_score = base_xy

    if src_for_score.shape[0] > max_points:
        src_for_score = src_for_score[np.random.choice(src_for_score.shape[0], max_points, replace=False)]
    if base_for_score.shape[0] > max_points:
        base_for_score = base_for_score[np.random.choice(base_for_score.shape[0], max_points, replace=False)]

    dys = np.arange(-dy_max, dy_max + 1e-9, dy_step, dtype=np.float32)
    best_dy = 0.0
    best_score = -1
    r2 = float(radius) * float(radius)

    bx = base_for_score[:, 0][None, :]  # (1,Nb)
    by = base_for_score[:, 1][None, :]  # (1,Nb)
    sx = src_for_score[:, 0][:, None]   # (Ns,1)
    sy0 = src_for_score[:, 1][:, None]  # (Ns,1)

    for dy in dys:
        sy = sy0 + dy
        dx = sx - bx
        dy_mat = sy - by
        dist2 = dx * dx + dy_mat * dy_mat
        min_d2 = dist2.min(axis=1)
        score = int((min_d2 <= r2).sum())
        if score > best_score or (score == best_score and abs(float(dy)) < abs(best_dy)):
            best_score = score
            best_dy = float(dy)

    out = src_xy.copy()
    out[:, 1] += best_dy
    return out


def make_m2pix_numpy(
    bev_h: int, bev_w: int,
    x_min: float, y_min: float,
    x_size: float, y_size: float,
):
    """
    world(x,y)[m] -> (iy,ix)[pixel]  (numpy)
    """
    cell_x = float(x_size) / float(bev_w)
    cell_y = float(y_size) / float(bev_h)

    def fn(xy_np: np.ndarray) -> np.ndarray:
        if xy_np.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        ix = ((xy_np[:, 0] - float(x_min)) / max(1e-9, cell_x)).astype(np.int64)
        iy = ((xy_np[:, 1] - float(y_min)) / max(1e-9, cell_y)).astype(np.int64)
        return np.stack([iy, ix], axis=1)

    return fn, (cell_x, cell_y)


def gaussian_splat_anisotropic_numpy(
    points_xy: np.ndarray,
    bev_h: int,
    bev_w: int,
    sigma_m: float,
    x_min: float, y_min: float,
    x_size: float, y_size: float,
    k_neighbors: int = 5,
    long_ratio: float = 3.0,
    min_sigma_short: float = 0.20,
    radius_factor: float = 3.0,
) -> np.ndarray:
    """
    NumPy implementation of the anisotropic (elliptical) Gaussian splat.
    - estimate a tangent direction per point with kNN
    - accumulate the elliptical Gaussian into the grid with a max reduction,
      using a long sigma along the tangent and a short sigma along the normal
    """
    Y = np.zeros((bev_h, bev_w), dtype=np.float32)
    if points_xy.size == 0:
        return Y

    m2pix, (cell_x, cell_y) = make_m2pix_numpy(bev_h, bev_w, x_min, y_min, x_size, y_size)

    iyix = m2pix(points_xy)  # (N,2) [iy,ix]
    mask = (iyix[:, 0] >= 0) & (iyix[:, 0] < bev_h) & (iyix[:, 1] >= 0) & (iyix[:, 1] < bev_w)
    if not mask.any():
        return Y

    iyix = iyix[mask]
    pts = points_xy[mask].astype(np.float32)  # (N,2)
    N = pts.shape[0]

    sig_short = max(float(min_sigma_short), float(sigma_m))
    sig_long = float(long_ratio) * sig_short

    r_m = float(radius_factor) * sig_long
    rx_pix = int(r_m / max(1e-9, cell_x)) + 1
    ry_pix = int(r_m / max(1e-9, cell_y)) + 1

    # kNN-based tangent direction estimate
    dirs = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        if N <= 1:
            dirs[i] = np.array([0.0, 1.0], dtype=np.float32)
            continue
        p = pts[i]
        diff = pts - p[None, :]
        d2 = np.sum(diff * diff, axis=1)
        d2[i] = np.inf
        k = min(int(k_neighbors), N - 1)
        idx = np.argsort(d2)[:k]
        idx = idx[np.isfinite(d2[idx])]
        if idx.size == 0:
            dirs[i] = np.array([0.0, 1.0], dtype=np.float32)
            continue
        vecs = pts[idx] - p[None, :]
        mean_vec = vecs.mean(axis=0)
        nrm = float(np.linalg.norm(mean_vec))
        if nrm < 1e-6:
            dirs[i] = np.array([0.0, 1.0], dtype=np.float32)
        else:
            dirs[i] = (mean_vec / nrm).astype(np.float32)

    # grid -> world conversion constants
    x0 = float(x_min)
    y0 = float(y_min)

    for (p, (iy, ix), d) in zip(pts, iyix, dirs):
        dn = float(np.linalg.norm(d))
        if dn < 1e-6:
            d = np.array([0.0, 1.0], dtype=np.float32)
        else:
            d = (d / dn).astype(np.float32)
        n = np.array([-d[1], d[0]], dtype=np.float32)

        y0_pix = max(0, int(iy) - ry_pix)
        y1_pix = min(bev_h - 1, int(iy) + ry_pix)
        x0_pix = max(0, int(ix) - rx_pix)
        x1_pix = min(bev_w - 1, int(ix) + rx_pix)

        for yy in range(y0_pix, y1_pix + 1):
            wy = y0 + (yy + 0.5) * cell_y
            for xx in range(x0_pix, x1_pix + 1):
                wx = x0 + (xx + 0.5) * cell_x
                dx = wx - float(p[0])
                dy = wy - float(p[1])

                # project onto the tangent / normal frame
                u = dx * float(d[0]) + dy * float(d[1])  # along
                v = dx * float(n[0]) + dy * float(n[1])  # across

                val = math.exp(-0.5 * ((u * u) / (sig_long * sig_long + 1e-12) + (v * v) / (sig_short * sig_short + 1e-12)))
                if val > float(Y[yy, xx]):
                    Y[yy, xx] = float(val)

    return Y


def _get_xy_lab_from_batch(batch: Dict[str, Any], b: int, off: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    off in {-3,-2,-1,0,+1,+2,...}
    """
    if off == 0:
        return batch["radar_xy_t"][b], batch["radar_label_t"][b]
    if off < 0:
        k = abs(off)
        xk = f"radar_xy_tm{k}_2t"
        lk = f"radar_label_tm{k}_2t"
        if xk in batch:
            return batch[xk][b], batch.get(lk, [None] * len(batch["radar_xy_t"]))[b]
        return None, None
    if off > 0:
        k = off
        xk = f"radar_xy_tp{k}_2t"
        lk = f"radar_label_tp{k}_2t"
        if xk in batch:
            return batch[xk][b], batch.get(lk, [None] * len(batch["radar_xy_t"]))[b]
        return None, None
    return None, None


def build_soft_label_teacher_wheelstyle(
    batch: Dict[str, Any],
    bev_h: int, bev_w: int,
    x_min: float, y_min: float,
    x_size: float, y_size: float,
    sigma_m: float = 0.30,
    use_future_for_teacher: bool = True,
    filter_label_id: int = 4,  # 1st_bounce_surface
    overlap_refine: bool = True,
    overlap_dy_max: float = 0.5,
    overlap_dy_step: float = 0.02,
    overlap_radius: float = 0.4,
    # anisotropic params
    aniso_k: int = 5,
    aniso_long_ratio: float = 3.0,
    aniso_min_sigma_short: float = 0.20,
    aniso_radius_factor: float = 3.0,
    # offsets control
    hist_offsets: List[int] = [-3, -2, -1, 0],
    fut_max_k: int = 2,  # use tp1..tpK (default=2) for teacher
) -> torch.Tensor:
    """
    Soft label Y_soft (B,1,H,W) for the reflective-surface (wall) head.

    - tm*_2t / tp*_2t points are already warped into the frame of t
    - neighbours are refined with the overlap-maximising y-shift
    - only filter_label_id (4 by default) is kept
    - the anisotropic splat sharpens the surface ridge
    """
    B = len(batch["radar_xy_t"])
    device = batch["image"].device

    offsets = list(hist_offsets)
    if use_future_for_teacher:
        offsets += list(range(1, int(fut_max_k) + 1))

    Ys = []
    for b in range(B):
        pts_all: List[np.ndarray] = []

        # points of the current frame t
        base_xy_t = batch["radar_xy_t"][b].detach().cpu().numpy()
        base_lab_t = batch["radar_label_t"][b].detach().cpu().numpy()
        m_base = (base_lab_t == int(filter_label_id))
        base_xy_for_align = base_xy_t[m_base] if m_base.any() else base_xy_t

        for off in offsets:
            xy_tk, lab_tk = _get_xy_lab_from_batch(batch, b, off)
            if xy_tk is None or xy_tk.numel() == 0:
                continue

            xy_np = xy_tk.detach().cpu().numpy()
            if xy_np.size == 0:
                continue

            # the teacher uses surface labels only
            if lab_tk is not None and isinstance(lab_tk, torch.Tensor) and lab_tk.numel() > 0:
                lab_np = lab_tk.detach().cpu().numpy()
                mk = (lab_np == int(filter_label_id))
                xy_np = xy_np[mk] if mk.any() else xy_np[:0]

            if xy_np.size == 0:
                continue

            # refine neighbours with the y-shift
            if overlap_refine and (off != 0) and (base_xy_for_align.size > 0):
                xy_np = refine_overlap_y_numpy(
                    xy_np, base_xy_for_align,
                    dy_max=float(overlap_dy_max),
                    dy_step=float(overlap_dy_step),
                    radius=float(overlap_radius),
                )

            pts_all.append(xy_np)

        if len(pts_all) == 0:
            Y = np.zeros((bev_h, bev_w), dtype=np.float32)
        else:
            pts = np.concatenate(pts_all, axis=0).astype(np.float32) if len(pts_all) > 1 else pts_all[0].astype(np.float32)
            Y = gaussian_splat_anisotropic_numpy(
                points_xy=pts,
                bev_h=bev_h,
                bev_w=bev_w,
                sigma_m=float(sigma_m),
                x_min=float(x_min), y_min=float(y_min),
                x_size=float(x_size), y_size=float(y_size),
                k_neighbors=int(aniso_k),
                long_ratio=float(aniso_long_ratio),
                min_sigma_short=float(aniso_min_sigma_short),
                radius_factor=float(aniso_radius_factor),
            )

        Ys.append(torch.from_numpy(Y)[None, :, :])  # (1,H,W)

    Y_soft = torch.stack(Ys, dim=0).to(device=device, dtype=torch.float32)  # (B,1,H,W)
    return Y_soft.clamp(0, 1)


# =========================================================
# Losses
# =========================================================
def point_seg_loss_only_t(pt_logits_list: List[torch.Tensor], radar_label_t_list: List[torch.Tensor]) -> torch.Tensor:
    """
    The loss is computed on the points of the current frame t only.
    """
    losses = []
    for logits, lab in zip(pt_logits_list, radar_label_t_list):
        if logits is None or logits.numel() == 0:
            continue
        if lab is None or lab.numel() == 0:
            continue
        m = (lab >= 0)
        if m.sum() == 0:
            continue
        losses.append(F.cross_entropy(logits[m], lab[m].long()))
    if len(losses) == 0:
        device = pt_logits_list[0].device if (len(pt_logits_list) and pt_logits_list[0] is not None) else "cpu"
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def soft_bce_logits_with_mask(logits: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    logits: (B,1,H,W)
    target: (B,1,H,W) in [0,1]
    valid_mask: (B,) bool or None
    """
    loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1,2,3))  # (B,)
    if valid_mask is None:
        return loss_map.mean()
    vm = valid_mask.float().view(-1)
    denom = vm.sum().clamp(min=1.0)
    return (loss_map * vm).sum() / denom


def front_sem_loss(front_sem_logits: torch.Tensor, sem_mask: torch.Tensor, sem_valid: torch.Tensor) -> torch.Tensor:
    """
    front_sem_logits: (B,3,H,W)
    sem_mask: (B,H,W)
    sem_valid: (B,) bool  (odd frames -> False)
    """
    losses = []
    for b in range(front_sem_logits.shape[0]):
        if not bool(sem_valid[b].item()):
            continue
        losses.append(F.cross_entropy(front_sem_logits[b:b+1], sem_mask[b:b+1].long()))
    if len(losses) == 0:
        return torch.tensor(0.0, device=front_sem_logits.device)
    return torch.stack(losses).mean()


def presence_loss(ped_present_logit: Optional[torch.Tensor], ped_present: torch.Tensor) -> torch.Tensor:
    if ped_present_logit is None:
        return torch.tensor(0.0, device=ped_present.device)
    logit = ped_present_logit.view(-1)
    target = ped_present.float().view(-1)
    return F.binary_cross_entropy_with_logits(logit, target)


def fov_consistency_loss(
    radar_xy_list: List[torch.Tensor],
    pt_logits_list: List[torch.Tensor],
    ped_present: torch.Tensor,
    fov_deg: float,
    dyn_class_ids: List[int],
) -> torch.Tensor:
    device = ped_present.device
    losses = []
    half = float(fov_deg) * math.pi / 180.0 / 2.0
    eps = 1e-6

    for b in range(len(radar_xy_list)):
        if ped_present[b].item() != 0:
            continue
        xy = radar_xy_list[b]
        if xy is None or xy.numel() == 0:
            continue
        logits = pt_logits_list[b]
        if logits is None or logits.numel() == 0:
            continue

        x = xy[:, 0]
        y = xy[:, 1]
        ang = torch.atan2(x, y.clamp(min=eps))
        m = (y > 0.0) & (ang.abs() <= half)
        if m.sum() == 0:
            continue

        prob = F.softmax(logits, dim=1)
        dyn_prob = prob[:, dyn_class_ids].sum(dim=1)
        p = dyn_prob[m].clamp(0.0, 1.0)
        losses.append((-torch.log((1.0 - p).clamp(min=1e-6))).mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# =========================================================
# Train / Val
# =========================================================
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    args: argparse.Namespace,
    is_train: bool,
) -> Dict[str, float]:
    model.train(is_train)
    pbar = tqdm(loader, desc="[train]" if is_train else "[val]", ncols=120)

    sum_loss = sum_pt = sum_wall = sum_sem = sum_fov = sum_pres = 0.0
    n = 0

    for batch in pbar:
        batch = to_device(batch, device)

        img = batch["image"]                 # (B,3,759,1006)
        radar_xy_t = batch["radar_xy_t"]     # list[(Ni,2)]
        radar_label_t = batch["radar_label_t"]

        sem_mask = batch.get("sem_mask", None)   # (B,759,1006)
        sem_valid = batch.get("sem_valid", None) # (B,) bool (odd frames False)
        ped_present = batch.get("ped_present", None)

        bev_r = build_bev_feature_from_history(
            batch=batch,
            bev_h=args.bev_h, bev_w=args.bev_w,
            x_min=args.x_min, y_min=args.y_min,
            x_size=args.x_size, y_size=args.y_size,
            use_offsets=[-3, -2, -1, 0],
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(
                img=img,
                bev_r=bev_r,
                radar_xy_list=radar_xy_t,   # the point head only sees frame t
                K=None,
                T_cam2ego=None,
            )

            loss_pt = point_seg_loss_only_t(out["pt_logits"], radar_label_t) if args.w_pt > 0 else torch.tensor(0.0, device=device)

            loss_wall = torch.tensor(0.0, device=device)
            if args.w_wall > 0:
                # anisotropic surface teacher
                Y_soft = build_soft_label_teacher_wheelstyle(
                    batch=batch,
                    bev_h=args.bev_h, bev_w=args.bev_w,
                    x_min=args.x_min, y_min=args.y_min,
                    x_size=args.x_size, y_size=args.y_size,
                    sigma_m=float(args.wall_sigma_m),
                    use_future_for_teacher=bool(args.teacher_use_future),
                    filter_label_id=int(args.wall_label_id),
                    overlap_refine=bool(args.teacher_overlap_refine),
                    overlap_dy_max=float(args.overlap_dy_max),
                    overlap_dy_step=float(args.overlap_dy_step),
                    overlap_radius=float(args.overlap_radius),
                    aniso_k=int(args.teacher_aniso_k),
                    aniso_long_ratio=float(args.teacher_aniso_long_ratio),
                    aniso_min_sigma_short=float(args.teacher_aniso_min_sigma_short),
                    aniso_radius_factor=float(args.teacher_aniso_radius_factor),
                    hist_offsets=[-3, -2, -1, 0],
                    fut_max_k=int(args.teacher_future_k),
                )
                loss_wall = soft_bce_logits_with_mask(out["wall_bev_logit"], Y_soft, valid_mask=None)

            loss_sem = torch.tensor(0.0, device=device)
            if args.w_sem > 0 and (sem_mask is not None) and (sem_valid is not None):
                loss_sem = front_sem_loss(out["front_sem_logits"], sem_mask, sem_valid)

            loss_fov = torch.tensor(0.0, device=device)
            if args.w_fov > 0 and (ped_present is not None):
                loss_fov = fov_consistency_loss(
                    radar_xy_list=radar_xy_t,
                    pt_logits_list=out["pt_logits"],
                    ped_present=ped_present,
                    fov_deg=float(args.fov_deg),
                    dyn_class_ids=list(args.dyn_class_ids),
                )

            loss_pres = torch.tensor(0.0, device=device)
            if args.w_pres > 0 and (ped_present is not None):
                loss_pres = presence_loss(out.get("ped_present_logit", None), ped_present)

            loss = (
                args.w_pt * loss_pt
                + args.w_wall * loss_wall
                + args.w_sem * loss_sem
                + args.w_fov * loss_fov
                + args.w_pres * loss_pres
            )

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        n += 1
        sum_loss += float(loss.item())
        sum_pt += float(loss_pt.item())
        sum_wall += float(loss_wall.item())
        sum_sem += float(loss_sem.item())
        sum_fov += float(loss_fov.item())
        sum_pres += float(loss_pres.item())

        pbar.set_postfix({
            "loss": f"{sum_loss/max(n,1):.4f}",
            "pt": f"{sum_pt/max(n,1):.4f}",
            "wall": f"{sum_wall/max(n,1):.4f}",
            "sem": f"{sum_sem/max(n,1):.4f}",
            "fov": f"{sum_fov/max(n,1):.4f}",
            "pres": f"{sum_pres/max(n,1):.4f}",
        })

    return {
        "loss": sum_loss / max(n,1),
        "pt": sum_pt / max(n,1),
        "wall": sum_wall / max(n,1),
        "sem": sum_sem / max(n,1),
        "fov": sum_fov / max(n,1),
        "pres": sum_pres / max(n,1),
    }


# =========================================================
# Args
# =========================================================
def _apply_config_defaults(ap):
    """Parse the CLI twice so that ``--config <json>`` can supply defaults.

    Any value given on the command line still wins over the JSON file.
    """
    args = ap.parse_args()
    cfg_path = getattr(args, "config", "")
    if not cfg_path:
        return args
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    known = {a.dest for a in ap._actions}
    unknown = sorted(set(cfg) - known)
    if unknown:
        print(f"[config] ignoring unknown keys: {unknown}")
    ap.set_defaults(**{k: v for k, v in cfg.items() if k in known})
    return ap.parse_args()


def build_args():
    ap = argparse.ArgumentParser()

    # ---- required ----
    ap.add_argument("--config", type=str, default="",
                    help="JSON file with default values for any of the options below.")

    ap.add_argument("--root", type=str, default="./data_folder/train/dynamic")
    ap.add_argument("--save_dir", type=str, default="./checkpoints/dynamic")

    ap.add_argument("--bev_h", type=int, default=128)
    ap.add_argument("--bev_w", type=int, default=128)

    ap.add_argument("--x_min", type=float, default=-15.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--x_size", type=float, default=30.0)
    ap.add_argument("--y_size", type=float, default=30.0)

    # ---- train basic ----
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    # ---- dataset ----
    ap.add_argument("--csv_dir", type=str, default="radar_data")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--ann_filename", type=str, default="front_annotations.json")
    ap.add_argument("--use_list_index_mapping", action="store_true")
    ap.add_argument("--front_dir_name", type=str, default="front_resized_img")

    # front_resized_img resolution
    ap.add_argument("--img_h", type=int, default=759)
    ap.add_argument("--img_w", type=int, default=1006)

    # original annotation resolution
    ap.add_argument("--ann_h0", type=int, default=3036)
    ap.add_argument("--ann_w0", type=int, default=4024)

    # ---- model ----
    ap.add_argument("--n_depth", type=int, default=32)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)

    # ---- loss weights ----
    ap.add_argument("--w_pt", type=float, default=1.0)
    ap.add_argument("--w_wall", type=float, default=1.0)
    ap.add_argument("--w_sem", type=float, default=1.0)
    ap.add_argument("--w_fov", type=float, default=0.0)
    ap.add_argument("--w_pres", type=float, default=0.0)

    # ---- teacher / wall ----
    ap.add_argument("--wall_label_id", type=int, default=4)       # 1st_bounce_surface
    ap.add_argument("--wall_sigma_m", type=float, default=0.30)

    ap.add_argument("--teacher_use_future", default=True)         # use tp1..tpK
    ap.add_argument("--teacher_future_k", type=int, default=10)   # tp1..tpK
    ap.add_argument("--teacher_no_future", action="store_true")
    ap.add_argument("--teacher_overlap_refine", action="store_true")

    ap.add_argument("--overlap_dy_max", type=float, default=0.5)
    ap.add_argument("--overlap_dy_step", type=float, default=0.02)
    ap.add_argument("--overlap_radius", type=float, default=0.4)

    # anisotropic splat parameters
    ap.add_argument("--teacher_aniso_k", type=int, default=5)
    ap.add_argument("--teacher_aniso_long_ratio", type=float, default=3.0)
    ap.add_argument("--teacher_aniso_min_sigma_short", type=float, default=0.20)
    ap.add_argument("--teacher_aniso_radius_factor", type=float, default=3.0)

    # ---- fov ----
    ap.add_argument("--fov_deg", type=float, default=90.0)
    ap.add_argument("--dyn_class_ids", type=int, nargs="+", default=[2, 3])

    args = _apply_config_defaults(ap)

    if args.teacher_no_future:
        args.teacher_use_future = False

    return args


# =========================================================
# Main
# =========================================================
def main():
    args = build_args()
    ensure_dir(args.save_dir)
    save_json(vars(args), os.path.join(args.save_dir, "args.json"))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    ds = WheelDatasetSeg(
        root=args.root,
        img_size=(args.img_h, args.img_w),
        csv_dir_name=args.csv_dir,
        dt=float(args.dt),
        n_hist=3,
        n_fut=max(0, int(args.teacher_future_k)),      # the teacher needs up to tpK
        ann_filename=args.ann_filename,
        front_dir_name=args.front_dir_name,
        use_list_index_mapping=bool(args.use_list_index_mapping),
        ann_orig_hw=(args.ann_h0, args.ann_w0),
        propagate_wall=True,
        debug_print_first=True,
        debug_sem=False,
    )

    n_total = len(ds)
    n_val = int(round(n_total * float(args.val_ratio)))
    n_train = max(1, n_total - n_val)
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_seg,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_fn_seg,
        drop_last=False,
    )

    # model
    model = WheelOccXAttnLSS_Seg(
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        x_min=float(args.x_min),
        y_min=float(args.y_min),
        x_size=float(args.x_size),
        y_size=float(args.y_size),
        bev_c=128,
        heads=int(args.heads),
        layers=int(args.layers),
        n_depth=int(args.n_depth),
        topk=int(args.topk),
        num_pt_classes=6,
        sem_num_classes=3,
        use_presence_head=(args.w_pres > 0),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp))

    # resume
    start_epoch = 0
    best_val = 1e9
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optim"])
        if ("scaler" in ckpt) and args.use_amp and (ckpt["scaler"] is not None):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[RESUME] epoch={start_epoch} best_val={best_val:.6f}")

    # train loop
    for ep in range(start_epoch, args.epochs):
        tr = run_one_epoch(model, train_loader, device, optimizer, bool(args.use_amp), scaler, args, is_train=True)
        va = run_one_epoch(model, val_loader, device, None, bool(args.use_amp), None, args, is_train=False) if n_val > 0 else {"loss": tr["loss"]}

        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scaler": scaler.state_dict() if args.use_amp else None,
            "best_val": best_val,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if va["loss"] < best_val:
            best_val = va["loss"]
            ckpt["best_val"] = best_val
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

        print(
            f"[Epoch {ep+1:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} (pt={tr['pt']:.4f}, wall={tr['wall']:.4f}, sem={tr['sem']:.4f}, fov={tr['fov']:.4f}, pres={tr['pres']:.4f}) | "
            f"val loss={va['loss']:.4f} | best={best_val:.4f}"
        )

    print(f"[DONE] saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
