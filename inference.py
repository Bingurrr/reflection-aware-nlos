#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_seg_ray_accum.py

Inference for radar point segmentation + reflective-surface localisation,
with temporal accumulation of the predicted wall heatmap.

Idea
- keep the wall heatmap (occ_map_t) predicted at every frame t
- at the next frame, warp the occ_maps of (t-1, t-2, t-3) into the current
  frame with the SE(2) ego-motion derived from the wheel encoder, and merge them
- reflector search (find_reflector_for_cluster) then runs on occ_map_acc instead
  of occ_map, which strongly reduces the angular jitter of the fitted segment.

Requirement
- a wheel-encoder pose must be readable from meta for the real SE(2) warp
- otherwise the code falls back to a plain EMA accumulation without alignment,
  which is only a partial substitute.

Supported meta pose keys (parsed leniently)
- meta["ego_pose"] = [x, y, yaw_deg or yaw_rad]
- meta["pose"] / meta["odom"] / meta["wheel_pose"] and similar
- scalar keys such as meta["ego_x"], meta["ego_y"], meta["ego_yaw"]

Output folder layout:
wheel_results/<scene_name>/
    - points_segmentation/
    - surface_reconstruction/
    - points_with_heatmap/
    - overlay/
    - Plot_final/
    - csv/
    - GT_seg/
    - front_seg/

Optional output
- surface_reconstruction_accum/ : pass --save_accum_heatmap to also write the
  accumulated heatmap
"""

from __future__ import annotations

import os
import csv
import json
import ast
import math
import argparse
import inspect
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Deque
from collections import deque

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw

# -----------------------------
# Try import dataset + collate
# -----------------------------
DATASET_CLS = None
COLLATE_FN = None
try:
    from dataset import WheelDatasetSeg as _DS, collate_fn_seg as _CF
    DATASET_CLS = _DS
    COLLATE_FN = _CF
except Exception:
    try:
        from dataset import WheelDataset as _DS, collate_fn as _CF
        DATASET_CLS = _DS
        COLLATE_FN = _CF
    except Exception as e:
        raise ImportError(
            "Cannot import dataset from dataset.py "
            "(WheelDatasetSeg/collate_fn_seg OR WheelDataset/collate_fn)."
        ) from e

from model import WheelOccXAttnLSS_Seg


# --------------------- global class / palette settings --------------------- #
NUM_CLASSES = 6  # 0..5 (points)
CLASS_NAMES = {
    0: "none",
    1: "1st_ped",
    2: "2nd_ped",
    3: "3rd_ped",
    4: "1st_wall",
    5: "3rd_wall",
}
PALETTE = {
    0: (0.6, 0.6, 0.6),
    1: (0.12, 0.47, 0.71),
    2: (1.0, 0.5, 0.05),
    3: (0.84, 0.15, 0.16),
    4: (0.17, 0.63, 0.17),
    5: (0.58, 0.40, 0.74),
}

# fixed figure size: 8 * 120 = 960 px
FIGSIZE = (8, 8)
SAVE_DPI = 120
SAVE_WPX = int(FIGSIZE[0] * SAVE_DPI)
SAVE_HPX = int(FIGSIZE[1] * SAVE_DPI)


# =========================================================
# Small utils
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(str(Path(path).parent))
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


def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


# =========================================================
# Figure helpers (fixed 960x960, no crop)
# =========================================================
def new_fixed_fig_ax():
    fig = plt.figure(figsize=FIGSIZE, dpi=SAVE_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    return fig, ax


def fig_to_rgba_pil(fig) -> Image.Image:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return Image.fromarray(buf).convert("RGBA")


def save_pil_rgba(img: Image.Image, path: str):
    ensure_dir(str(Path(path).parent))
    img.save(path)


def _apply_common_view(ax, xlim, ylim):
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_facecolor((0, 0, 0, 0))


# =========================================================
# BEV mapping utils
# =========================================================
def make_m2pix(bev_hw, grid_origin, grid_size):
    H, W = bev_hw
    x0, y0 = grid_origin
    gx, gy = grid_size
    cell_x = gx / W
    cell_y = gy / H

    def fn(xy_np: np.ndarray):
        if xy_np.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        ix = ((xy_np[:, 0] - x0) / cell_x).astype(np.int64)
        iy = ((xy_np[:, 1] - y0) / cell_y).astype(np.int64)
        return np.stack([iy, ix], 1)

    return fn, (cell_x, cell_y)


def make_pix2m(bev_hw, grid_origin, grid_size):
    H, W = bev_hw
    x0, y0 = grid_origin
    gx, gy = grid_size
    cell_x = gx / W
    cell_y = gy / H

    def fn(iyix_np: np.ndarray):
        if iyix_np.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        iy = iyix_np[:, 0].astype(np.float32)
        ix = iyix_np[:, 1].astype(np.float32)
        x = x0 + (ix + 0.5) * cell_x
        y = y0 + (iy + 0.5) * cell_y
        return np.stack([x, y], axis=1)

    return fn


# =========================================================
# Seg inference flow BEV feature (history only) - keep yours
# =========================================================
def meters_to_bev_xy(
    xy_m: torch.Tensor,
    x_min: float, y_min: float,
    x_size: float, y_size: float,
    bev_h: int, bev_w: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return None

    for b in range(B):
        acc = torch.zeros((bev_h, bev_w), device=device, dtype=torch.float32)
        for off in use_offsets:
            xy = get_xy(b, off)
            if xy is None or xy.numel() == 0:
                continue
            ix, iy, m = meters_to_bev_xy(xy, x_min, y_min, x_size, y_size, bev_h, bev_w)
            if m.any():
                acc.index_put_((iy[m], ix[m]),
                               torch.ones_like(ix[m], dtype=torch.float32),
                               accumulate=True)
        if acc.max() > 0:
            acc = (acc / acc.max()).clamp(0, 1)
        bev[b, 0] = acc
    return bev


# =========================================================
# Front image helpers (seg overlay)
# =========================================================
def read_front_from_tensor(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


def overlay_sem_on_front(front_rgb: np.ndarray, sem_map: np.ndarray, out_png: str, alpha: float = 0.35):
    ensure_dir(str(Path(out_png).parent))
    H, W = sem_map.shape
    col = np.zeros((H, W, 3), dtype=np.uint8)
    col[sem_map == 1] = np.array([0, 255, 0], dtype=np.uint8)   # wall
    col[sem_map == 2] = np.array([255, 0, 0], dtype=np.uint8)   # ped

    over = front_rgb.astype(np.float32) * (1 - alpha) + col.astype(np.float32) * alpha
    over = np.clip(over, 0, 255).astype(np.uint8)
    Image.fromarray(over).save(out_png)


def _safe_float_xy(points) -> np.ndarray:
    arr = np.array(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr


def build_gt_mask_from_json(
    ann_json_path: str,
    frame_idx: int,
    out_hw: Tuple[int, int],           # (H_resized, W_resized)
    ann_orig_hw: Tuple[int, int],      # (H_orig, W_orig)
    wall_labels: Tuple[str, ...] = ("wall", "Wall", "WALL"),
    ped_labels: Tuple[str, ...]  = ("ped", "Ped", "PED", "person", "Person", "human", "Human"),
) -> Optional[np.ndarray]:
    if (ann_json_path is None) or (not os.path.isfile(ann_json_path)):
        return None

    H1, W1 = out_hw
    H0, W0 = ann_orig_hw
    sx = float(W1) / max(1.0, float(W0))
    sy = float(H1) / max(1.0, float(H0))

    try:
        with open(ann_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    items = data if isinstance(data, list) else [data]
    sem = np.zeros((H1, W1), dtype=np.uint8)
    found_any = False

    for item in items:
        shapes = item.get("shapes", []) if isinstance(item, dict) else []
        for shp in shapes:
            fr = shp.get("frame", None)
            if fr is None or int(fr) != int(frame_idx):
                continue

            label = str(shp.get("label", ""))
            pts = _safe_float_xy(shp.get("points", []))
            if pts.size == 0:
                continue

            pts[:, 0] *= sx
            pts[:, 1] *= sy

            img = Image.fromarray(sem)
            draw = ImageDraw.Draw(img)
            poly = [(float(x), float(y)) for x, y in pts.tolist()]

            if label in wall_labels:
                draw.polygon(poly, fill=1)
                found_any = True
            elif label in ped_labels:
                draw.polygon(poly, fill=2)
                found_any = True

            sem = np.array(img, dtype=np.uint8)

    if not found_any:
        return None
    return sem


# =========================================================
# GT positions loader (for BEV boxes)
# =========================================================
def load_gt_positions_from_meta(meta: Dict[str, Any]) -> np.ndarray:
    paths = meta.get("paths", {}) if isinstance(meta, dict) else {}
    csv_path = paths.get("csv", None)

    if (not csv_path) or (not os.path.exists(csv_path)):
        return np.zeros((0, 2), dtype=np.float32)

    gt_list = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("GT_Position", "")
                if not val:
                    continue
                try:
                    obj = ast.literal_eval(val)
                except Exception:
                    continue

                if isinstance(obj, (list, tuple)) and len(obj) == 2 and \
                   isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float)):
                    gt_list.append([float(obj[0]), float(obj[1])])

                elif isinstance(obj, (list, tuple)) and len(obj) > 0 and \
                     isinstance(obj[0], (list, tuple)) and len(obj[0]) >= 2:
                    for p in obj:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            gt_list.append([float(p[0]), float(p[1])])

                elif isinstance(obj, (list, tuple)) and len(obj) == 1 and \
                     isinstance(obj[0], (list, tuple)) and len(obj[0]) > 0 and \
                     isinstance(obj[0][0], (list, tuple)):
                    for p in obj[0]:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            gt_list.append([float(p[0]), float(p[1])])

    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

    if len(gt_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    gt_arr = np.array(gt_list, dtype=np.float32)
    gt_arr = np.unique(gt_arr, axis=0)
    return gt_arr


# =========================================================
# SE(2) pose parsing + heatmap warp / accumulation
# =========================================================
def _try_get_pose_from_meta(meta: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """
    Return pose as (x_world, y_world, yaw_rad).
    Parse as many pose encodings as possible.
    """
    if not isinstance(meta, dict):
        return None

    # (A) scalar keys
    for kx, ky, kyaw in [
        ("ego_x", "ego_y", "ego_yaw"),
        ("x", "y", "yaw"),
        ("odom_x", "odom_y", "odom_yaw"),
        ("wheel_x", "wheel_y", "wheel_yaw"),
    ]:
        if (kx in meta) and (ky in meta) and (kyaw in meta):
            try:
                x = float(meta[kx]); y = float(meta[ky]); yaw = float(meta[kyaw])
                # guess the yaw unit (degrees are common)
                if abs(yaw) > 2.5 * math.pi:
                    yaw = math.radians(yaw)
                return (x, y, yaw)
            except Exception:
                pass

    # (B) list pose
    for key in ["ego_pose", "pose", "odom", "wheel_pose", "wheel_odom"]:
        if key in meta:
            try:
                v = meta[key]
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    x = float(v[0]); y = float(v[1]); yaw = float(v[2])
                    if abs(yaw) > 2.5 * math.pi:
                        yaw = math.radians(yaw)
                    return (x, y, yaw)
                if isinstance(v, dict) and ("x" in v) and ("y" in v) and ("yaw" in v):
                    x = float(v["x"]); y = float(v["y"]); yaw = float(v["yaw"])
                    if abs(yaw) > 2.5 * math.pi:
                        yaw = math.radians(yaw)
                    return (x, y, yaw)
            except Exception:
                pass

    # (C) nested paths
    for key in ["state", "ego", "vehicle", "wheel"]:
        if key in meta and isinstance(meta[key], dict):
            sub = meta[key]
            for kx, ky, kyaw in [("x", "y", "yaw"), ("ego_x", "ego_y", "ego_yaw")]:
                if (kx in sub) and (ky in sub) and (kyaw in sub):
                    try:
                        x = float(sub[kx]); y = float(sub[ky]); yaw = float(sub[kyaw])
                        if abs(yaw) > 2.5 * math.pi:
                            yaw = math.radians(yaw)
                        return (x, y, yaw)
                    except Exception:
                        pass

    return None


def _se2_compose(x: float, y: float, yaw: float, p: torch.Tensor) -> torch.Tensor:
    """
    p: (...,2) in local frame
    returns (...,2) in world frame
    """
    c = math.cos(yaw); s = math.sin(yaw)
    R = torch.tensor([[c, -s],
                      [s,  c]], dtype=p.dtype, device=p.device)
    t = torch.tensor([x, y], dtype=p.dtype, device=p.device)
    return p @ R.T + t


def _se2_inv_apply(x: float, y: float, yaw: float, pw: torch.Tensor) -> torch.Tensor:
    """
    world -> local
    pw: (...,2) in world
    """
    c = math.cos(yaw); s = math.sin(yaw)
    Rinv = torch.tensor([[ c, s],
                         [-s, c]], dtype=pw.dtype, device=pw.device)
    t = torch.tensor([x, y], dtype=pw.dtype, device=pw.device)
    return (pw - t) @ Rinv.T


def _pixgrid_to_meters(
    bev_h: int, bev_w: int,
    x_min: float, y_min: float,
    x_size: float, y_size: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    returns (H,W,2) meters for pixel centers in current BEV
    """
    ys = torch.arange(bev_h, device=device, dtype=dtype)
    xs = torch.arange(bev_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cell_x = float(x_size) / float(bev_w)
    cell_y = float(y_size) / float(bev_h)
    x = float(x_min) + (xx + 0.5) * cell_x
    y = float(y_min) + (yy + 0.5) * cell_y
    return torch.stack([x, y], dim=-1)  # (H,W,2)


def warp_occ_map_past_to_current(
    occ_past: torch.Tensor,  # (1,1,H,W)
    pose_past: Tuple[float, float, float],
    pose_cur: Tuple[float, float, float],
    bev_h: int, bev_w: int,
    x_min: float, y_min: float,
    x_size: float, y_size: float,
) -> torch.Tensor:
    """
    Warp a past ego-frame heatmap into the current ego frame.
    For grid_sample, every current pixel centre is mapped to world coordinates
    and then inverted back into the past frame.
    """
    assert occ_past.ndim == 4 and occ_past.shape[2] == bev_h and occ_past.shape[3] == bev_w
    device = occ_past.device
    dtype = occ_past.dtype

    # current pixel centers in current ego frame (meters)
    p_cur = _pixgrid_to_meters(bev_h, bev_w, x_min, y_min, x_size, y_size, device, dtype)  # (H,W,2)

    # current ego -> world
    x_c, y_c, yaw_c = pose_cur
    p_world = _se2_compose(x_c, y_c, yaw_c, p_cur.reshape(-1, 2)).reshape(bev_h, bev_w, 2)

    # world -> past ego
    x_p, y_p, yaw_p = pose_past
    p_past = _se2_inv_apply(x_p, y_p, yaw_p, p_world.reshape(-1, 2)).reshape(bev_h, bev_w, 2)

    # past meters -> past pixel coords (continuous)
    cell_x = float(x_size) / float(bev_w)
    cell_y = float(y_size) / float(bev_h)
    u = (p_past[..., 0] - float(x_min)) / max(1e-6, cell_x) - 0.5  # pixel center -> index
    v = (p_past[..., 1] - float(y_min)) / max(1e-6, cell_y) - 0.5

    # normalize to [-1,1] for grid_sample (align_corners=False)
    # x: 0..W-1 -> -1..1 => (u + 0.5)/W *2 -1
    # but u already is "index", so:
    gx = (u + 0.5) / float(bev_w) * 2.0 - 1.0
    gy = (v + 0.5) / float(bev_h) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,H,W,2)

    warped = F.grid_sample(
        occ_past, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False
    )
    return warped  # (1,1,H,W)


def accumulate_occ_maps(
    occ_cur: torch.Tensor,  # (1,1,H,W)
    pose_cur: Optional[Tuple[float, float, float]],
    history: Deque[Tuple[torch.Tensor, Optional[Tuple[float, float, float]]]],  # list of (occ,pose)
    max_hist: int,
    alpha: float,
    do_warp: bool,
    bev_h: int, bev_w: int,
    x_min: float, y_min: float,
    x_size: float, y_size: float,
) -> torch.Tensor:
    """
    occ_acc = occ_cur + alpha*warp(occ_{t-1}) + alpha^2*warp(occ_{t-2}) + ...
    Without a pose, this degrades to a plain EMA accumulation.
    """
    acc = occ_cur.clone()
    w = 1.0

    items = list(history)[-max_hist:]  # older -> newer
    # apply the decay starting from the newest frame
    for occ_past, pose_past in reversed(items):
        w *= float(alpha)
        if do_warp and (pose_cur is not None) and (pose_past is not None):
            occ_w = warp_occ_map_past_to_current(
                occ_past, pose_past, pose_cur,
                bev_h, bev_w, x_min, y_min, x_size, y_size
            )
            acc = acc + w * occ_w
        else:
            # fallback: without a pose the maps are simply summed (no alignment)
            acc = acc + w * occ_past

    # normalize to [0,1] (robust)
    mx = acc.max().clamp(min=1e-6)
    acc = (acc / mx).clamp(0.0, 1.0)
    return acc


# =========================================================
# Ray-tracing blocks (same as your wheel inference)
# =========================================================
def cluster_points(points: np.ndarray, eps: float = 0.8, min_samples: int = 3):
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    try:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(points)
    except Exception:
        labels = np.full((points.shape[0],), -1, dtype=int)
    return labels


def merge_cluster_centers(centers_1st: np.ndarray,
                          centers_3rd_mir: np.ndarray,
                          merge_radius: float = 1.5):
    all_centers = []
    if centers_1st is not None and centers_1st.size > 0:
        all_centers.append(centers_1st)
    if centers_3rd_mir is not None and centers_3rd_mir.size > 0:
        all_centers.append(centers_3rd_mir)
    if len(all_centers) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    all_centers = np.concatenate(all_centers, axis=0)
    if all_centers.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    db = DBSCAN(eps=merge_radius, min_samples=1)
    lbl = db.fit_predict(all_centers)
    final_centers = []
    for cid in np.unique(lbl):
        pts = all_centers[lbl == cid]
        if pts.shape[0] == 0:
            continue
        final_centers.append(pts.mean(axis=0))
    if len(final_centers) == 0:
        return np.zeros((0, 2), np.float32)
    return np.stack(final_centers, axis=0).astype(np.float32)


def mirror_points_across_line(points: np.ndarray,
                              center: np.ndarray,
                              direction: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    d = direction / (np.linalg.norm(direction) + 1e-8)
    n = np.array([-d[1], d[0]], dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-8)

    v = points - center[None, :]
    dist = np.sum(v * n[None, :], axis=1, keepdims=True)
    mirrored = points - 2.0 * dist * n[None, :]
    return mirrored


def segment_score_on_heatmap(occ_map: np.ndarray,
                             center: np.ndarray,
                             direction: np.ndarray,
                             length_m: float,
                             thickness_m: float,
                             m2pix):
    H, W = occ_map.shape
    d = direction / (np.linalg.norm(direction) + 1e-8)
    n = np.array([-d[1], d[0]], dtype=np.float32)

    L_samp = 21
    T_samp = 5
    half_L = length_m / 2.0
    half_T = thickness_m / 2.0

    ls = np.linspace(-half_L, half_L, L_samp)
    ts = np.linspace(-half_T, half_T, T_samp)

    pts = []
    for a in ls:
        for t in ts:
            pts.append(center + a * d + t * n)
    pts = np.array(pts, dtype=np.float32)

    iyix = m2pix(pts)
    if iyix.size == 0:
        return 0.0, 0

    iy = iyix[:, 0]
    ix = iyix[:, 1]
    mask = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
    if not mask.any():
        return 0.0, 0

    iy = iy[mask]
    ix = ix[mask]
    vals = occ_map[iy, ix]
    if vals.size == 0:
        return 0.0, 0

    return float(vals.mean()), int(vals.size)


def find_reflector_for_cluster(center_3rd: np.ndarray,
                               occ_map: np.ndarray,
                               m2pix,
                               length_m: float = 5.0,
                               thickness_m: float = 1.0,
                               n_r_samples: int = 20,
                               n_theta_samples: int = 18,
                               dist_gamma: float = 0.0,
                               dist_ref: float = 10.0,
                               min_valid: int = 10):
    """Search for the reflective segment that best explains a 3rd-bounce cluster.

    Radar returns thin out with range, so the predicted surface heatmap is
    systematically weaker far from the sensor and the raw mean score favours
    near candidates. ``dist_gamma`` > 0 compensates for that by scaling the
    score of a candidate at range r by (r / dist_ref) ** dist_gamma.
    ``dist_gamma = 0`` reproduces the unweighted search.
    """
    O = np.array([0.0, 0.0], dtype=np.float32)

    c = center_3rd.astype(np.float32)
    dist = np.linalg.norm(c)
    if dist < 1e-3:
        return None, None

    dir_ray = c / (dist + 1e-8)

    r_min = max(2.0, 0.2 * dist)
    r_max = max(r_min + 0.5, dist - 0.5)
    if r_max <= r_min:
        return None, None

    r_list = np.linspace(r_min, r_max, n_r_samples)
    theta_list = np.linspace(0.0, np.pi, n_theta_samples)

    L_min = 2.5
    L_max = max(L_min, length_m)
    if L_max - L_min < 0.5:
        length_list = [L_min, L_max]
    else:
        length_list = np.linspace(L_min, L_max, 3)

    best_score = -1.0
    best_center = None
    best_dir = None

    for r in r_list:
        p = O + r * dir_ray
        for th in theta_list:
            d = np.array([np.cos(th), np.sin(th)], dtype=np.float32)
            for L in length_list:
                score, n_valid = segment_score_on_heatmap(
                    occ_map, p, d, length_m=L, thickness_m=thickness_m, m2pix=m2pix
                )
                if n_valid < int(min_valid):
                    continue
                if dist_gamma != 0.0:
                    score = score * (max(float(r), 1e-3) / float(dist_ref)) ** float(dist_gamma)
                if score > best_score:
                    best_score = score
                    best_center = p.copy()
                    best_dir = d.copy()

    if best_center is None or best_dir is None:
        return None, None
    return best_center, best_dir


# =========================================================
# Confusion update (points)
# =========================================================
def confusion_update(conf_mat: np.ndarray | None,
                     gt: np.ndarray,
                     pred: np.ndarray,
                     num_classes: int = NUM_CLASSES,
                     ignore: int = -1):
    if conf_mat is None:
        conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    if gt.size == 0 or pred.size == 0:
        return conf_mat

    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)

    mask = (gt != ignore)
    gt = gt[mask]
    pred = pred[mask]
    if gt.size == 0:
        return conf_mat

    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    gt = gt[valid]
    pred = pred[valid]
    if gt.size == 0:
        return conf_mat

    np.add.at(conf_mat, (gt, pred), 1)
    return conf_mat


# =========================================================
# Drawing helpers (BEV)
# =========================================================
def draw_boxes(ax, centers, w=2.0, h=2.0, edgecolor="red", linewidth=2.5):
    if centers is None or centers.size == 0:
        return
    half_w, half_h = w / 2.0, h / 2.0
    for (cx, cy) in centers:
        rect = Rectangle((cx - half_w, cy - half_h), w, h,
                         fill=False, edgecolor=edgecolor, linewidth=linewidth)
        ax.add_patch(rect)


def draw_reflector_segments(ax, refl_centers, refl_dirs,
                            length=3.0, edgecolor="black", linewidth=2.0):
    if refl_centers is None or refl_dirs is None:
        return
    if refl_centers.size == 0 or refl_dirs.size == 0:
        return

    half = length / 2.0
    for c, d in zip(refl_centers, refl_dirs):
        d = d / (np.linalg.norm(d) + 1e-8)
        p1 = c - half * d
        p2 = c + half * d
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=edgecolor, linewidth=linewidth)


def draw_rays(ax, r1_points=None, pm_points=None,
              color_1st=(0.0, 0.6, 1.0),
              color_3rd=(0.85, 0.00, 0.45)):
    O = np.array([0.0, 0.0], dtype=np.float32)

    if r1_points is not None and r1_points.size > 0:
        for r1 in r1_points:
            ax.plot([O[0], r1[0]], [O[1], r1[1]],
                    color=color_1st, linewidth=2.0)
            ax.scatter([r1[0]], [r1[1]], s=50,
                       color=color_1st, linewidths=2.0)

    if pm_points is not None and pm_points.size > 0 and \
       r1_points is not None and r1_points.shape == pm_points.shape:
        for r1, pm in zip(r1_points, pm_points):
            ax.plot([O[0], r1[0]], [O[1], r1[1]],
                    color=color_3rd, linewidth=2.0)
            ax.plot([r1[0], pm[0]], [r1[1], pm[1]],
                    color=color_3rd, linewidth=2.0)
            ax.scatter([r1[0]], [r1[1]], s=50,
                       color=color_3rd, linewidths=2.0)
            ax.scatter([pm[0]], [pm[1]], s=50,
                       color=color_3rd, linewidths=1.5)


# =========================================================
# Plotting funcs (BEV)
# =========================================================
def plot_points_segmentation(save_path: str,
                             xy: np.ndarray,
                             pred_cls: np.ndarray | None,
                             xlim,
                             ylim):
    fig, ax = new_fixed_fig_ax()

    if xy.size > 0 and pred_cls is not None:
        for c in range(NUM_CLASSES):
            m = (pred_cls == c)
            if m.any():
                ax.scatter(
                    xy[m, 0], xy[m, 1],
                    s=25,
                    c=[PALETTE.get(c, (0.3, 0.3, 0.3))],
                    alpha=0.9,
                )
    elif xy.size > 0:
        ax.scatter(xy[:, 0], xy[:, 1], s=25, alpha=0.8, c="#7f7f7f")

    _apply_common_view(ax, xlim, ylim)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ensure_dir(str(Path(save_path).parent))
    fig.savefig(save_path, transparent=True, dpi=SAVE_DPI)
    plt.close(fig)


def render_surface_reconstruction_rgba(occ_map: np.ndarray,
                                       xlim,
                                       ylim,
                                       thr: float = 0.1) -> Image.Image:
    fig, ax = new_fixed_fig_ax()

    hm = np.clip(occ_map.astype(np.float32), 0.0, 1.0)
    hm_masked = np.ma.masked_where(hm <= thr, hm)

    cmap = plt.cm.Reds.copy()
    cmap.set_bad((0, 0, 0, 0))

    ax.imshow(
        hm_masked,
        origin="lower",
        cmap=cmap,
        vmin=thr,
        vmax=1.0,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        interpolation="nearest"
    )

    _apply_common_view(ax, xlim, ylim)
    ax.axis("off")
    fig.patch.set_alpha(0)

    img = fig_to_rgba_pil(fig)
    plt.close(fig)
    return img


def plot_surface_reconstruction(save_path: str,
                                occ_map: np.ndarray,
                                xlim,
                                ylim,
                                thr: float = 0.1):
    img = render_surface_reconstruction_rgba(occ_map, xlim, ylim, thr=thr)
    save_pil_rgba(img, save_path)


def plot_points_with_heatmap(save_path: str,
                             occ_map: np.ndarray,
                             xy: np.ndarray,
                             pred_cls: np.ndarray | None,
                             xlim,
                             ylim):
    fig, ax = new_fixed_fig_ax()

    ax.imshow(
        occ_map,
        origin="lower",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        interpolation="nearest"
    )

    if xy.size > 0:
        if pred_cls is None:
            ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.7, c="#7f7f7f")
        else:
            for c in range(NUM_CLASSES):
                m = (pred_cls == c)
                if m.any():
                    ax.scatter(
                        xy[m, 0], xy[m, 1],
                        s=20,
                        alpha=0.9,
                        c=[PALETTE.get(c, (0.3, 0.3, 0.3))]
                    )

    _apply_common_view(ax, xlim, ylim)
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ensure_dir(str(Path(save_path).parent))
    fig.savefig(save_path, transparent=True, dpi=SAVE_DPI)
    plt.close(fig)


def render_overlay_rgba(xy: np.ndarray,
                        pred_cls: np.ndarray | None,
                        centers_1st: np.ndarray,
                        centers_3rd_mir: np.ndarray | None,
                        refl_centers: np.ndarray | None,
                        refl_dirs: np.ndarray | None,
                        r1_points: np.ndarray | None,
                        pm_points: np.ndarray | None,
                        gt_centers: np.ndarray | None,
                        final_centers: np.ndarray | None,
                        xlim,
                        ylim) -> Image.Image:
    fig, ax = new_fixed_fig_ax()

    if xy.size > 0:
        if pred_cls is None:
            ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.7, c="#7f7f7f")
        else:
            for c in range(NUM_CLASSES):
                m = (pred_cls == c)
                if m.any():
                    ax.scatter(
                        xy[m, 0], xy[m, 1],
                        s=20,
                        alpha=0.8,
                        c=[PALETTE.get(c, (0.3, 0.3, 0.3))]
                    )

    if centers_1st is not None and centers_1st.size > 0:
        ax.scatter(centers_1st[:, 0], centers_1st[:, 1],
                   s=180, linewidths=2.0, c="blue")

    if centers_3rd_mir is not None and centers_3rd_mir.size > 0:
        ax.scatter(centers_3rd_mir[:, 0], centers_3rd_mir[:, 1],
                   s=200, linewidths=1.5,
                   edgecolors="black", facecolors="yellow")

    draw_reflector_segments(ax, refl_centers, refl_dirs,
                            length=3.0, edgecolor="black", linewidth=2.0)

    draw_rays(ax, r1_points=r1_points, pm_points=pm_points,
              color_1st=(0.0, 0.6, 1.0), color_3rd=(0.85, 0.0, 0.45))

    if gt_centers is not None and gt_centers.size > 0:
        draw_boxes(ax, gt_centers, w=2.0, h=2.0,
                   edgecolor="red", linewidth=2.5)

    if final_centers is not None and final_centers.size > 0:
        draw_boxes(ax, final_centers, w=2.0, h=2.0,
                   edgecolor="blue", linewidth=2.5)

    _apply_common_view(ax, xlim, ylim)
    ax.axis("off")
    fig.patch.set_alpha(0)

    img = fig_to_rgba_pil(fig)
    plt.close(fig)
    return img


def plot_overlay(save_path: str,
                 xy: np.ndarray,
                 pred_cls: np.ndarray | None,
                 centers_1st: np.ndarray,
                 centers_3rd_mir: np.ndarray | None,
                 refl_centers: np.ndarray | None,
                 refl_dirs: np.ndarray | None,
                 r1_points: np.ndarray | None,
                 pm_points: np.ndarray | None,
                 gt_centers: np.ndarray | None,
                 final_centers: np.ndarray | None,
                 xlim,
                 ylim):
    img = render_overlay_rgba(
        xy=xy,
        pred_cls=pred_cls,
        centers_1st=centers_1st,
        centers_3rd_mir=centers_3rd_mir,
        refl_centers=refl_centers,
        refl_dirs=refl_dirs,
        r1_points=r1_points,
        pm_points=pm_points,
        gt_centers=gt_centers,
        final_centers=final_centers,
        xlim=xlim,
        ylim=ylim,
    )
    save_pil_rgba(img, save_path)


def plot_final_composite(save_path: str,
                         occ_map: np.ndarray,
                         xy: np.ndarray,
                         pred_cls: np.ndarray | None,
                         centers_1st: np.ndarray,
                         centers_3rd_mir: np.ndarray | None,
                         refl_centers: np.ndarray | None,
                         refl_dirs: np.ndarray | None,
                         r1_points: np.ndarray | None,
                         pm_points: np.ndarray | None,
                         gt_centers: np.ndarray | None,
                         final_centers: np.ndarray | None,
                         xlim,
                         ylim,
                         thr: float):
    img_surface = render_surface_reconstruction_rgba(occ_map, xlim, ylim, thr=thr)
    img_overlay = render_overlay_rgba(
        xy=xy,
        pred_cls=pred_cls,
        centers_1st=centers_1st,
        centers_3rd_mir=centers_3rd_mir,
        refl_centers=refl_centers,
        refl_dirs=refl_dirs,
        r1_points=r1_points,
        pm_points=pm_points,
        gt_centers=gt_centers,
        final_centers=final_centers,
        xlim=xlim,
        ylim=ylim,
    )

    if img_surface.size != img_overlay.size:
        img_overlay = img_overlay.resize(img_surface.size, resample=Image.NEAREST)

    img_final = Image.alpha_composite(img_surface, img_overlay)
    save_pil_rgba(img_final, save_path)


# =========================================================
# CSV writers
# =========================================================
def save_points_csv(out_csv: str,
                    xy: np.ndarray,
                    gt: Optional[np.ndarray],
                    pred: np.ndarray,
                    prob: np.ndarray,
                    scene: str,
                    frame_idx: int):
    ensure_dir(str(Path(out_csv).parent))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x_m", "y_m", "gt_label", "pred_label", "pred_prob", "scene", "frame_idx"])
        if xy.size == 0:
            return
        if gt is None:
            gt = -1 * np.ones((xy.shape[0],), dtype=np.int32)
        for i in range(xy.shape[0]):
            w.writerow([
                float(xy[i, 0]), float(xy[i, 1]),
                int(gt[i]),
                int(pred[i]),
                float(prob[i]),
                scene,
                int(frame_idx),
            ])


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

    ap.add_argument("--config", type=str, default="",
                    help="JSON file with default values for any of the options below.")

    ap.add_argument("--root", type=str, default="./data_folder/test/dynamic")
    ap.add_argument("--ckpt", type=str, default="./checkpoints/dynamic/best.pt")
    ap.add_argument("--out_dir", type=str, default="./results/dynamic")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, dest="num_workers", default=4)
    ap.add_argument("--num_workers", type=int, dest="num_workers", required=False)
    ap.add_argument("--device_id", type=int, default=0)

    ap.add_argument("--bev_h", "--bev_H", type=int, dest="bev_h", default=128)
    ap.add_argument("--bev_w", "--bev_W", type=int, dest="bev_w", default=128)
    ap.add_argument("--x_min", type=float, default=-15.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--x_size", type=float, default=30.0)
    ap.add_argument("--y_size", type=float, default=30.0)

    ap.add_argument("--csv_dir", type=str, default="radar_data")

    ap.add_argument("--n_depth", type=int, default=32)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)

    # front image / annotation
    ap.add_argument("--img_h", type=int, default=720)
    ap.add_argument("--img_w", type=int, default=960)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--ann_filename", type=str, default="front_annotations.json")
    ap.add_argument("--front_dir_name", type=str, default="front_resized_img")
    ap.add_argument("--use_list_index_mapping", action="store_true")

    # true capture resolution of the polygons in front_annotations.json.
    # Do not leave this at -1: the fallback assumes img_h*4 / img_w*4, which is
    # only correct when the network input is exactly 1006 x 759.
    ap.add_argument("--ann_orig_h", type=int, default=3036)
    ap.add_argument("--ann_orig_w", type=int, default=4024)

    # wall heatmap threshold for rendering/composite
    ap.add_argument("--occ_thr", type=float, default=0.5)

    # ray tracing (cluster/reflector)
    ap.add_argument("--eps_1st", type=float, default=0.3)
    ap.add_argument("--eps_3rd", type=float, default=0.3)
    ap.add_argument("--min_samples", type=int, default=5)
    ap.add_argument("--min_samples_3rd", type=int, default=3)
    ap.add_argument("--merge_radius", type=float, default=1.5)

    ap.add_argument("--refl_len", type=float, default=3.5)
    ap.add_argument("--refl_thick", type=float, default=1.0)
    ap.add_argument("--refl_r_samples", type=int, default=20)
    ap.add_argument("--refl_theta_samples", type=int, default=18)
    ap.add_argument("--refl_dist_gamma", type=float, default=0.0,
                    help="range compensation for the reflector score: a candidate at "
                         "range r is scaled by (r/refl_dist_ref)**refl_dist_gamma. "
                         "0 disables it.")
    ap.add_argument("--refl_dist_ref", type=float, default=10.0,
                    help="reference range in metres for --refl_dist_gamma")
    ap.add_argument("--refl_min_valid", type=int, default=10,
                    help="minimum number of in-grid samples for a candidate segment")

    # temporal accumulation
    ap.add_argument("--accum_hist", type=int, default=3, help="use t-1..t-accum_hist heatmaps")
    ap.add_argument("--accum_alpha", type=float, default=0.75, help="decay factor per step")
    ap.add_argument("--accum_use_warp", action="store_true",
                    help="warp past heatmaps using wheel pose in meta (SE2). if pose missing, fallback to no-warp")
    ap.add_argument("--save_accum_heatmap", action="store_true",
                    help="also save accumulated heatmap images into surface_reconstruction_accum/")
    ap.add_argument("--accum_for_reflector_only", action="store_true",
                    help="if set, visualization uses raw occ_map, but reflector search uses occ_acc")

    # control
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--use_amp", action="store_true")

    return _apply_config_defaults(ap)


# =========================================================
# Main
# =========================================================
@torch.no_grad()
def main():
    args = build_args()

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] dataset: {DATASET_CLS.__name__}")

    ensure_dir(args.out_dir)
    save_json(vars(args), str(Path(args.out_dir) / "inference_args.json"))

    # annotation orig size
    if args.ann_orig_h > 0 and args.ann_orig_w > 0:
        ann_orig_hw = (int(args.ann_orig_h), int(args.ann_orig_w))
    else:
        ann_orig_hw = (int(args.img_h) * 4, int(args.img_w) * 4)
    ann_path = str(Path(args.root) / args.ann_filename)

    # dataset (robust kwargs)
    cand = {
        "root": args.root,
        "img_size": (args.img_h, args.img_w),
        "csv_dir_name": args.csv_dir,
        "csv_dir": args.csv_dir,
        "dt": float(args.dt),
        "n_hist": 3,
        "n_fut": 2,
        "ann_filename": args.ann_filename,
        "front_dir_name": args.front_dir_name,
        "use_list_index_mapping": bool(args.use_list_index_mapping),
        "ann_orig_hw": ann_orig_hw,
        "propagate_wall": True,
        "debug_print_first": False,
        "debug_sem": False,
        "require_img": True,
        "require_csv": True,
        "use_prev": True,
        "use_next": False,
    }
    ds_kwargs = filter_kwargs_for_callable(DATASET_CLS.__init__, cand)
    ds = DATASET_CLS(**ds_kwargs)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=COLLATE_FN,
        drop_last=False,
    )

    # BEV map helpers
    bev_hw = (int(args.bev_h), int(args.bev_w))
    grid_origin = (float(args.x_min), float(args.y_min))
    grid_size = (float(args.x_size), float(args.y_size))
    m2pix, cell_xy = make_m2pix(bev_hw, grid_origin, grid_size)
    _ = make_pix2m(bev_hw, grid_origin, grid_size)

    xlim = (float(args.x_min), float(args.x_min) + float(args.x_size))
    ylim = (float(args.y_min), float(args.y_min) + float(args.y_size))

    # model
    model = WheelOccXAttnLSS_Seg(
        bev_h=int(args.bev_h),
        bev_w=int(args.bev_w),
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
        use_presence_head=False,
    ).to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and ("model" in ckpt) else ckpt
    model.load_state_dict(state, strict=False)
    print(f"[LOAD] ckpt={args.ckpt}")

    # per-scene summary
    scen_conf: Dict[str, np.ndarray] = {}
    scen_loc: Dict[str, List[Dict[str, Any]]] = {}

    # per-scene heatmap history cache (temporal accumulation)
    # scene -> deque[(occ_map_tensor(1,1,H,W), pose(3))]
    scene_hist: Dict[str, Deque[Tuple[torch.Tensor, Optional[Tuple[float, float, float]]]]] = {}

    global_count = 0
    for batch in tqdm(dl, desc="Inference(seg+ray+accum)", ncols=120):
        batch = to_device(batch, device)

        img = batch["image"]                 # (B,3,H,W)
        radar_xy_t = batch["radar_xy_t"]     # list[(Ni,2)]
        radar_label_t = batch.get("radar_label_t", None)
        metas = batch.get("meta", None)
        if metas is None:
            metas = [{"scenario": "unknown_scene", "frame_idx": -1, "paths": {}} for _ in range(img.shape[0])]

        sem_mask = batch.get("sem_mask", None)
        sem_valid = batch.get("sem_valid", None)

        # segmentation flow: history frames only
        bev_r = build_bev_feature_from_history(
            batch=batch,
            bev_h=int(args.bev_h),
            bev_w=int(args.bev_w),
            x_min=float(args.x_min),
            y_min=float(args.y_min),
            x_size=float(args.x_size),
            y_size=float(args.y_size),
            use_offsets=[-3, -2, -1, 0],
        )

        with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
            out = model(
                img=img,
                bev_r=bev_r,
                radar_xy_list=radar_xy_t,
                K=None,
                T_cam2ego=None,
            )

        # wall prob (BEV) for current frame t
        wall_prob = torch.sigmoid(out["wall_bev_logit"])  # (B,1,H,W), torch
        # front sem pred (optional)
        sem_pred = None
        if ("front_sem_logits" in out) and (out["front_sem_logits"] is not None):
            sem_pred = out["front_sem_logits"].argmax(dim=1).detach().cpu().numpy()  # (B,H,W)
        # point logits list
        pt_logits_list = out.get("pt_logits", None)

        B = img.shape[0]
        for b in range(B):
            meta = metas[b] if b < len(metas) else {}
            scene_raw = str(meta.get("scenario", "unknown_scene"))
            scene_name = Path(scene_raw).name if scene_raw else "unknown_scene"
            frame_idx = int(meta.get("frame_idx", global_count))

            scn_root = Path(args.out_dir) / scene_name
            dir_seg   = scn_root / "points_segmentation"
            dir_surf  = scn_root / "surface_reconstruction"
            dir_hm    = scn_root / "points_with_heatmap"
            dir_ovr   = scn_root / "overlay"
            dir_final = scn_root / "Plot_final"
            dir_csv   = scn_root / "csv"
            dir_gtseg = scn_root / "GT_seg"
            dir_fseg  = scn_root / "front_seg"
            if bool(args.save_accum_heatmap):
                dir_surf_acc = scn_root / "surface_reconstruction_accum"
            else:
                dir_surf_acc = None

            for d in [dir_seg, dir_surf, dir_hm, dir_ovr, dir_final, dir_csv, dir_gtseg, dir_fseg]:
                ensure_dir(str(d))
            if dir_surf_acc is not None:
                ensure_dir(str(dir_surf_acc))

            base_name = f"{frame_idx:06d}.png"
            tag = f"{frame_idx:06d}"

            # ---------- points / preds ----------
            xy_t = radar_xy_t[b].detach().cpu().numpy() if radar_xy_t[b] is not None else np.zeros((0, 2), np.float32)
            gt_t = None
            if radar_label_t is not None:
                gt_t = radar_label_t[b].detach().cpu().numpy() if radar_label_t[b] is not None else None

            if (pt_logits_list is None) or (len(pt_logits_list) <= b) or (pt_logits_list[b] is None) or (pt_logits_list[b].numel() == 0):
                pred_cls = np.zeros((xy_t.shape[0],), dtype=np.int32)
                pred_prob = np.zeros((xy_t.shape[0],), dtype=np.float32)
            else:
                logits = pt_logits_list[b].detach().cpu()
                prob = F.softmax(logits, dim=1).numpy()
                pred_cls = prob.argmax(axis=1).astype(np.int32)
                pred_prob = prob.max(axis=1).astype(np.float32)

            # ---------- wall heatmap (raw, current t) ----------
            occ_t_torch = wall_prob[b:b+1, 0:1, :, :]  # (1,1,H,W) torch
            occ_map = occ_t_torch[0, 0].detach().cpu().numpy().astype(np.float32)

            # ---------- pose (wheel encoder / odom) ----------
            pose_cur = _try_get_pose_from_meta(meta)

            # ---------- temporal accumulation ----------
            if scene_name not in scene_hist:
                scene_hist[scene_name] = deque(maxlen=max(1, int(args.accum_hist)))
            hist = scene_hist[scene_name]

            # build accumulated map (torch -> numpy)
            occ_acc_torch = accumulate_occ_maps(
                occ_cur=occ_t_torch,
                pose_cur=pose_cur,
                history=hist,
                max_hist=int(args.accum_hist),
                alpha=float(args.accum_alpha),
                do_warp=bool(args.accum_use_warp),
                bev_h=int(args.bev_h), bev_w=int(args.bev_w),
                x_min=float(args.x_min), y_min=float(args.y_min),
                x_size=float(args.x_size), y_size=float(args.y_size),
            )
            occ_acc = occ_acc_torch[0, 0].detach().cpu().numpy().astype(np.float32)

            # update history AFTER using it
            hist.append((occ_t_torch.detach(), pose_cur))

            # which map to use for reflector search?
            occ_for_reflector = occ_acc if True else occ_map  # keep explicit
            if bool(args.accum_for_reflector_only):
                # viz: raw, reflector: accum
                occ_viz = occ_map
            else:
                # visualise the accumulated map (more stable)
                occ_viz = occ_acc
            # ---------- GT centers (for BEV boxes) ----------
            gt_centers = load_gt_positions_from_meta(meta)

            # =========================================================
            # ray-tracing postprocess (the reflector search uses occ_for_reflector)
            # =========================================================
            centers_1st = np.zeros((0, 2), dtype=np.float32)
            centers_3rd_mir = np.zeros((0, 2), dtype=np.float32)

            refl_centers_list = []
            refl_dirs_list = []
            r1_list = []
            pm_list = []

            if pred_cls is not None and xy_t.shape[0] > 0:
                mask_1st = (pred_cls == 1)   # 1st_ped
                mask_3rd = (pred_cls == 3)   # 3rd_ped

                xy_1st = xy_t[mask_1st]
                xy_3rd = xy_t[mask_3rd]

                # cluster 1st
                if xy_1st.size > 0:
                    lbl_1st = cluster_points(xy_1st, eps=float(args.eps_1st), min_samples=int(args.min_samples))
                    c_list = []
                    for cid in np.unique(lbl_1st):
                        if cid < 0:
                            continue
                        pts = xy_1st[lbl_1st == cid]
                        if pts.shape[0] == 0:
                            continue
                        c_list.append(pts.mean(axis=0))
                    if len(c_list) > 0:
                        centers_1st = np.stack(c_list, axis=0).astype(np.float32)

                # cluster 3rd + reflector search + mirror
                if xy_3rd.size > 0:
                    lbl_3rd = cluster_points(xy_3rd, eps=float(args.eps_3rd), min_samples=int(args.min_samples_3rd))
                    mir_list = []
                    for cid in np.unique(lbl_3rd):
                        if cid < 0:
                            continue
                        pts = xy_3rd[lbl_3rd == cid]
                        if pts.shape[0] == 0:
                            continue
                        c = pts.mean(axis=0).astype(np.float32)

                        refl_center, refl_dir = find_reflector_for_cluster(
                            center_3rd=c,
                            occ_map=occ_for_reflector,   # accumulated heatmap stabilises the reflector
                            m2pix=m2pix,
                            length_m=float(args.refl_len),
                            thickness_m=float(args.refl_thick),
                            n_r_samples=int(args.refl_r_samples),
                            n_theta_samples=int(args.refl_theta_samples),
                            dist_gamma=float(args.refl_dist_gamma),
                            dist_ref=float(args.refl_dist_ref),
                            min_valid=int(args.refl_min_valid),
                        )
                        if refl_center is None or refl_dir is None:
                            continue

                        refl_centers_list.append(refl_center.astype(np.float32))
                        refl_dirs_list.append(refl_dir.astype(np.float32))

                        pm = mirror_points_across_line(
                            points=c[None, :],
                            center=refl_center,
                            direction=refl_dir
                        )[0].astype(np.float32)

                        mir_list.append(pm)
                        r1_list.append(refl_center.astype(np.float32))
                        pm_list.append(pm)

                    if len(mir_list) > 0:
                        centers_3rd_mir = np.stack(mir_list, axis=0).astype(np.float32)

            refl_centers = np.stack(refl_centers_list, axis=0).astype(np.float32) if len(refl_centers_list) > 0 else np.zeros((0, 2), np.float32)
            refl_dirs    = np.stack(refl_dirs_list, axis=0).astype(np.float32)    if len(refl_dirs_list) > 0 else np.zeros((0, 2), np.float32)
            r1_points    = np.stack(r1_list, axis=0).astype(np.float32)           if len(r1_list) > 0 else np.zeros((0, 2), np.float32)
            pm_points    = np.stack(pm_list, axis=0).astype(np.float32)           if len(pm_list) > 0 else np.zeros((0, 2), np.float32)

            final_centers = merge_cluster_centers(
                centers_1st=centers_1st,
                centers_3rd_mir=centers_3rd_mir,
                merge_radius=float(args.merge_radius)
            )

            # confusion update (points)
            if gt_t is not None and pred_cls is not None and gt_t.size == pred_cls.size:
                prev = scen_conf.get(scene_name, None)
                scen_conf[scene_name] = confusion_update(prev, gt_t.copy(), pred_cls.copy(),
                                                         num_classes=NUM_CLASSES, ignore=-1)

            # localization summary
            gt_list = gt_centers.tolist() if gt_centers is not None and gt_centers.size > 0 else []
            pred_list = final_centers.tolist() if final_centers is not None and final_centers.size > 0 else []
            los_list = centers_1st.tolist() if centers_1st.size > 0 else []
            nlos_list = centers_3rd_mir.tolist() if centers_3rd_mir.size > 0 else []
            scen_loc.setdefault(scene_name, []).append(
                {"frame": frame_idx, "gt_positions": gt_list, "pred_positions": pred_list,
                 "pred_los": los_list, "pred_nlos": nlos_list}
            )

            # =========================================================
            # save BEV outputs
            # =========================================================
            plot_points_segmentation(
                save_path=str(dir_seg / base_name),
                xy=xy_t,
                pred_cls=pred_cls,
                xlim=xlim,
                ylim=ylim,
            )

            # surface reconstruction (viz uses occ_viz)
            plot_surface_reconstruction(
                save_path=str(dir_surf / base_name),
                occ_map=occ_viz,
                xlim=xlim,
                ylim=ylim,
                thr=float(args.occ_thr),
            )

            if dir_surf_acc is not None:
                plot_surface_reconstruction(
                    save_path=str(dir_surf_acc / base_name),
                    occ_map=occ_acc,
                    xlim=xlim,
                    ylim=ylim,
                    thr=float(args.occ_thr),
                )

            # points + heatmap (viz uses occ_viz)
            plot_points_with_heatmap(
                save_path=str(dir_hm / base_name),
                occ_map=occ_viz,
                xy=xy_t,
                pred_cls=pred_cls,
                xlim=xlim,
                ylim=ylim,
            )

            # overlay (BEV rays + boxes) - overlay doesn't depend on heatmap
            plot_overlay(
                save_path=str(dir_ovr / base_name),
                xy=xy_t,
                pred_cls=pred_cls,
                centers_1st=centers_1st,
                centers_3rd_mir=centers_3rd_mir if centers_3rd_mir.size > 0 else None,
                refl_centers=refl_centers if refl_centers.size > 0 else None,
                refl_dirs=refl_dirs if refl_dirs.size > 0 else None,
                r1_points=r1_points if r1_points.size > 0 else None,
                pm_points=pm_points if pm_points.size > 0 else None,
                gt_centers=gt_centers if gt_centers.size > 0 else None,
                final_centers=final_centers if final_centers.size > 0 else None,
                xlim=xlim,
                ylim=ylim,
            )

            # Plot_final = surface + overlay (viz uses occ_viz)
            plot_final_composite(
                save_path=str(dir_final / base_name),
                occ_map=occ_viz,
                xy=xy_t,
                pred_cls=pred_cls,
                centers_1st=centers_1st,
                centers_3rd_mir=centers_3rd_mir if centers_3rd_mir.size > 0 else None,
                refl_centers=refl_centers if refl_centers.size > 0 else None,
                refl_dirs=refl_dirs if refl_dirs.size > 0 else None,
                r1_points=r1_points if r1_points.size > 0 else None,
                pm_points=pm_points if pm_points.size > 0 else None,
                gt_centers=gt_centers if gt_centers.size > 0 else None,
                final_centers=final_centers if final_centers.size > 0 else None,
                xlim=xlim,
                ylim=ylim,
                thr=float(args.occ_thr),
            )

            # =========================================================
            # save front segmentation (pred + GT)
            # =========================================================
            front_rgb = read_front_from_tensor(img[b])

            pred_front_path = str(dir_fseg / f"{tag}.png")
            if sem_pred is not None:
                overlay_sem_on_front(front_rgb, sem_pred[b].astype(np.uint8), pred_front_path, alpha=0.35)
            else:
                Image.fromarray(front_rgb).save(pred_front_path)

            # GT overlay (prefer the dataset sem_mask, fall back to the JSON)
            gt_sem = None
            if (sem_mask is not None) and (sem_valid is not None):
                try:
                    if bool(sem_valid[b].detach().cpu().item()):
                        gt_sem = sem_mask[b].detach().cpu().numpy().astype(np.uint8)
                except Exception:
                    gt_sem = None
            if gt_sem is None:
                gt_sem = build_gt_mask_from_json(
                    ann_json_path=ann_path,
                    frame_idx=frame_idx,
                    out_hw=(int(args.img_h), int(args.img_w)),
                    ann_orig_hw=ann_orig_hw,
                )

            gt_front_path = str(dir_gtseg / f"{tag}.png")
            if gt_sem is None:
                Image.fromarray(front_rgb).save(gt_front_path)
            else:
                overlay_sem_on_front(front_rgb, gt_sem.astype(np.uint8), gt_front_path, alpha=0.35)

            # =========================================================
            # per-frame points csv
            # =========================================================
            save_points_csv(
                out_csv=str(dir_csv / f"points_{tag}.csv"),
                xy=xy_t,
                gt=gt_t,
                pred=pred_cls,
                prob=pred_prob,
                scene=scene_name,
                frame_idx=frame_idx,
            )

            global_count += 1
            if args.limit > 0 and global_count >= args.limit:
                print(f"[STOP] limit={args.limit}")
                break

        if args.limit > 0 and global_count >= args.limit:
            break

    # =========================================================
    # save per-scene confusion + localization summary
    # =========================================================
    for scn_name, conf in scen_conf.items():
        scn_root = Path(args.out_dir) / scn_name
        dir_csv = scn_root / "csv"
        ensure_dir(str(dir_csv))
        cm_path = dir_csv / "confusion_matrix.csv"
        if conf is None:
            conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with open(str(cm_path), "w", newline="") as f:
            wcsv = csv.writer(f)
            header = ["gt \\ pred"] + [str(c) for c in range(NUM_CLASSES)]
            wcsv.writerow(header)
            for g in range(NUM_CLASSES):
                row = [str(g)] + [int(conf[g, p]) for p in range(NUM_CLASSES)]
                wcsv.writerow(row)

    for scn_name, loc_list in scen_loc.items():
        scn_root = Path(args.out_dir) / scn_name
        dir_csv = scn_root / "csv"
        ensure_dir(str(dir_csv))
        loc_path = dir_csv / "localization_summary.csv"
        loc_list_sorted = sorted(loc_list, key=lambda d: d["frame"])
        with open(str(loc_path), "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["frame", "GT_positions", "Pred_positions",
                           "Pred_LoS", "Pred_NLOS"])
            for item in loc_list_sorted:
                wcsv.writerow([
                    int(item["frame"]),
                    repr(item["gt_positions"]),
                    repr(item["pred_positions"]),
                    repr(item.get("pred_los", [])),
                    repr(item.get("pred_nlos", [])),
                ])

    print("[DONE] out_dir =", args.out_dir)


if __name__ == "__main__":
    main()
