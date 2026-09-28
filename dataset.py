#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py - unified radar / camera dataset for the released splits.

- image: front_resized_img only (W=1006, H=759)
- annotation coords: original image space (W0=4024, H0=3036)
  -> scale to resized using sx = W/W0 (= 0.25), sy = H/H0 (= 0.25)

- frame mapping:
  resized frame index t  ->  json frame index = 2 * (t - base_front_frame)
  (base_front_frame = first frame number existing in front_resized_img folder)

sem_mask:
  0=bg, 1=wall, 2=ped
"""

from __future__ import annotations

import os, re, glob, json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset


# ============================================================================
#                           0) Common utilities
# ============================================================================
def _parse_frame_idx(fname: str, kind: str) -> Optional[int]:
    name = Path(fname).name
    if kind == "img":
        m = re.search(r"_frame_(\d+)\.(?:jpg|jpeg|png)$", name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d{4,})\.(?:jpg|jpeg|png)$", name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    if kind == "csv":
        for pat in [r"_(\d{4,})th_frame", r"_frame_(\d+)\.csv$", r"_(\d{4,})\.csv$"]:
            m = re.search(pat, name, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
    m = re.findall(r"(\d+)", name)
    return int(m[-1]) if m else None


def _list_by_idx(dir_path: Optional[str], exts: Tuple[str, ...], kind: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if (not dir_path) or (not os.path.isdir(dir_path)):
        return out
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith(exts):
            continue
        idx = _parse_frame_idx(fn, kind)
        if idx is None:
            continue
        out[idx] = os.path.join(dir_path, fn)
    return out


def _load_image(path: str, img_size: Optional[Tuple[int, int]]):
    # img_size: (H,W)
    with Image.open(path) as im:
        im = im.convert("RGB")
        if img_size is not None:
            im = im.resize((img_size[1], img_size[0]), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))  # (3,H,W)


def _try_float_col(df: pd.DataFrame, names: List[str]) -> Optional[np.ndarray]:
    for n in names:
        if n in df.columns:
            try:
                return df[n].astype(float).to_numpy()
            except Exception:
                pass
    return None


def _parse_label_column_extended(df: pd.DataFrame) -> np.ndarray:
    if df is None or len(df) == 0:
        return -1 * np.ones((0,), dtype=int)

    import re as _re
    norm = lambda s: _re.sub(r"[\s_]+", "", str(s).lower())
    by_norm = {norm(c): c for c in df.columns}

    cand_cols = ["ghost_class"]
    col = None
    for key in cand_cols:
        kn = norm(key)
        if kn in by_norm:
            col = by_norm[kn]
            break
    if col is None:
        return -1 * np.ones((len(df),), dtype=int)

    s = df[col].astype(str).str.strip().str.lower()
    s = s.str.replace(r"\s+", "", regex=True).str.replace("-", "_", regex=False)

    alias = {
        "none": "none",
        "1stbounce": "1st_bounce",
        "2ndbounce": "2nd_bounce",
        "3rdbounce": "3rd_bounce",
        "1stbouncesurface": "1st_bounce_surface",
        "3rdbouncesurface": "3rd_bounce_surface",
    }
    s_norm = s.map(lambda v: alias.get(v, v))
    MAP = {
        "none": 0,
        "1st_bounce": 1,
        "2nd_bounce": 2,
        "3rd_bounce": 3,
        "1st_bounce_surface": 4,
        "3rd_bounce_surface": 5,
    }
    return s_norm.map(MAP).fillna(-1).astype(int).to_numpy()


def _load_radar_csv(path: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)
    x = _try_float_col(df, ["x"])
    y = _try_float_col(df, ["y"])
    rcs = _try_float_col(df, ["rcs"])
    v = _try_float_col(df, ["v"])

    if x is None or y is None:
        n = 0
        return dict(
            x=np.zeros((n,)),
            y=np.zeros((n,)),
            rcs=np.zeros((n,)),
            vel=np.zeros((n,)),
            label=np.zeros((n,), dtype=int),
        )

    if rcs is None:
        rcs = np.zeros(len(df), dtype=float)
    if v is None:
        v = np.zeros(len(df), dtype=float)

    label = _parse_label_column_extended(df)
    n = min(len(x), len(y), len(rcs), len(v), len(label))
    return dict(
        x=x[:n].copy(),
        y=y[:n].copy(),
        rcs=rcs[:n].copy(),
        vel=v[:n].copy(),
        label=label[:n].copy(),
    )


# ============================================================================
#                           1) Wheel helper
# ============================================================================
def load_wheel_speed_map(scene_dir: str) -> Dict[int, float]:
    xls_list = glob.glob(os.path.join(scene_dir, "*wheel.xlsx"))
    if len(xls_list) == 0:
        return {}
    xls_path = sorted(xls_list)[-1]
    try:
        df = pd.read_excel(xls_path)
    except Exception:
        return {}

    frame_col = None
    for cand in ["frame", "Frame", "FRAME", "frame_idx", "frame number"]:
        if cand in df.columns:
            frame_col = cand
            break
    if frame_col is None:
        df = df.reset_index().rename(columns={"index": "frame"})
        frame_col = "frame"

    speed_col = None
    for cand in ["speed", "vel", "v", "vx", "spd", "wheel"]:
        if cand in df.columns:
            speed_col = cand
            break
    if speed_col is None:
        for c in df.columns:
            if c != frame_col and pd.api.types.is_numeric_dtype(df[c]):
                speed_col = c
                break
    if speed_col is None:
        return {}

    fmap = {}
    for _, row in df.iterrows():
        try:
            f = int(row[frame_col])
            s = float(row[speed_col])
            fmap[f] = s
        except Exception:
            continue
    return fmap


def cumulative_y_shift(speed_map: Dict[int, float], a: int, b: int, dt: float = 0.1) -> float:
    if a == b:
        return 0.0
    lo, hi = (a, b) if a < b else (b, a)
    s = 0.0
    for k in range(lo, hi):
        v = speed_map.get(k, 0.0)
        s += v * dt
    return s


# ============================================================================
#                           2) Semantic JSON helpers
# ============================================================================
def _load_json_any(path: str) -> Union[Dict[str, Any], List[Any]]:
    if (not path) or (not os.path.isfile(path)):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def _parse_front_frame_idx(fname: str) -> Optional[int]:
    name = Path(fname).name
    m = re.search(r"_frame_(\d+)\.(jpg|jpeg|png)$", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"(\d+)", name)
    return int(nums[-1]) if nums else None


def _list_front_frames(front_dir: str) -> List[int]:
    if not front_dir or (not os.path.isdir(front_dir)):
        return []
    idxs: List[int] = []
    for fn in os.listdir(front_dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        k = _parse_front_frame_idx(fn)
        if k is not None:
            idxs.append(k)
    idxs.sort()
    return idxs


def _poly_to_mask(polygons_xy: List[List[Tuple[float, float]]], out_hw: Tuple[int, int], value: int) -> np.ndarray:
    H, W = out_hw
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons_xy:
        if poly is None or len(poly) < 3:
            continue
        draw.polygon(poly, outline=value, fill=value)
    return np.asarray(mask_img, dtype=np.uint8)


def _points_flat_to_poly(points: Any) -> List[Tuple[float, float]]:
    if points is None or (not isinstance(points, list)) or len(points) < 6:
        return []
    if isinstance(points[0], (int, float)):
        if len(points) % 2 != 0:
            points = points[:-1]
        out = []
        for i in range(0, len(points), 2):
            out.append((float(points[i]), float(points[i + 1])))
        return out
    if isinstance(points[0], (list, tuple)) and len(points[0]) >= 2:
        return [(float(p[0]), float(p[1])) for p in points]
    return []


def _is_wall_label(lbl: str) -> bool:
    s = str(lbl).strip().lower()
    return ("wall" in s) or (s in ["fence", "barrier"])


def _is_ped_label(lbl: str) -> bool:
    s = str(lbl).strip().lower()
    return ("ped" in s) or ("person" in s) or (s in ["human", "pedestrian"])


def _extract_polys_from_item_any(item: Any) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    wall_polys: List[List[Tuple[float, float]]] = []
    ped_polys: List[List[Tuple[float, float]]] = []

    if isinstance(item, dict) and isinstance(item.get("shapes", None), list):
        for sh in item["shapes"]:
            if not isinstance(sh, dict):
                continue
            if str(sh.get("type", "")).lower() not in ["polygon", "polyline"]:
                continue
            lbl = sh.get("label", "")
            pts = _points_flat_to_poly(sh.get("points", None))
            if len(pts) < 3:
                continue
            if _is_wall_label(lbl):
                wall_polys.append(pts)
            elif _is_ped_label(lbl):
                ped_polys.append(pts)
        return wall_polys, ped_polys

    return wall_polys, ped_polys


@dataclass
class AnnSceneCache:
    ann_path: str
    ann_raw: Union[Dict[str, Any], List[Any]]
    front_frames: List[int]
    base_front_frame: Optional[int]
    use_list_index: bool
    frame_map: Dict[str, Dict[str, Any]]


def _build_frame_map_any(ann_raw: Union[Dict[str, Any], List[Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    # CVAT list based: aggregate shapes by frame
    if isinstance(ann_raw, list):
        frame_to_shapes: Dict[str, List[Dict[str, Any]]] = {}
        for entry in ann_raw:
            if not isinstance(entry, dict):
                continue
            shapes = entry.get("shapes", None)
            if not isinstance(shapes, list):
                continue
            for sh in shapes:
                if not isinstance(sh, dict) or ("frame" not in sh):
                    continue
                try:
                    fk = str(int(sh["frame"]))
                except Exception:
                    continue
                frame_to_shapes.setdefault(fk, []).append(sh)

        for fk, shapes in frame_to_shapes.items():
            out[fk] = {"shapes": shapes}
        return out

    # dict fallback (optional)
    if isinstance(ann_raw, dict):
        for k, v in ann_raw.items():
            if isinstance(v, dict) and isinstance(k, str) and re.fullmatch(r"\d+", k):
                out[k] = v
        frames = ann_raw.get("frames", None)
        if isinstance(frames, list):
            for it in frames:
                if not isinstance(it, dict) or ("frame" not in it):
                    continue
                try:
                    fk = str(int(it["frame"]))
                except Exception:
                    continue
                out[fk] = it
        return out

    return out


def _build_scene_ann_cache(scene_dir: str, ann_filename: str, front_dir_name: str, use_list_index: bool) -> AnnSceneCache:
    ann_path = os.path.join(scene_dir, ann_filename)
    ann_raw = _load_json_any(ann_path)

    front_dir = os.path.join(scene_dir, front_dir_name)
    front_frames = _list_front_frames(front_dir)
    base_front_frame = front_frames[0] if len(front_frames) > 0 else None

    frame_map = _build_frame_map_any(ann_raw)

    return AnnSceneCache(
        ann_path=ann_path,
        ann_raw=ann_raw,
        front_frames=front_frames,
        base_front_frame=base_front_frame,
        use_list_index=bool(use_list_index),
        frame_map=frame_map,
    )


def _lookup_frame_item(cache: AnnSceneCache, scene_name: str, json_frame: int) -> Optional[Dict[str, Any]]:
    k = str(int(json_frame))
    if k in cache.frame_map:
        return cache.frame_map[k]
    return None

def _make_sem_mask_for_scene_frame(
    cache: AnnSceneCache,
    scene_name: str,
    real_frame_idx: int,
    out_hw: Tuple[int, int],
    ann_orig_hw: Optional[Tuple[int, int]] = None,  # (H0,W0) = (3036,4024)
    propagate_wall: bool = True,
    frame_scale: int = 2,          # image frames per annotated json frame
    strict_even_only: bool = True, # annotations exist on even frames only
) -> Tuple[np.ndarray, bool]:
    H, W = out_hw

    t = int(real_frame_idx)

    # annotations exist on even frames only, so odd frames are marked invalid
    if strict_even_only and (t % frame_scale != 0):
        return np.zeros((H, W), dtype=np.uint8), False

    # note: divide the frame number itself instead of subtracting the base frame
    json_frame = t // int(frame_scale)
    if json_frame < 0:
        return np.zeros((H, W), dtype=np.uint8), False

    item = _lookup_frame_item(cache, scene_name, json_frame)
    if item is None:
        return np.zeros((H, W), dtype=np.uint8), False

    # scaling original -> resized
    if ann_orig_hw is not None:
        H0, W0 = ann_orig_hw  # (3036,4024)
        sx = float(W) / max(1.0, float(W0))
        sy = float(H) / max(1.0, float(H0))
    else:
        sx, sy = 1.0, 1.0

    def _clip_poly(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        out = []
        for x, y in poly:
            xx = float(np.clip(x, 0.0, W - 1.0))
            yy = float(np.clip(y, 0.0, H - 1.0))
            out.append((xx, yy))
        return out

    def _poly_area(poly: List[Tuple[float, float]]) -> float:
        if len(poly) < 3:
            return 0.0
        s = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    wall_polys, ped_polys = _extract_polys_from_item_any(item)

    # scale -> clip -> filter degenerate
    wall_polys = [_clip_poly([(x * sx, y * sy) for (x, y) in poly]) for poly in wall_polys]
    ped_polys  = [_clip_poly([(x * sx, y * sy) for (x, y) in poly]) for poly in ped_polys]

    wall_polys = [p for p in wall_polys if len(p) >= 3 and _poly_area(p) >= 5.0]
    ped_polys  = [p for p in ped_polys  if len(p) >= 3 and _poly_area(p) >= 5.0]

    sem = np.zeros((H, W), dtype=np.uint8)
    if propagate_wall and len(wall_polys) > 0:
        mw = _poly_to_mask(wall_polys, (H, W), value=1)
        sem[mw > 0] = 1
    if len(ped_polys) > 0:
        mp = _poly_to_mask(ped_polys, (H, W), value=2)
        sem[mp > 0] = 2
    return sem, True


    # return sem, True


# ============================================================================
#                           3) Dataset
# ============================================================================
class WheelDatasetSeg(Dataset):
    def __init__(
        self,
        root: str,
        # fixed input resolution
        img_size: Optional[Tuple[int, int]] = (759, 1006),  # (H,W)
        csv_dir_name: str = "radar_data",
        dt: float = 0.1,
        n_hist: int = 3,
        n_fut: int = 2,
        # semantic
        ann_filename: str = "front_annotations.json",
        front_dir_name: str = "front_resized_img",
        use_list_index_mapping: bool = False,  # unused in this setup
        # original annotation resolution (H0, W0)
        ann_orig_hw: Optional[Tuple[int, int]] = (3036, 4024),
        propagate_wall: bool = True,
        debug_print_first: bool = False,
        debug_sem: bool = False,
        # image frames per annotated frame
        frame_scale: int = 2,
    ):
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.csv_dir_name = csv_dir_name
        self.dt = float(dt)
        self.n_hist = int(n_hist)
        self.n_fut = int(n_fut)

        self.ann_filename = ann_filename
        self.front_dir_name = front_dir_name
        self.use_list_index_mapping = bool(use_list_index_mapping)
        self.ann_orig_hw = ann_orig_hw
        self.propagate_wall = bool(propagate_wall)
        self.debug_print_first = bool(debug_print_first)
        self.debug_sem = bool(debug_sem)
        self.frame_scale = int(frame_scale)
        self._debug_done = False

        rows = []
        for dirpath, _, _ in os.walk(root):
            scn = Path(dirpath)

            # use the front_resized_img folder only
            img_dir = scn / front_dir_name
            csv_dir = scn / csv_dir_name
            if not img_dir.is_dir() and not csv_dir.is_dir():
                continue

            img_map = _list_by_idx(str(img_dir), (".jpg", ".jpeg", ".png"), "img") if img_dir.is_dir() else {}
            csv_map = _list_by_idx(str(csv_dir), (".csv",), "csv") if csv_dir.is_dir() else {}

            inter = set(img_map.keys()) & set(csv_map.keys())
            if len(inter) == 0:
                continue

            rows.append((str(scn), sorted(inter), img_map, csv_map))
        rows.sort(key=lambda x: x[0])

        self.scenes: List[Dict[str, Any]] = []
        for scn_dir, idxs, img_map, csv_map in rows:
            speed_map = load_wheel_speed_map(scn_dir)
            self.scenes.append(dict(
                scn_dir=scn_dir,
                idxs=idxs,
                img_map=img_map,
                csv_map=csv_map,
                speed_map=speed_map,
            ))

        self.samples: List[Dict[str, Any]] = []
        for S in self.scenes:
            for fidx in S["idxs"]:
                self.samples.append(dict(scn=S["scn_dir"], frame_idx=fidx))

        self._scene_ann_cache: Dict[str, AnnSceneCache] = {}
        print(f"[WheelDatasetSeg] frames matched: {len(self.samples)} | img_size={self.img_size} | ann_orig_hw={self.ann_orig_hw} | frame_scale={self.frame_scale}")

    def __len__(self):
        return len(self.samples)

    def _find_scene_entry(self, scn_dir: str) -> Dict[str, Any]:
        for S in self.scenes:
            if S["scn_dir"] == scn_dir:
                return S
        raise KeyError(scn_dir)

    def _get_ann_cache(self, scene_dir: str) -> AnnSceneCache:
        if scene_dir in self._scene_ann_cache:
            return self._scene_ann_cache[scene_dir]
        cache = _build_scene_ann_cache(
            scene_dir=scene_dir,
            ann_filename=self.ann_filename,
            front_dir_name=self.front_dir_name,
            use_list_index=False,  # unused in this setup
        )
        self._scene_ann_cache[scene_dir] = cache

        if self.debug_sem:
            sn = Path(scene_dir).name
            print(f"[SEM CACHE] scene={sn}")
            print(f"  ann_path={cache.ann_path} exists={os.path.isfile(cache.ann_path)}")
            if cache.front_frames:
                print(f"  front_frames: n={len(cache.front_frames)} min={cache.front_frames[0]} max={cache.front_frames[-1]}")
            else:
                print("  front_frames: n=0")
            print(f"  base_front_frame={cache.base_front_frame}")
            print(f"  frame_map: n={len(cache.frame_map)}")
        return cache

    @staticmethod
    def _warp_frame_to_t(
        csv_map: Dict[int, str],
        speed_map: Dict[int, float],
        src_f: int,
        t: int,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if src_f not in csv_map:
            return np.zeros((0, 2), np.float32), -1 * np.ones((0,), np.int32)

        rd = _load_radar_csv(csv_map[src_f])
        x, y, lab = rd["x"], rd["y"], rd["label"]
        if x.size == 0:
            return np.zeros((0, 2), np.float32), -1 * np.ones((0,), np.int32)

        xy = np.stack([x, y], axis=1).astype(np.float32)
        lab = lab.astype(np.int32)

        if src_f < t:
            dy = cumulative_y_shift(speed_map, src_f, t, dt=dt)
            xy[:, 1] -= dy
        elif src_f > t:
            dy = cumulative_y_shift(speed_map, t, src_f, dt=dt)
            xy[:, 1] += dy
        return xy.astype(np.float32), lab

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.samples[i]
        scn_dir, t = rec["scn"], int(rec["frame_idx"])
        S = self._find_scene_entry(scn_dir)
        img_map, csv_map, speed_map = S["img_map"], S["csv_map"], S["speed_map"]

        # image (front_resized_img only)
        img_path = img_map.get(t, None)
        if img_path is None:
            img = torch.zeros((3, *(self.img_size or (759, 1006))), dtype=torch.float32)
        else:
            img = _load_image(img_path, self.img_size)

        # current t
        xy_t, lab_t = self._warp_frame_to_t(csv_map, speed_map, t, t, self.dt)

        out: Dict[str, Any] = {
            "image": img,
            "radar_xy_t": torch.from_numpy(xy_t),
            "radar_label_t": torch.from_numpy(lab_t),
            "meta": {
                "scenario": scn_dir,
                "frame_idx": t,
                "paths": {"img": img_map.get(t), "csv": csv_map.get(t)},
            }
        }

        # history
        for k in range(1, self.n_hist + 1):
            src = t - k
            xy, lab = self._warp_frame_to_t(csv_map, speed_map, src, t, self.dt)
            out[f"radar_xy_tm{k}_2t"] = torch.from_numpy(xy)
            out[f"radar_label_tm{k}_2t"] = torch.from_numpy(lab)

        # future (teacher)
        for k in range(1, self.n_fut + 1):
            src = t + k
            xy, lab = self._warp_frame_to_t(csv_map, speed_map, src, t, self.dt)
            out[f"radar_xy_tp{k}_2t"] = torch.from_numpy(xy)
            out[f"radar_label_tp{k}_2t"] = torch.from_numpy(lab)

        # semantic
        H, W = img.shape[-2], img.shape[-1]
        cache = self._get_ann_cache(scn_dir)
        scene_name = Path(scn_dir).name

        sem_np, sem_valid = _make_sem_mask_for_scene_frame(
            cache=cache,
            scene_name=scene_name,
            real_frame_idx=t,
            out_hw=(H, W),
            ann_orig_hw=self.ann_orig_hw,      # (3036,4024)
            propagate_wall=self.propagate_wall,
            frame_scale=self.frame_scale,
        )

        out["sem_mask"] = torch.from_numpy(sem_np.astype(np.int64))
        out["sem_valid"] = torch.tensor(bool(sem_valid), dtype=torch.bool)
        out["ped_present"] = torch.tensor(int((sem_np == 2).any()), dtype=torch.long)

        if self.debug_print_first and (not self._debug_done):
            print(f"[DEBUG] scene={scene_name} t={t} base_front={cache.base_front_frame} json_frame={self.frame_scale*(t-int(cache.base_front_frame))} sem_valid={bool(sem_valid)}")
            self._debug_done = True

        return out


def collate_fn_seg(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["image"] = torch.stack([b["image"] for b in batch], dim=0)
    out["meta"] = [b["meta"] for b in batch]

    out["radar_xy_t"] = [b["radar_xy_t"] for b in batch]
    out["radar_label_t"] = [b["radar_label_t"] for b in batch]

    for k in [1, 2, 3]:
        kx = f"radar_xy_tm{k}_2t"
        kl = f"radar_label_tm{k}_2t"
        if kx in batch[0]:
            out[kx] = [b[kx] for b in batch]
            out[kl] = [b[kl] for b in batch]

    for k in [1, 2]:
        kx = f"radar_xy_tp{k}_2t"
        kl = f"radar_label_tp{k}_2t"
        if kx in batch[0]:
            out[kx] = [b[kx] for b in batch]
            out[kl] = [b[kl] for b in batch]

    out["sem_mask"] = torch.stack([b["sem_mask"] for b in batch], dim=0)
    out["sem_valid"] = torch.stack([b["sem_valid"] for b in batch], dim=0)
    out["ped_present"] = torch.stack([b["ped_present"] for b in batch], dim=0)
    return out


# backwards-compatible aliases
WheelDataset = WheelDatasetSeg
collate_fn = collate_fn_seg
