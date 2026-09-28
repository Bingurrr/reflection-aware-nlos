# -*- coding: utf-8 -*-
"""Model definition for radar point segmentation with camera-guided BEV fusion.

This file is the concatenation of two modules from the research tree:
  1. WheelOccXAttnLSS      - the camera-LSS / radar-BEV cross-attention backbone
  2. WheelOccXAttnLSS_Seg  - the released wrapper that adds the segmentation,
                             reflective-surface and auxiliary heads

Wrapper model for:
  - Main: Radar point segmentation + Reflective surface estimation (BEV wall head)
  - Aux : Front semantic segmentation (wall/ped/bg) to train camera backbone feature
  - Aux : Ped presence head (optional)
  - Aux : FoV consistency regularization computed in train.py

It reuses WheelOccXAttnLSS (defined above in this file) as-is and only adds
extra heads and extra outputs.

Expected base interface (typical):
  - base.cam_lss : LSS module (has backbone; possibly depth_head/feat_head/splat internals)
  - base.rad_enc : radar BEV encoder
  - base.fuser   : BEV fusion module (cross-attn / deformable / etc.)
  - base.dec     : BEV decoder (optional occ head)
  - base.compute_point_logits(fused_bev, radar_xy_list) -> list[Tensor] (per sample point logits)

Outputs:
  - "pt_logits": list length B, each (Ni, C_pt) or None
  - "wall_bev_logit": (B,1,Hbev,Wbev)     # reflective surface estimation (MAIN)
  - "front_sem_logits": (B,3,Himg,Wimg)   # 0 bg, 1 wall, 2 ped (AUX)
  - "ped_present_logit": (B,1) or None    # (AUX)
  - "occ_logits": optional passthrough from base (if base has dec head)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================== Tiny image backbone & LSS ==============================

class TinyImgBackbone(nn.Module):
    """Small, stable CNN backbone (downsamples to H/4, W/4)."""
    def __init__(self, c: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.BatchNorm2d(2 * c), nn.GELU(),
            nn.Conv2d(2 * c, 2 * c, 3, 1, 1), nn.BatchNorm2d(2 * c), nn.GELU(),
        )
        self.out_ch = 2 * c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)  # (B, C', H/4, W/4)


class DepthHead(nn.Module):
    """Depth-bin probabilities (softmax along D)."""
    def __init__(self, in_ch: int, n_depth: int = 32):
        super().__init__()
        self.n_depth = n_depth
        self.head = nn.Conv2d(in_ch, n_depth, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        logits = self.head(f)                  # (B, D, Hs, Ws)
        return F.softmax(logits, dim=1)        # (B, D, Hs, Ws)


class FeatHead(nn.Module):
    """Project image features to the BEV channel width."""
    def __init__(self, in_ch: int, out_ch: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.out_ch = out_ch

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.proj(f)                    # (B, Cbev, Hs, Ws)


class CameraLSSProjector(nn.Module):
    """
    Monocular LSS with top-K depth bins (saves memory and time):
      - img -> backbone -> depth prob (D bins), img feature
      - back-project and splat only the top-K depth bins of every pixel

    Note:
      - without K / T_cam2ego, falls back to a simple pinhole IPM with a
        learnable scale
    """
    def __init__(
        self,
        bev_h: int, bev_w: int,
        x_min: float, y_min: float, x_size: float, y_size: float,
        n_depth: int = 32, z_min: float = 1.0, z_max: float = 50.0,
        c_backbone: int = 64, c_bev: int = 128,
        topk: int = 8
    ):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.x_min, self.y_min = x_min, y_min
        self.x_size, self.y_size = x_size, y_size
        self.n_depth = n_depth
        self.z_min, self.z_max = z_min, z_max
        self.topk = topk

        self.backbone = TinyImgBackbone(c=c_backbone)
        self.depth_head = DepthHead(self.backbone.out_ch, n_depth)
        self.feat_head  = FeatHead(self.backbone.out_ch, out_ch=c_bev)

        # learnable inverse focal scale (used by the fallback)
        self.register_parameter("inv_depth_scale", nn.Parameter(torch.tensor(1.0)))

    @staticmethod
    def _meshgrid(h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        return x, y  # (h,w)

    def forward(
        self,
        img: torch.Tensor,
        out_hw: tuple[int, int],
        K: torch.Tensor | None = None,
        T_cam2ego: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
          img: (B,3,H,W)
          out_hw: (Hbev, Wbev)
        Returns:
          F_bev: (B, Cbev, Hbev, Wbev)
        """
        B, _, H, W = img.shape
        Hb, Wb = out_hw
        f_img = self.backbone(img)          # (B,C',Hs,Ws)
        Dprob = self.depth_head(f_img)      # (B,D,Hs,Ws)
        Fimg  = self.feat_head(f_img)       # (B,Cbev,Hs,Ws)
        _, Cbev, Hs, Ws = Fimg.shape

        # depth bins
        z_bins = torch.linspace(self.z_min, self.z_max, self.n_depth, device=img.device)  # (D,)

        # feature-map pixel grid -> original pixel coordinates
        px, py = self._meshgrid(Hs, Ws, img.device)
        scale_y, scale_x = H / Hs, W / Ws
        u = (px + 0.5) * scale_x  # (Hs,Ws)
        v = (py + 0.5) * scale_y

        # select the top-K depth bins
        topk = min(self.topk, self.n_depth)
        pvals, didx = torch.topk(Dprob, k=topk, dim=1)  # (B,K,Hs,Ws), (B,K,Hs,Ws)

        # initialise the BEV grid
        F_bev = img.new_zeros((B, Cbev, Hb, Wb))

        cell_x = self.x_size / Wb
        cell_y = self.y_size / Hb

        # per-sample loop (simple and numerically stable)
        for b in range(B):
            # fallback model parameters
            fx = fy = (W * self.inv_depth_scale.abs().clamp(min=0.1))
            cx = W * 0.5
            cy = H * 0.6  # assume the camera looks below the horizon

            # (C, Npix)
            F_b = Fimg[b].reshape(Cbev, -1)
            out_b = img.new_zeros((Cbev, Hb * Wb))

            # prepare flat indices
            uu = u.reshape(-1)       # (Npix,)
            vv = v.reshape(-1)

            for k in range(topk):
                prob_k = pvals[b, k].reshape(-1)        # (Npix,)
                z_k    = z_bins[didx[b, k]].reshape(-1) # (Npix,)

                # camera -> ego (fallback pinhole)
                X = (uu - cx) / fx * z_k    # (Npix,)
                Y = z_k                     # forward axis

                ix = ((X - self.x_min) / cell_x).long()
                iy = ((Y - self.y_min) / cell_y).long()
                m = (ix >= 0) & (ix < Wb) & (iy >= 0) & (iy < Hb) & (prob_k > 1e-6)
                if m.sum() == 0:
                    continue

                idx_lin = (iy[m] * Wb + ix[m])  # (M,)
                p_sel = prob_k[m].unsqueeze(0)  # (1,M)
                F_sel = F_b[:, m] * p_sel       # (C,M)

                # scatter_add into BEV (shared column index)
                index = idx_lin.unsqueeze(0).expand(Cbev, -1)  # (C,M)
                out_b.scatter_add_(dim=1, index=index, src=F_sel)

            F_bev[b] = out_b.view(Cbev, Hb, Wb)

        return F_bev


# ============================== Radar encoder & Fusion & Decoder ==============================

class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=(2, 3), keepdim=True)
        s = (x - u).pow(2).mean(dim=(2, 3), keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.w[:, None, None] * x + self.b[:, None, None]


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd.floor_()
        return x / keep * rnd


class ConvNeXtBlock(nn.Module):
    """Lightweight ConvNeXt-style block."""
    def __init__(self, dim: int, drop_path: float = 0.0, ls_init: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)
        self.ls  = nn.Parameter(ls_init * torch.ones(dim))
        self.dp  = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.norm(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.pw2(h)
        return x + self.dp(h * self.ls[:, None, None])


class RadarBEVEncoder(nn.Module):
    """Encode the single-channel radar BEV into features."""
    def __init__(self, in_ch: int = 1, c: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, c // 2, 3, 1, 1), nn.GELU(),
            ConvNeXtBlock(c // 2),
            nn.Conv2d(c // 2, c, 3, 1, 1), nn.GELU(),
            ConvNeXtBlock(c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)  # (B, C, H, W)


# ============================== Low-Mem Cross-Attn (KV pooled) ==============================

class CrossAttnBlock(nn.Module):
    """Q: radar tokens, K/V: camera tokens."""
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=drop, batch_first=True
        )
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        qn = self.nq(q)
        kn = self.nk(k)
        out, _ = self.attn(qn, kn, kn)  # (B, Nq, C)
        q = q + self.dp(out)
        q = q + self.dp(self.mlp(q))
        return q


def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )
    half = dim // 2
    omega = torch.arange(half // 2, device=device).float()
    omega = 1.0 / (10000 ** (omega / (half // 2)))
    pe_x = torch.cat([torch.sin((x[..., None]) * omega),
                      torch.cos((x[..., None]) * omega)], dim=-1)
    pe_y = torch.cat([torch.sin((y[..., None]) * omega),
                      torch.cos((y[..., None]) * omega)], dim=-1)
    pe = torch.cat([pe_x, pe_y], dim=-1)  # (H, W, dim)
    return pe.view(1, h * w, dim)


class CrossFusionKVPool(nn.Module):
    """
    Memory-efficient cross-attention:
      - Q (radar) keeps the full resolution (H x W)
      - K/V (camera) are average-pooled by kv_stride (H/s x W/s)
    Cost ~ O(Nq * Nk); with s=4 the number of keys drops by 16x.
    """
    def __init__(self, c: int = 128, heads: int = 8, layers: int = 2,
                 kv_stride: int = 4, q_stride: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttnBlock(c, heads=heads) for _ in range(layers)])
        self.kv_stride = max(1, int(kv_stride))
        self.q_stride  = max(1, int(q_stride))  # optionally downsample Q as well (1 = keep)

    def _seq_with_pe(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        B, C, H, W = feat.shape
        pe = build_2d_sincos_pos_embed(H, W, C, feat.device)  # (1, HW, C)
        seq = feat.flatten(2).transpose(1, 2) + pe            # (B, HW, C)
        return seq, pe, H, W

    def forward(self, fr: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        fr: radar BEV (B,C,H,W)  -> Q
        fc: camera BEV (B,C,H,W) -> K/V (downsampled)
        """
        B, C, H, W = fr.shape

        # (1) Q: optional downsampling
        if self.q_stride > 1:
            Hq, Wq = H // self.q_stride, W // self.q_stride
            fr_q = F.avg_pool2d(fr, kernel_size=self.q_stride, stride=self.q_stride)  # (B,C,Hq,Wq)
        else:
            Hq, Wq = H, W
            fr_q = fr

        # (2) K/V: downsampling
        if self.kv_stride > 1:
            Hk, Wk = H // self.kv_stride, W // self.kv_stride
            fc_kv = F.avg_pool2d(fc, kernel_size=self.kv_stride, stride=self.kv_stride)  # (B,C,Hk,Wk)
        else:
            Hk, Wk = H, W
            fc_kv = fc

        # (3) tokenise and add positional encoding
        q, _, _, _ = self._seq_with_pe(fr_q)   # (B, Nq, C)
        k, _, _, _ = self._seq_with_pe(fc_kv)  # (B, Nk, C)

        # (4) L cross-attention blocks
        for blk in self.layers:
            q = blk(q, k)

        # (5) restore the Q resolution (upsample and add the residual if q_stride > 1)
        fused_q = q.transpose(1, 2).view(B, C, Hq, Wq)  # (B,C,Hq,Wq)
        if self.q_stride > 1:
            fused_q = F.interpolate(fused_q, size=(H, W), mode='bilinear', align_corners=False)
            fused_q = fused_q + fr
        return fused_q


class BEVDecoder(nn.Module):
    """Single-channel occupancy logit head."""
    def __init__(self, c: int = 128):
        super().__init__()
        self.dec = nn.Sequential(
            ConvNeXtBlock(c),
            nn.Conv2d(c, c // 2, 3, 1, 1), nn.GELU(),
            ConvNeXtBlock(c // 2),
            nn.Conv2d(c // 2, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(x)  # (B,1,H,W)


# ============================== Top model: Occ + Point Seg ==============================

class WheelOccXAttnLSS(nn.Module):
    """
    Inputs:
      img:            (B,3,Himg,Wimg)
      bev_r:          (B,1,Hbev,Wbev)  radar splat of the current frame t
      radar_xy_list:  list of length B, each (N_i, 2) [x, y] in metres
                      (wheel-compensated frame)
    Output (dict):
      {
        "occ_logits": (B,1,Hbev,Wbev),
        "pt_logits":  [ (N_i, C_cls) or None ]  # point-wise seg logits
      }
    """
    def __init__(
        self,
        bev_h: int, bev_w: int,
        x_min: float, y_min: float, x_size: float, y_size: float,
        bev_c: int = 128, heads: int = 8, layers: int = 2,
        n_depth: int = 32, topk: int = 8,
        num_classes: int = 6,   # 0..5 (none, 1st, 2nd, 3rd, 1st_wall, 3rd_wall)
    ):
        super().__init__()
        # BEV grid info (used to index points into the BEV grid)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_min = x_min
        self.y_min = y_min
        self.x_size = x_size
        self.y_size = y_size
        self.cell_x = x_size / bev_w
        self.cell_y = y_size / bev_h
        self.num_classes = num_classes

        # camera LSS
        self.cam_lss = CameraLSSProjector(
            bev_h, bev_w, x_min, y_min, x_size, y_size,
            n_depth=n_depth, c_backbone=64, c_bev=bev_c, topk=topk
        )
        # radar encoder
        self.rad_enc = RadarBEVEncoder(1, bev_c)
        # cross-attention fusion
        self.fuser   = CrossFusionKVPool(bev_c, heads=heads, layers=layers,
                                         kv_stride=4, q_stride=1)
        # occupancy head
        self.dec     = BEVDecoder(bev_c)
        # point segmentation head
        self.seg_head = nn.Conv2d(bev_c, num_classes, kernel_size=1)

    # ---------------------- BEV index helpers ---------------------- #
    def xy_to_bev_idx(self, xy: torch.Tensor):
        """
        xy: (N, 2) [x, y] in metres (wheel-compensated frame)
        return:
          iy, ix: (N,) long, clamped to 0..H-1 / 0..W-1
        """
        ix = ((xy[:, 0] - self.x_min) / self.cell_x).long()
        iy = ((xy[:, 1] - self.y_min) / self.cell_y).long()
        ix = ix.clamp(min=0, max=self.bev_w - 1)
        iy = iy.clamp(min=0, max=self.bev_h - 1)
        return iy, ix

    def compute_point_logits(
        self,
        bev_feat: torch.Tensor,
        radar_xy_list
    ):
        """
        bev_feat: (B, C_bev, H, W)  # fused BEV feature
        radar_xy_list: list of (N_i, 2) tensor on same device
        return: list of (N_i, num_classes) or None
        """
        B, C, H, W = bev_feat.shape
        assert H == self.bev_h and W == self.bev_w

        cls_map = self.seg_head(bev_feat)  # (B, C_cls, H, W)
        pt_logits_list = []

        for b, xy in enumerate(radar_xy_list):
            if (xy is None) or (xy.numel() == 0):
                pt_logits_list.append(None)
                continue

            # xy: (N,2)
            iy, ix = self.xy_to_bev_idx(xy)        # (N,)
            # (C_cls, N) -> (N, C_cls)
            logits_pts = cls_map[b, :, iy, ix].permute(1, 0).contiguous()
            pt_logits_list.append(logits_pts)

        return pt_logits_list

    # ------------------------------ forward ------------------------------ #
    def forward(
        self,
        img: torch.Tensor,
        bev_r: torch.Tensor,
        radar_xy_list=None,
        K: torch.Tensor | None = None,
        T_cam2ego: torch.Tensor | None = None
    ):
        """
        img: (B,3,Himg,Wimg)
        bev_r: (B,1,Hbev,Wbev)
        radar_xy_list: list of (N_i,2) tensors (optional; None skips the seg head)
        """
        B, _, H, W = bev_r.shape

        # 1) camera LSS BEV
        fc = self.cam_lss(img, (H, W), K=K, T_cam2ego=T_cam2ego)  # (B,C,H,W)

        # 2) radar BEV encoding
        fr = self.rad_enc(bev_r)                                   # (B,C,H,W)

        # 3) cross-attention fusion
        fused = self.fuser(fr, fc)                                 # (B,C,H,W)

        # 4) occupancy logits
        occ_logits = self.dec(fused)                               # (B,1,H,W)

        # 5) point-segmentation logits (optional)
        pt_logits_list = None
        if radar_xy_list is not None:
            pt_logits_list = self.compute_point_logits(fused, radar_xy_list)

        # return {
        #     "occ_logits": occ_logits,
        #     "pt_logits":  pt_logits_list,
        # }
        return {
            "occ_logits": occ_logits,
            "pt_logits":  pt_logits_list,
            # extra output
            "fused": fused,
            "fr": fr,
            "fc": fc,
        }


# ============================== Segmentation wrapper ==============================
# -------------------------
# Extra heads
# -------------------------
class FrontSemanticHead(nn.Module):
    """
    Lightweight semantic head on image-plane backbone feature.
    f_img: (B, C, Hs, Ws) -> logits (B, num_classes, Himg, Wimg)
    """
    def __init__(self, in_ch: int, num_classes: int = 3):
        super().__init__()
        mid = max(64, in_ch // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, num_classes, 1, bias=True),
        )

    def forward(self, f_img: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        logits_small = self.net(f_img)
        return F.interpolate(logits_small, size=out_hw, mode="bilinear", align_corners=False)


class PresenceHead(nn.Module):
    """Global pedestrian presence head from image-plane backbone feature."""
    def __init__(self, in_ch: int, hidden: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, f_img: torch.Tensor) -> torch.Tensor:
        x = f_img.mean(dim=(2, 3))  # GAP
        return self.fc(x)


class WallBEVHead(nn.Module):
    """Reflective surface estimation head from fused BEV feature."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, fused_bev_feat: torch.Tensor) -> torch.Tensor:
        return self.head(fused_bev_feat)  # (B,1,Hbev,Wbev)


# -------------------------
# Wrapper Model
# -------------------------
class WheelOccXAttnLSS_Seg(nn.Module):
    """
    Wrapper around WheelOccXAttnLSS to expose:
      - MAIN: point segmentation + wall BEV head
      - AUX : image semantic segmentation head on camera backbone feature
      - AUX : ped presence head (optional)
    """
    def __init__(
        self,
        bev_h: int, bev_w: int,
        x_min: float, y_min: float, x_size: float, y_size: float,
        bev_c: int = 128, heads: int = 8, layers: int = 2,
        n_depth: int = 32, topk: int = 8,
        num_pt_classes: int = 6,
        sem_num_classes: int = 3,
        use_presence_head: bool = True,
    ):
        super().__init__()

        # base model (unchanged)
        self.base = WheelOccXAttnLSS(
            bev_h=bev_h, bev_w=bev_w,
            x_min=x_min, y_min=y_min,
            x_size=x_size, y_size=y_size,
            bev_c=bev_c,
            heads=heads,
            layers=layers,
            n_depth=n_depth,
            topk=topk,
            num_classes=num_pt_classes,
        )

        # camera backbone out channels (best-effort)
        backbone_out_ch = None
        if hasattr(self.base, "cam_lss") and hasattr(self.base.cam_lss, "backbone"):
            if hasattr(self.base.cam_lss.backbone, "out_ch"):
                backbone_out_ch = int(self.base.cam_lss.backbone.out_ch)
            elif hasattr(self.base.cam_lss.backbone, "out_channels"):
                backbone_out_ch = int(self.base.cam_lss.backbone.out_channels)

        if backbone_out_ch is None:
            raise RuntimeError(
                "[model] Cannot infer camera backbone out channels. "
                "Expected base.cam_lss.backbone.out_ch (or out_channels)."
            )

        # extra heads
        self.front_sem_head = FrontSemanticHead(backbone_out_ch, num_classes=sem_num_classes)

        self.use_presence_head = bool(use_presence_head)
        self.presence_head = PresenceHead(backbone_out_ch) if self.use_presence_head else None

        # MAIN: reflective surface estimation
        self.wall_bev_head = WallBEVHead(in_ch=bev_c)

    # ---------- internal: build camera BEV feature (fc) without double backbone ----------
    def _cam_bev_from_backbone_feature(
        self,
        img: torch.Tensor,
        f_img: torch.Tensor,
        out_bev_hw: Tuple[int, int],
        K: Optional[torch.Tensor] = None,
        T_cam2ego: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Try to generate camera BEV feature (fc) using cam_lss internals to avoid recomputing backbone.
        If base cam_lss does not expose needed hooks, fall back to cam_lss(img, ...).
        """
        cam_lss = self.base.cam_lss

        # Preferred path: if cam_lss exposes depth_head/feat_head + a splat function that accepts them.
        # We try common naming patterns to be robust.
        has_depth = hasattr(cam_lss, "depth_head")
        has_feat  = hasattr(cam_lss, "feat_head")

        # candidate splat functions
        splat_fn = None
        for name in ["splat", "lift_splat", "lift_splat_bev", "splat_to_bev", "forward_splat"]:
            if hasattr(cam_lss, name):
                splat_fn = getattr(cam_lss, name)
                break

        if has_depth and has_feat and (splat_fn is not None):
            try:
                Dprob = cam_lss.depth_head(f_img)  # (B,D,Hs,Ws) or logits/prob
                Fimg  = cam_lss.feat_head(f_img)   # (B,Cbev,Hs,Ws)

                # Try calling splat in a few signature variants.
                # Variant A: splat(Dprob, Fimg, out_hw, K, T)
                try:
                    fc = splat_fn(Dprob, Fimg, out_bev_hw, K=K, T_cam2ego=T_cam2ego)
                    if isinstance(fc, torch.Tensor):
                        return fc
                except TypeError:
                    pass

                # Variant B: splat(Dprob, Fimg, out_hw)
                try:
                    fc = splat_fn(Dprob, Fimg, out_bev_hw)
                    if isinstance(fc, torch.Tensor):
                        return fc
                except TypeError:
                    pass

                # Variant C: splat(Dprob, Fimg, Hbev, Wbev, ...)
                Hbev, Wbev = out_bev_hw
                try:
                    fc = splat_fn(Dprob, Fimg, Hbev, Wbev, K=K, T_cam2ego=T_cam2ego)
                    if isinstance(fc, torch.Tensor):
                        return fc
                except TypeError:
                    pass

                # If all failed, fall back.
            except Exception:
                # fall back
                pass

        # Fallback: call cam_lss forward (may recompute backbone, but safe/correct).
        fc = cam_lss(img, out_bev_hw, K=K, T_cam2ego=T_cam2ego)
        return fc

    def forward(
        self,
        img: torch.Tensor,
        bev_r: torch.Tensor,
        radar_xy_list: Optional[List[torch.Tensor]] = None,
        K: Optional[torch.Tensor] = None,
        T_cam2ego: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
          img:   (B,3,Himg,Wimg)
          bev_r: (B,1,Hbev,Wbev) radar BEV input
          radar_xy_list: list length B, each (Ni,2) in meters (x,y)

        Returns:
          dict with keys: pt_logits, wall_bev_logit, front_sem_logits, ped_present_logit
          plus optional occ_logits if base.dec exists.
        """
        B = img.shape[0]
        Himg, Wimg = img.shape[-2], img.shape[-1]
        Hbev, Wbev = bev_r.shape[-2], bev_r.shape[-1]

        # ---- camera backbone feature (image-plane) ----
        f_img = self.base.cam_lss.backbone(img)  # (B,C',Hs,Ws)

        # ---- camera BEV feature (LSS) ----
        fc = self._cam_bev_from_backbone_feature(
            img=img,
            f_img=f_img,
            out_bev_hw=(Hbev, Wbev),
            K=K,
            T_cam2ego=T_cam2ego,
        )  # (B,Cbev,Hbev,Wbev)

        # ---- radar BEV encode ----
        fr = self.base.rad_enc(bev_r)  # (B,Cbev,Hbev,Wbev) usually

        # ---- fusion (BEV) ----
        fused = self.base.fuser(fr, fc)  # (B,Cbev,Hbev,Wbev)

        # ---- optional occupancy head passthrough ----
        occ_logits = None
        if hasattr(self.base, "dec") and (self.base.dec is not None):
            occ_logits = self.base.dec(fused)

        # ---- MAIN: point logits ----
        pt_logits_list = None
        if radar_xy_list is not None and hasattr(self.base, "compute_point_logits"):
            pt_logits_list = self.base.compute_point_logits(fused, radar_xy_list)

        # ---- MAIN: wall(surface) BEV head ----
        wall_bev_logit = self.wall_bev_head(fused)  # (B,1,Hbev,Wbev)

        # ---- AUX: front semantic head (image plane) ----
        front_sem_logits = self.front_sem_head(f_img, out_hw=(Himg, Wimg))  # (B,3,Himg,Wimg)

        # ---- AUX: presence head ----
        ped_present_logit = None
        if self.use_presence_head and (self.presence_head is not None):
            ped_present_logit = self.presence_head(f_img)  # (B,1)

        out: Dict[str, Any] = {
            "pt_logits": pt_logits_list,
            "wall_bev_logit": wall_bev_logit,
            "front_sem_logits": front_sem_logits,
            "ped_present_logit": ped_present_logit,
        }
        if occ_logits is not None:
            out["occ_logits"] = occ_logits
        return out
