---
license: cc-by-4.0
task_categories:
  - object-detection
tags:
  - radar
  - automotive
  - nlos
  - multipath
  - sensor-fusion
  - bird-eye-view
size_categories:
  - 10K<n<100K
---

# Radar NLOS Localization Dataset

Synchronized automotive radar, front camera and annotations for **non-line-of-sight
(NLOS) pedestrian localization through radar multipath**. Radar points carry a
`ghost_class` label that distinguishes direct returns from first-, second- and
third-bounce ghosts, and from returns off the reflecting surface itself. This is
the data release accompanying the code at `<GITHUB_URL>`.

Recordings come from two sites and several scene layouts, each with a moving
(`dynamic`) and a stationary (`static`) ego vehicle.

## Layout

```
data_folder/
├── manifest.json
├── train/dynamic/<TRIAL>/     57 trials, 4581 frames
├── train/static/<TRIAL>/      43 trials, 6003 frames
├── test/dynamic/<TRIAL>/      10 trials,  953 frames
└── test/static/<TRIAL>/        5 trials,  625 frames
```

Trial folders are named `PCD_Site<N>_Scene<N>_{Move,Stop}_Case<N>_Trial<N>` and
contain:

| Path | Content |
|---|---|
| `radar_data/ref_<frame>.csv` | radar points: `x, y, v, rcs, ghost_class, GT_Position` |
| `front_resized_img/*_frame_<frame>.jpg` | front camera, 960 x 724 |
| `front_annotations.json` | wall / pedestrian polygons, in 4024 x 3036 capture space |
| `BEV_synced/*_<frame>.jpg` | synchronized bird-eye-view camera, 1024 px long side |
| `*_wheel.xlsx` | wheel-encoder speed per frame (dynamic trials only) |

Frame indices are shared across modalities. A usable sample exists wherever a
frame index appears in **both** `radar_data` and `front_resized_img`; the camera
folders are already cropped to that intersection.

## Conventions

* Radar coordinates are metres in the ego frame: `x` lateral (right positive),
  `y` forward. The released models use a 30 x 30 m BEV grid spanning
  `x in [-15, 15]`, `y in [0, 30]`.
* `ghost_class`: `none`, `1st_bounce`, `2nd_bounce`, `3rd_bounce`,
  `1st_bounce_surface`, `3rd_bounce_surface`.
* `GT_Position` repeats the annotated pedestrian position(s) for the frame as a
  list of `[x, y]` pairs.
* Polygon annotations exist on **even frames only**; image frame `t` maps to
  annotation frame `t // 2`. Polygon coordinates are in the original
  4024 x 3036 capture space and must be rescaled to the network input size.
* Frame rate is 10 Hz (`dt = 0.1 s`).

## Coverage notes

`manifest.json` records, per trial, the frame count and index range plus
`has_annotations`, `has_wheel_speed` and `has_bev`.

* Five static training trials have no `front_annotations.json`. They still carry
  radar labels; the released training code masks their auxiliary semantic loss.
* `BEV_synced` is missing for 34 of the 115 trials, mostly in the static split.
  It is reference imagery and is not read by the model.
* One static training trial (`PCD_Site1_Scene1_Stop_Case4_Trial1`) has no
  `front_resized_img` and is therefore not part of the release.

## Usage

```bash
huggingface-cli download <REPO_ID> --repo-type dataset --local-dir .
```

Place `data_folder/` next to `train.py` from the code repository, then:

```bash
python train.py     --config configs/train_dynamic.json
python inference.py --config configs/inference_dynamic.json
```

## Not included

The raw capture also holds 4024 x 3036 front images, LiDAR point clouds and
radar BEV renderings. They are not required to reproduce the released results
and are not distributed here.
