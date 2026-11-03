# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Stitch DPG per-prompt samples into 2x2 grids for compute_dpg_bench.py.

Input : <src>/<item_id>_0.png .. <item_id>_3.png  (each RxR)
Output: <dst>/<item_id>.png                        (2R x 2R grid)
Quadrant order matches compute_dpg_bench crop_tuples: TL, TR, BL, BR.
"""
import argparse
import os
from collections import defaultdict

from PIL import Image


def main(args):
    groups = defaultdict(dict)
    for fn in os.listdir(args.src):
        if not fn.lower().endswith(".png"):
            continue
        stem, _ = os.path.splitext(fn)
        if "_" not in stem:
            continue
        item_id, idx = stem.rsplit("_", 1)
        if not idx.isdigit():
            continue
        groups[item_id][int(idx)] = os.path.join(args.src, fn)

    os.makedirs(args.dst, exist_ok=True)
    n = 0
    for item_id, d in groups.items():
        if len(d) < 4:
            print(f"WARN {item_id}: only {len(d)} samples, skip")
            continue
        imgs = [Image.open(d[i]).convert("RGB") for i in range(4)]
        R = imgs[0].size[0]
        grid = Image.new("RGB", (2 * R, 2 * R))
        grid.paste(imgs[0], (0, 0))       # TL
        grid.paste(imgs[1], (R, 0))       # TR
        grid.paste(imgs[2], (0, R))       # BL
        grid.paste(imgs[3], (R, R))       # BR
        grid.save(os.path.join(args.dst, f"{item_id}.png"))
        n += 1
    print(f">>> stitched {n} grids -> {args.dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    main(p.parse_args())
