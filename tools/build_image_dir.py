#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Write a folder of images into the cropped image directory the prior-only
stage reads.

The output contract is fixed by ``CustomDataset`` in ``Prior_only/dataset.py``,
which walks ``<data_dir>/images`` and pairs it with the latent cache by sorting
each file list independently:

    <dest>/00000/img00000000.png
    <dest>/00000/img00000001.png
    ...
    <dest>/dataset.json          {"labels": [[relpath, class_index], ...]}

``tools/encode_latents.py`` consumes this directory and mirrors its layout, so
the zero-padded names here are what keeps images and latents aligned.

Example:
  python tools/build_image_dir.py \\
      --source /path/to/imagenet/train \\
      --dest data/imagenet256/images \\
      --resolution=256x256 --transform=center-crop-dhariwal
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_prep import (  # noqa: E402
    LABEL_ITEM,
    build_entries,
    check_geometry,
    encode_png,
    init_worker,
    parse_resolution,
)

# The only crop this repository has ever used; kept as a flag so the documented
# command line still works.
TRANSFORM = "center-crop-dhariwal"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="directory holding the source images")
    parser.add_argument("--dest", required=True, help="output directory for the cropped images")
    parser.add_argument("--resolution", default="256")
    parser.add_argument("--transform", default=TRANSFORM, choices=[TRANSFORM])
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    resolution = parse_resolution(args.resolution)
    entries = build_entries(args.source, args.max_images)
    print(f"converting {len(entries)} images")

    os.makedirs(args.dest, exist_ok=True)
    labelled: List[List] = []
    skipped = 0
    verified = False

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(resolution,),
    ) as pool:
        stream = pool.map(encode_png, entries, chunksize=32)
        for entry, payload in tqdm(stream, total=len(entries), desc="converting"):
            if payload is None:
                skipped += 1
                continue
            if not verified:
                check_geometry(payload, resolution)
                verified = True
            out_path = os.path.join(args.dest, entry.arcname)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as handle:
                handle.write(payload)
            if entry.label is not None:
                labelled.append([entry.arcname, entry.label])

    labelled.sort(key=lambda row: row[0])
    with open(os.path.join(args.dest, LABEL_ITEM), "w") as handle:
        json.dump({"labels": labelled}, handle)

    print(f"wrote {args.dest} ({len(labelled)} labelled, {skipped} skipped)")


if __name__ == "__main__":
    main()
