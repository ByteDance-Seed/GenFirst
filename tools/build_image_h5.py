#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Pack a folder of images into the HDF5 archive the end-to-end loaders expect.

The output contract is fixed by ``CustomINH5Dataset`` / ``CustomH5Dataset`` in
``dataset.py``:

    <dest>/images.h5
        00000/img00000000.png    uint8 array holding a PNG byte stream
        00000/img00000001.png
        ...
        dataset.json             uint8 array holding UTF-8 JSON of the form
                                 {"labels": [[arcname, class_index], ...]}
    <dest>/images_h5.json        JSON list of every item name inside images.h5

Example:
  python tools/build_image_h5.py \\
      --source /path/to/imagenet/train \\
      --dest data --resolution 256
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List

import h5py
import numpy as np
from tqdm import tqdm

# "tools" collides with an installed site-packages distribution, so the sibling
# module is imported by directory rather than as a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_prep import (  # noqa: E402
    LABEL_ITEM,
    build_entries,
    check_geometry,
    encode_png,
    init_worker,
    parse_resolution,
)


def put_bytes(archive: h5py.File, name: str, payload: bytes) -> None:
    """Store raw bytes at ``name``, creating intermediate groups as needed."""
    head, _, leaf = name.rpartition("/")
    target = archive.require_group(head) if head else archive
    target.create_dataset(leaf, data=np.frombuffer(payload, dtype=np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", "--imagenet-path", dest="source", required=True,
                        help="directory holding the source images")
    parser.add_argument("--dest", "--output-path", dest="dest", required=True,
                        help="output directory for images.h5")
    parser.add_argument("--resolution", default="256")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    resolution = parse_resolution(args.resolution)
    entries = build_entries(args.source, args.max_images)
    print(f"packing {len(entries)} images")

    os.makedirs(args.dest, exist_ok=True)
    archive_path = os.path.join(args.dest, "images.h5")
    listing_path = os.path.join(args.dest, "images_h5.json")

    labelled: List[List] = []
    skipped = 0
    verified = False

    with h5py.File(archive_path, "w") as archive:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker,
            initargs=(resolution,),
        ) as pool:
            stream = pool.map(encode_png, entries, chunksize=32)
            for entry, payload in tqdm(stream, total=len(entries), desc="packing"):
                if payload is None:
                    skipped += 1
                    continue
                if not verified:
                    check_geometry(payload, resolution)
                    verified = True
                put_bytes(archive, entry.arcname, payload)
                if entry.label is not None:
                    labelled.append([entry.arcname, entry.label])

        labelled.sort(key=lambda row: row[0])
        put_bytes(archive, LABEL_ITEM, json.dumps({"labels": labelled}).encode("utf-8"))

    names: List[str] = []
    with h5py.File(archive_path, "r") as archive:
        archive.visititems(lambda name, _obj: names.append(name))
    with open(listing_path, "w") as handle:
        json.dump(names, handle, indent=2)

    print(f"wrote {archive_path} ({len(labelled)} labelled, {skipped} skipped)")
    print(f"wrote {listing_path} ({len(names)} items)")


if __name__ == "__main__":
    main()
