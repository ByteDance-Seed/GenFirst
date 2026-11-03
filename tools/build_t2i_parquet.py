#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Build parquet shards for T2I training from images + captions.

Expected schema (matches ParquetT2IDataset):
  - image: binary JPEG/PNG bytes
  - one or more caption columns (strings), e.g. caption / prompt / text

Example:
  python tools/build_t2i_parquet.py \\
      --images-dir data/my_images \\
      --captions captions.jsonl \\
      --out-dir data/my_parquet \\
      --caption-column caption \\
      --rows-per-file 128
"""

import argparse
import glob
import io
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def expand_braces(pattern):
    start = pattern.find("{")
    end = pattern.find("}", start + 1)
    if start == -1 or end == -1:
        return [pattern]
    prefix, suffix = pattern[:start], pattern[end + 1:]
    return [p for alt in pattern[start + 1:end].split(",")
            for p in expand_braces(prefix + alt.strip() + suffix)]


def load_captions(path):
    ext = os.path.splitext(path)[1].lower()
    mapping = {}
    if ext == ".jsonl":
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                key = row.get("file_name") or row.get("filename") or row.get("id")
                cap = row.get("caption") or row.get("text") or row.get("prompt")
                if key and cap:
                    mapping[os.path.basename(key)] = cap
    elif ext == ".json":
        data = json.load(open(path))
        if isinstance(data, list):
            for row in data:
                key = row.get("file_name") or row.get("filename") or row.get("id")
                cap = row.get("caption") or row.get("text") or row.get("prompt")
                if key and cap:
                    mapping[os.path.basename(key)] = cap
        else:
            mapping = {os.path.basename(k): v for k, v in data.items()}
    else:
        with open(path) as f:
            for line in f:
                if "\t" in line:
                    fname, cap = line.rstrip("\n").split("\t", 1)
                elif "," in line:
                    fname, cap = line.rstrip("\n").split(",", 1)
                else:
                    continue
                mapping[os.path.basename(fname.strip())] = cap.strip()
    return mapping


def encode_image(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        return buf.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--captions", type=str, required=True,
                   help="Captions file: .jsonl, .json, .tsv, or .csv (filename<TAB>caption).")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--caption-column", type=str, default="caption",
                   help="Parquet column name for the caption string.")
    p.add_argument("--rows-per-file", type=int, default=128)
    p.add_argument("--glob", type=str, default="*.{jpg,jpeg,png,webp}",
                   dest="image_glob")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    caps = load_captions(args.captions)

    # glob has no brace expansion, so "*.{jpg,png}" has to be split by hand.
    patterns = expand_braces(args.image_glob)
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(args.images_dir, pat)))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.images_dir} with pattern {args.image_glob}")

    batch_images, batch_caps = [], []
    shard_idx = 0
    written = 0

    def flush():
        nonlocal shard_idx, batch_images, batch_caps, written
        if not batch_images:
            return
        table = pa.table({
            "image": batch_images,
            args.caption_column: batch_caps,
        })
        out_path = os.path.join(args.out_dir, f"shard-{shard_idx:05d}.parquet")
        pq.write_table(table, out_path)
        written += len(batch_images)
        shard_idx += 1
        batch_images, batch_caps = [], []

    for path in image_paths:
        fname = os.path.basename(path)
        if fname not in caps:
            continue
        batch_images.append(encode_image(path))
        batch_caps.append(caps[fname])
        if len(batch_images) >= args.rows_per_file:
            flush()
    flush()
    print(f"Wrote {written} rows in {shard_idx} shards -> {args.out_dir}")


if __name__ == "__main__":
    main()
