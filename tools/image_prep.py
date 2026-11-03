#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Shared image discovery, labelling and cropping for the dataset builders.

Both dataset layouts in this repository start from the same raw ImageNet tree
and apply the same centre crop, so the pieces live here and the CLIs in
``build_image_h5.py``, ``build_image_dir.py`` and ``encode_latents.py`` stay
thin.

Class labels come from a ``dataset.json`` in the source root when one exists,
otherwise they are derived from the first path component of each image (the
usual ``<split>/<wnid>/<file>.JPEG`` layout), assigning indices in sorted order.
"""

import io
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import center_crop_arr  # noqa: E402

LABEL_ITEM = "dataset.json"
GROUP_WIDTH = 5
INDEX_WIDTH = 8


@dataclass(frozen=True)
class Entry:
    """One image, as it will appear in the output."""

    index: int
    path: str  # absolute path on disk
    rel: str  # posix path relative to the source root
    label: Optional[int]

    @property
    def arcname(self) -> str:
        stem = f"{self.index:0{INDEX_WIDTH}d}"
        return f"{stem[:GROUP_WIDTH]}/img{stem}.png"


def image_suffixes() -> frozenset:
    Image.init()
    return frozenset(Image.EXTENSION)


def discover(root: str, suffixes: frozenset) -> List[str]:
    """Return every image path under ``root``, relative and posix-normalised."""
    found = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in suffixes:
                continue
            abs_path = os.path.join(dirpath, name)
            found.append(os.path.relpath(abs_path, root).replace(os.sep, "/"))
    found.sort()
    return found


def labels_from_sidecar(root: str) -> Dict[str, int]:
    """Read ``<root>/dataset.json`` if present, else return an empty mapping."""
    sidecar = os.path.join(root, LABEL_ITEM)
    if not os.path.isfile(sidecar):
        return {}
    with open(sidecar, "r") as handle:
        payload = json.load(handle)
    pairs = payload.get("labels") or []
    return {rel: label for rel, label in pairs}


SPLIT_NAMES = frozenset({"train", "val", "valid", "validation", "test"})


def labels_from_layout(rels: List[str]) -> Dict[str, int]:
    """Assign a class index per top-level directory, in sorted order."""
    tops = {rel: rel.split("/")[0] if "/" in rel else "" for rel in rels}
    distinct = sorted(set(tops.values()))
    if distinct == [""]:
        return {}
    if SPLIT_NAMES & set(distinct):
        # Pointing at the parent of train/ and val/ would otherwise turn the
        # split into the class label and fold the validation set into training.
        raise SystemExit(
            f"top-level directories {distinct[:5]} look like dataset splits, not "
            "classes; pass the split directory itself, e.g. --source .../train"
        )
    index_of = {name: i for i, name in enumerate(distinct)}
    return {rel: index_of[top] for rel, top in tops.items()}


def build_entries(root: str, limit: Optional[int]) -> List[Entry]:
    rels = discover(root, image_suffixes())
    if not rels:
        raise SystemExit(f"no images found under {root}")

    # Labels are resolved over the whole tree before --max-images truncates it,
    # so a subset keeps the class indices of the full dataset.
    table = labels_from_sidecar(root) or labels_from_layout(rels)
    missing = [rel for rel in rels if rel not in table]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(rels)} images have no class label (first: "
            f"{missing[0]}); lay images out as <source>/<class>/<file> or provide "
            f"<source>/{LABEL_ITEM}"
        )
    print(f"found {len(rels)} images over {len(set(table[r] for r in rels))} classes under {root}")

    if limit is not None:
        rels = rels[:limit]
    return [
        Entry(index=i, path=os.path.join(root, rel), rel=rel, label=table[rel])
        for i, rel in enumerate(rels)
    ]


def parse_resolution(text: str) -> int:
    """Accept either ``256`` or the ``256x256`` spelling used by the old CLI."""
    if "x" in text.lower():
        width, _, height = text.lower().partition("x")
        if width != height:
            raise SystemExit(f"only square resolutions are supported, got {text}")
        text = width
    try:
        value = int(text)
    except ValueError:
        raise SystemExit(f"could not parse resolution {text!r}")
    if value & (value - 1):
        raise SystemExit(f"resolution must be a power of two, got {value}")
    return value


_RESOLUTION = None


def init_worker(resolution: int) -> None:
    global _RESOLUTION
    _RESOLUTION = resolution
    Image.init()


def encode_png(entry: Entry) -> Tuple[Entry, Optional[bytes]]:
    """Crop, resize and PNG-encode one image. ``None`` payload means skip."""
    try:
        with Image.open(entry.path) as handle:
            pixels = np.array(handle.convert("RGB"))
    except (OSError, ValueError):
        return entry, None

    pixels = center_crop_arr(pixels, _RESOLUTION)
    if pixels is None:
        return entry, None

    buffer = io.BytesIO()
    # compress_level=0 keeps packing cheap; the data is read sequentially and
    # decode speed matters more than bytes on disk.
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=0, optimize=False)
    return entry, buffer.getvalue()


def check_geometry(payload: bytes, resolution: int) -> None:
    probe = np.array(Image.open(io.BytesIO(payload)))
    height, width = probe.shape[:2]
    if (height, width) != (resolution, resolution):
        raise SystemExit(f"expected {resolution}x{resolution} after crop, got {width}x{height}")
