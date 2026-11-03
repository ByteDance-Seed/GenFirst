# Copyright (c) 2024 Sihyun Yu
#
# From REPA (https://github.com/sihyun-yu/REPA), of which this directory is a
# fork. Licensed under the MIT license in Prior_only/LICENSE.

import os
import json
import random
import glob
import io
import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    def __init__(self, data_dir, feature_subdir='vae-sd'):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_dir = os.path.join(data_dir, 'images')
        self.features_dir = os.path.join(data_dir, feature_subdir)

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
            )
        # features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.features_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        features = np.load(os.path.join(self.features_dir, feature_fname))
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])

def _center_crop_resize(pil_img, resolution):
    w, h = pil_img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    pil_img = pil_img.crop((left, top, left + s, top + s))
    if s != resolution:
        pil_img = pil_img.resize((resolution, resolution), PIL.Image.BICUBIC)
    return pil_img


class ParquetT2IDataset(Dataset):
    """Image + caption dataset backed by parquet shards."""

    def __init__(
        self,
        data_roots,
        resolution=256,
        image_column="image",
        caption_columns=("caption", "prompt", "text"),
        rows_per_file=None,
        random_caption=True,
        max_files=None,
    ):
        if isinstance(data_roots, str):
            data_roots = [data_roots]
        self.resolution = int(resolution)
        self.image_column = image_column
        self.caption_columns = list(caption_columns)
        self.random_caption = random_caption
        self.rows_per_file = rows_per_file

        files = []
        for root in data_roots:
            if os.path.isdir(root):
                files.extend(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
            else:
                files.extend(glob.glob(root, recursive=True))
        files = sorted(set(files))
        if max_files is not None:
            files = files[:max_files]
        if len(files) == 0:
            raise FileNotFoundError(f"No parquet files found under: {data_roots}")

        import pyarrow.parquet as pq

        self.files = files
        self._offsets = [0]
        if rows_per_file is not None:
            for _ in files:
                self._offsets.append(self._offsets[-1] + int(rows_per_file))
        else:
            for f in files:
                n = pq.ParquetFile(f).metadata.num_rows
                self._offsets.append(self._offsets[-1] + n)
        self._total = self._offsets[-1]
        self._cache_path = None
        self._cache_cols = None

    def __len__(self):
        return self._total

    def _locate(self, idx):
        import bisect

        fi = bisect.bisect_right(self._offsets, idx) - 1
        fi = max(0, min(fi, len(self.files) - 1))
        local = idx - self._offsets[fi]
        return fi, local

    def _load_file(self, path):
        if self._cache_path == path and self._cache_cols is not None:
            return self._cache_cols
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        cols = {self.image_column: table.column(self.image_column).to_pylist()}
        for c in self.caption_columns:
            if c in table.column_names:
                cols[c] = table.column(c).to_pylist()
        self._cache_path = path
        self._cache_cols = cols
        return cols

    def _pick_caption(self, cols, local):
        available = []
        for c in self.caption_columns:
            if c in cols:
                v = cols[c][local]
                if isinstance(v, str) and len(v.strip()) > 0:
                    available.append(v.strip())
        if not available:
            return ""
        if self.random_caption:
            return random.choice(available)
        return available[0]

    def __getitem__(self, idx):
        fi, local = self._locate(idx)
        cols = self._load_file(self.files[fi])
        img_list = cols[self.image_column]
        if local >= len(img_list):
            local = local % len(img_list)
        raw = img_list[local]
        if isinstance(raw, dict) and "bytes" in raw:
            raw = raw["bytes"]
        try:
            pil = PIL.Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pil = PIL.Image.new("RGB", (self.resolution, self.resolution))
        pil = _center_crop_resize(pil, self.resolution)
        image = np.asarray(pil, dtype=np.uint8).transpose(2, 0, 1)
        caption = self._pick_caption(cols, local)
        return torch.from_numpy(image.copy()), caption