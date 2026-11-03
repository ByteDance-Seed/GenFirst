# Copyright (c) 2025 Xingjian Leng and Jaskirat Singh
#
# From REPA-E (https://github.com/End2End-Diffusion/REPA-E), which this tree
# derives from. Licensed under the MIT license in LICENSE.

import glob
import io
import json
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image


def load_h5_file(hf, path):
    # Helper function to load files from h5 file
    if path.endswith('.png'):
        rtn = np.array(PIL.Image.open(io.BytesIO(np.array(hf[path]))))
        rtn = rtn.reshape(*rtn.shape[:2], -1).transpose(2, 0, 1)
    elif path.endswith('.json'):
        rtn = json.loads(np.array(hf[path]).tobytes().decode('utf-8'))
    elif path.endswith('.npy'):
        rtn= np.array(hf[path])
    else:
        raise ValueError('Unknown file type: {}'.format(path))
    return rtn


class CustomINH5Dataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.data_dir = data_dir
        self.h5_path = os.path.join(self.data_dir, "images.h5")
        self.h5_json_path = os.path.join(self.data_dir, "images_h5.json")
        self.h5f = h5py.File(self.h5_path, 'r')

        with open(self.h5_json_path, 'r') as f:
            self.h5_json = json.load(f)
        self.filelist = {fname for fname in self.h5_json}
        self.filelist = sorted(fname for fname in self.filelist if self._file_ext(fname) in supported_ext)

        labels = load_h5_file(self.h5f, "dataset.json")["labels"]
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.filelist]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

    def __len__(self):
        return len(self.filelist)

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, index):
        """
        Images should be '.png'
        """
        image_fname = self.filelist[index]
        image = load_h5_file(self.h5f, image_fname)
        return torch.from_numpy(image), torch.tensor(self.labels[index])


class CustomH5Dataset(Dataset):
    def __init__(self, data_dir, vae_latents_name="latents"):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_h5 = h5py.File(os.path.join(data_dir, 'images.h5'), "r")
        self.features_h5 = h5py.File(os.path.join(data_dir, f'{vae_latents_name}.h5'), "r")
        images_json = os.path.join(data_dir, 'images_h5.json')
        features_json = os.path.join(data_dir, f'{vae_latents_name}_h5.json')

        with open(images_json, 'r') as f:
            images_json = json.load(f)
        with open(features_json, 'r') as f:
            features_json = json.load(f)

        # images
        self._image_fnames = {fname for fname in images_json}
        self.image_fnames = sorted(fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext)

        # features
        self._feature_fnames = {fname for fname in features_json}
        self.feature_fnames = sorted(fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext)
        
        # labels
        fname = 'dataset.json'
        labels = load_h5_file(self.features_h5, fname)['labels']
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

    def __del__(self):
        self.images_h5.close()
        self.features_h5.close()

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)

        image = load_h5_file(self.images_h5, image_fname)
        if image_ext == '.npy':
            # npy needs some extra care
            image = image.reshape(-1, *image.shape[-2:])

        features = load_h5_file(self.features_h5, feature_fname)
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])


def _center_crop_resize(pil_img, resolution):
    """Center-crop to square then resize to (resolution, resolution)."""
    w, h = pil_img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    pil_img = pil_img.crop((left, top, left + s, top + s))
    if s != resolution:
        pil_img = pil_img.resize((resolution, resolution), PIL.Image.BICUBIC)
    return pil_img


class ParquetT2IDataset(Dataset):
    """Image + caption dataset backed by parquet shards.

    Each row has an ``image`` column (encoded JPEG/PNG bytes, or an HF
    ``{"bytes": ...}`` struct) plus one or more caption columns. Returns
    ``(image_uint8_CHW, caption_str)``.
    """

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


class ImageFolderDataset(Dataset):
    """Class-conditional images laid out as ``root/<synset>/*.JPEG``.

    Labels are the index of the class folder in sorted order, which for ImageNet
    synsets (``n01440764`` first) is the canonical 0-999 id space -- the same one
    the class-name list induces on the c2i parquet shards, so a probe trained on
    the parquet train split can be evaluated here directly.

    Returns ``(image_uint8_CHW, label_int)``, matching ``ParquetClassDataset``.
    """

    def __init__(self, root, resolution=256, extensions=(".jpeg", ".jpg", ".png")):
        self.root = root
        self.resolution = int(resolution)
        classes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        if len(classes) == 0:
            raise FileNotFoundError(f"No class subdirectories under: {root}")
        self.classes = classes
        self.samples = []
        for label, cls in enumerate(classes):
            cdir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cdir)):
                if fname.lower().endswith(extensions):
                    self.samples.append((os.path.join(cdir, fname), label))
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images with {extensions} under: {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            pil = PIL.Image.open(path).convert("RGB")
        except Exception:
            pil = PIL.Image.new("RGB", (self.resolution, self.resolution))
        pil = _center_crop_resize(pil, self.resolution)
        image = np.asarray(pil, dtype=np.uint8).transpose(2, 0, 1)  # CHW
        return torch.from_numpy(image.copy()), torch.tensor(label, dtype=torch.long)


class ParquetClassDataset(Dataset):
    """Class-conditional image dataset backed by parquet shards."""

    def __init__(
        self,
        data_roots,
        resolution=256,
        image_column="image",
        label_column="class_label",
        label_map_file=None,
        rows_per_file=None,
        max_files=None,
    ):
        if isinstance(data_roots, str):
            data_roots = [data_roots]
        self.resolution = int(resolution)
        self.image_column = image_column
        self.label_column = label_column
        self.rows_per_file = rows_per_file
        self.name2idx = None
        if label_map_file:
            names = [ln.strip() for ln in open(label_map_file).read().splitlines()]
            self.name2idx = {n: i for i, n in enumerate(names)}

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
        cols = {
            self.image_column: table.column(self.image_column).to_pylist(),
            self.label_column: table.column(self.label_column).to_pylist(),
        }
        self._cache_path = path
        self._cache_cols = cols
        return cols

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
        raw_label = cols[self.label_column][local]
        if self.name2idx is not None and not isinstance(raw_label, (int, np.integer)):
            label = self.name2idx[str(raw_label).strip()]
        else:
            label = int(raw_label)
        return torch.from_numpy(image.copy()), label


class CaptionCollator:
    """Collate ``(image, caption)`` pairs into block-aligned tokenized text.

    Two different kinds of padding are produced and must not be confused:

    * *Within-sample* padding fills a caption's last block up to a multiple of
      ``block_size`` by repeating end-of-text. These are real supervised
      positions -- a block-wise generator has to emit something at every position
      of its final block, so it needs to learn what follows the end. The filler
      is end-of-text rather than ``<|pad|>`` because the pretrained text VAE
      reconstructs repeated end-of-text perfectly but decodes ``<|pad|>`` to
      ordinary words: Cola's no-padding layout means its VAE never saw that
      token in a reconstruction target.
    * *Cross-sample* padding extends short samples to the batch's longest
      sequence. These positions are not encoded by the text VAE at all and are
      marked invalid so attention and the losses skip them.

    Returns a dict with:
        ``images``      ``(B, 3, H, W)`` uint8
        ``text_ids``    list of ``(n_i,)`` LongTensors, each a multiple of
                        ``block_size``; this is the NA input the text VAE wants
        ``text_valid``  ``(B, L_pad)`` bool over the padded text grid
        ``captions``    the raw strings, for logging
    """

    def __init__(
        self,
        tokenizer_path,
        block_size=16,
        max_tokens=256,
        append_eos=True,
        eos_id=100257,
        fill_id=None,
    ):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(tokenizer_path)
        self.block_size = int(block_size)
        self.max_tokens = int(max_tokens)
        self.append_eos = append_eos
        self.eos_id = int(eos_id)
        self.fill_id = int(eos_id if fill_id is None else fill_id)

    def encode_one(self, caption):
        ids = self.tok.encode(caption or "").ids
        budget = self.max_tokens - (1 if self.append_eos else 0)
        ids = ids[:budget]
        if self.append_eos:
            ids = ids + [self.eos_id]
        bs = self.block_size
        n = ((len(ids) + bs - 1) // bs) * bs
        return ids + [self.fill_id] * (n - len(ids))

    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch])
        captions = [b[1] for b in batch]
        seqs = [self.encode_one(c) for c in captions]
        L_pad = max(len(s) for s in seqs)
        valid = torch.zeros(len(seqs), L_pad, dtype=torch.bool)
        for i, s in enumerate(seqs):
            valid[i, : len(s)] = True
        return dict(
            images=images,
            text_ids=[torch.tensor(s, dtype=torch.long) for s in seqs],
            text_valid=valid,
            captions=captions,
        )
