# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Streaming parquet image data for multi-stage end-to-end VAE training.

Reads parquet shards from local directories or HDFS (if `hdfs://` URIs are used),
mixes sources by weight, and emits aspect-ratio-bucketed batches at a fixed pixel
area:

    every batch is [B, 3, H, W] uint8 with H*W ~= side**2, H and W multiples of
    `multiple` (16 = VAE f8 x DiT patch 2), aspect ratio within [1/2, 2].

Different ranks may emit different buckets on the same step; that is fine for
DDP and SyncBatchNorm since every bucket has about the same token count.

File lists are built once by `scripts/build_vae_filelists.py` and read here.
"""

import fnmatch
import glob
import io
import json
import math
import os
import random
import subprocess

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

Image.MAX_IMAGE_PIXELS = None


def ensure_hdfs_env():
    """Configure libhdfs environment variables when HADOOP_HOME is set."""
    hadoop_home = os.environ.get("HADOOP_HOME", "")
    if not hadoop_home or not os.path.isdir(hadoop_home):
        return
    if "hadoop" not in os.environ.get("CLASSPATH", ""):
        hadoop_bin = os.path.join(hadoop_home, "bin", "hadoop")
        if os.path.isfile(hadoop_bin):
            cp = subprocess.check_output([hadoop_bin, "classpath", "--glob"], text=True).strip()
            os.environ["CLASSPATH"] = cp
    os.environ.setdefault("ARROW_LIBHDFS_DIR", os.path.join(hadoop_home, "lib", "native"))


def split_data_uri(uri):
    """Parse either a local path pattern or an 'hdfs://host/path/*.parquet' URI."""
    if uri.startswith("hdfs://"):
        host, path = uri[len("hdfs://"):].split("/", 1)
        root, pattern = os.path.split("/" + path)
        return host, root, pattern
    root, pattern = os.path.split(os.path.abspath(uri))
    if not pattern or ("*" not in pattern and os.path.isdir(uri)):
        root = os.path.abspath(uri)
        pattern = "*.parquet"
    return None, root, pattern


def _strip_host(path):
    if "://" in path:
        path = path.split("://", 1)[1]
    return path if path.startswith("/") else path[path.index("/"):]


def hdfs_fs(host):
    import pyarrow.fs as pfs
    return pfs.HadoopFileSystem(host)


def load_sources(data_json):
    with open(data_json) as f:
        sources = json.load(f)
    for s in sources:
        s.setdefault("weight", 1.0)
        s["host"], s["root"], s["pattern"] = split_data_uri(s["data_path"])
    return sources


def filelist_path(filelist_dir, source):
    return os.path.join(filelist_dir, f"{source['dataset_name']}.txt")


def build_filelist(source, out_path, max_files=50000, seed=0):
    if source["host"] is None:
        matched = sorted(glob.glob(os.path.join(source["root"], source["pattern"])))
        total_infos = len(matched)
        paths = matched
    else:
        import pyarrow.fs as pfs
        fs = hdfs_fs(source["host"])
        infos = fs.get_file_info(pfs.FileSelector(source["root"], recursive=False))
        total_infos = len(infos)
        paths = [_strip_host(i.path) for i in infos if i.type == pfs.FileType.File]
        paths = [p for p in paths if fnmatch.fnmatch(os.path.basename(p), source["pattern"])]
        paths.sort()
    if max_files and len(paths) > max_files:
        paths = random.Random(seed).sample(paths, max_files)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write("\n".join(paths) + "\n")
    os.replace(tmp, out_path)
    return total_infos, len(paths)


# ----------------------------------------------------------------------------
# Buckets / image prep
# ----------------------------------------------------------------------------
def make_buckets(side, multiple=16, max_ratio=2.0):
    area = side * side
    buckets = set()
    h = multiple
    while h * h <= area * max_ratio + 1:
        w = int(round(area / h / multiple)) * multiple
        if w > 0 and 1.0 / max_ratio <= h / w <= max_ratio:
            buckets.add((h, w))
        h += multiple
    return sorted(buckets)


def _to_rgb(im):
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


class _Prep:
    def __init__(self, side, multiple, max_ratio, min_area_ratio, random_crop):
        self.buckets = make_buckets(side, multiple, max_ratio)
        self.log_ratios = np.array([math.log(h / w) for h, w in self.buckets])
        self.min_area = min_area_ratio * side * side
        self.max_ratio = max_ratio
        self.random_crop = random_crop

    def __call__(self, raw, rng):
        im = Image.open(io.BytesIO(raw))
        w, h = im.size
        if w * h < self.min_area or max(w, h) / min(w, h) > self.max_ratio * 1.5:
            return None
        bh, bw = self.buckets[int(np.abs(self.log_ratios - math.log(h / w)).argmin())]
        s = max(bw / w, bh / h)
        nw, nh = max(bw, math.ceil(w * s)), max(bh, math.ceil(h * s))
        im.draft("RGB", (nw, nh))  # JPEG DCT-domain downscale; result stays >= (nw, nh)
        im = _to_rgb(im).resize((nw, nh), Image.BICUBIC)
        if self.random_crop:
            left, top = rng.randint(0, nw - bw), rng.randint(0, nh - bh)
        else:
            left, top = (nw - bw) // 2, (nh - bh) // 2
        im = im.crop((left, top, left + bw, top + bh))
        arr = np.asarray(im, dtype=np.uint8)
        return (bh, bw), torch.from_numpy(arr.copy()).permute(2, 0, 1)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class ParquetImageStream(IterableDataset):
    def __init__(
        self,
        data_json,
        filelist_dir,
        side,
        local_batch_size,
        rank=0,
        world_size=1,
        seed=0,
        multiple=16,
        max_ratio=2.0,
        min_area_ratio=0.5625,
        random_crop=True,
        pool_size=64,
        row_groups_per_visit=2,
    ):
        super().__init__()
        self.sources = load_sources(data_json)
        self.files = {}
        for s in self.sources:
            path = filelist_path(filelist_dir, s)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"missing file list {path}; run scripts/build_vae_filelists.py --data-json {data_json} "
                    f"--out-dir {filelist_dir}")
            with open(path) as f:
                self.files[s["dataset_name"]] = [l.strip() for l in f if l.strip()]
        self.side = side
        self.local_batch_size = local_batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.prep_args = (side, multiple, max_ratio, min_area_ratio, random_crop)
        self.pool_size = pool_size
        self.row_groups_per_visit = row_groups_per_visit

    def _worker_files(self, name, gid, total, rng):
        files = list(self.files[name])
        random.Random(self.seed).shuffle(files)
        files = files[gid::total] if len(files) >= total else files
        rng.shuffle(files)
        return files

    def _raw_stream(self, source, files, rng):
        import pyarrow.parquet as pq
        fs = hdfs_fs(source["host"]) if source["host"] is not None else None
        failures = 0
        while True:
            rng.shuffle(files)
            for path in files:
                try:
                    pf = pq.ParquetFile(fs.open_input_file(path) if fs is not None else path)
                    cols = pf.schema_arrow.names
                    col = "image" if "image" in cols else "images"
                    rgs = list(range(pf.metadata.num_row_groups))
                    rng.shuffle(rgs)
                    for rg in rgs[: self.row_groups_per_visit]:
                        values = pf.read_row_group(rg, columns=[col]).column(0).to_pylist()
                        for v in values:
                            if isinstance(v, list):
                                v = v[0] if v else None
                            if v:
                                yield v
                    failures = 0
                except Exception as e:
                    failures += 1
                    if failures <= 3 or failures % 100 == 0:
                        print(f"[vae_data] skip {path}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    if failures > 1000:
                        raise

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        gid, total = self.rank * nw + wid, self.world_size * nw
        rng = random.Random(self.seed * 1000003 + gid)
        prep = _Prep(*self.prep_args)

        names = [s["dataset_name"] for s in self.sources]
        weights = [float(s["weight"]) for s in self.sources]
        streams = {
            s["dataset_name"]: self._raw_stream(s, self._worker_files(s["dataset_name"], gid, total, rng), rng)
            for s in self.sources
        }
        pools = {n: [] for n in names}
        buffers = {}
        while True:
            name = rng.choices(names, weights)[0]
            pool = pools[name]
            while len(pool) < self.pool_size:
                pool.append(next(streams[name]))
            raw = pool.pop(rng.randrange(len(pool)))
            try:
                out = prep(raw, rng)
            except Exception:
                out = None
            if out is None:
                continue
            bucket, img = out
            buf = buffers.setdefault(bucket, [])
            buf.append(img)
            if len(buf) == self.local_batch_size:
                yield torch.stack(buffers.pop(bucket))
