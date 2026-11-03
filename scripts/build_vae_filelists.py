# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""List the parquet shards of every source in a VAE data json, once.

The large sources hold many shards, so each list is capped at --max-files
randomly chosen shards; existing lists are kept unless --force.

    python scripts/build_vae_filelists.py --data-json configs/vae_data/stage1_mix.json \
        --out-dir data/vae_filelists
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vae_data import build_filelist, ensure_hdfs_env, filelist_path, load_sources  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-json", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-files", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    ensure_hdfs_env()
    todo = []
    for dj in args.data_json:
        for s in load_sources(dj):
            out = filelist_path(args.out_dir, s)
            if args.force or not os.path.isfile(out):
                todo.append((s, out))
            else:
                print(f"[filelist] keep {out}")

    def run(item):
        s, out = item
        t = time.time()
        n_all, n_kept = build_filelist(s, out, max_files=args.max_files, seed=args.seed)
        return f"[filelist] {s['dataset_name']}: {n_kept}/{n_all} shards -> {out} ({time.time() - t:.0f}s)"

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(todo)))) as ex:
        for msg in ex.map(run, todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
