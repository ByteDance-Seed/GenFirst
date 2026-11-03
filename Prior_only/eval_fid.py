# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""FID / Inception Score for a folder of generated PNGs.

Scores against precomputed reference statistics rather than a reference image set,
which is the protocol MAR uses (https://github.com/LTH14/mar). That is what the
fork adds:

    pip install git+https://github.com/LTH14/torch-fidelity.git@master

since `fid_statistics_file` does not exist in upstream torch-fidelity. If only
upstream is installed, this falls back to driving its own primitives directly --
the same Inception extractor (`inception-v3-compat`, 2048-d pool features) and the
same Frechet distance, with the reference mu/sigma read from the .npz instead of
computed from a second image folder. The two paths agree; the fallback exists
because the fork needs GitHub access, which not every environment has.

The statistics that ship in fid_stats/ are the ones the paper's numbers were
computed against:

    adm_in256_stats.npz   ImageNet 256x256
    jit_in512_stats.npz   ImageNet 512x512

Usage:
    python eval_fid.py --sample-dir samples/<run-folder>
    python eval_fid.py --sample-dir samples/<run-folder> --resolution 512
"""
import argparse
import os

DEFAULT_STATS = {
    256: "adm_in256_stats.npz",
    512: "jit_in512_stats.npz",
}


def score_against_statistics(sample_dir, stats_file, isc=True, verbose=False):
    """FID (and IS) from precomputed reference statistics, on upstream torch-fidelity.

    Reproduces what the fork's `fid_statistics_file` does: extract features once
    with torch-fidelity's own extractor, turn them into (mu, sigma), and compare
    against the reference statistics. IS reuses the same forward pass.
    """
    import numpy as np
    from torch_fidelity.defaults import DEFAULTS
    from torch_fidelity.metric_fid import (
        fid_featuresdict_to_statistics, fid_statistics_to_metric,
    )
    from torch_fidelity.metric_isc import isc_featuresdict_to_metric
    from torch_fidelity.utils import (
        create_feature_extractor, extract_featuresdict_from_input_id,
        resolve_feature_extractor, resolve_feature_layer_for_metric,
    )

    kwargs = dict(DEFAULTS)
    kwargs.update(
        input1=sample_dir, input2=None, cuda=True, verbose=verbose,
        fid=True, isc=isc, kid=False, prc=False,
    )

    extractor_name = resolve_feature_extractor(**kwargs)
    layer_fid = resolve_feature_layer_for_metric("fid", **kwargs)
    layer_isc = resolve_feature_layer_for_metric("isc", **kwargs) if isc else None
    layers = [layer_fid] + ([layer_isc] if isc else [])
    extractor = create_feature_extractor(extractor_name, layers, **kwargs)
    featuresdict = extract_featuresdict_from_input_id(1, extractor, **kwargs)

    ref = np.load(stats_file)
    generated = fid_featuresdict_to_statistics(featuresdict, layer_fid)
    # fid_statistics_to_metric asserts matching dtypes, and the shipped .npz stores
    # mu as float32 against numpy's float64 covariance.
    reference = {k: np.asarray(ref[k]).astype(generated[k].dtype) for k in ("mu", "sigma")}

    metrics = fid_statistics_to_metric(generated, reference, verbose)
    if isc:
        metrics.update(isc_featuresdict_to_metric(featuresdict, layer_isc, **kwargs))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", type=str, required=True,
                   help="Folder of generated .png files (what generate.py writes).")
    p.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    p.add_argument("--fid-stats", type=str, default=None,
                   help="Override the reference statistics .npz.")
    p.add_argument("--fid-stats-dir", type=str, default=None,
                   help="Directory holding the shipped statistics (default: ../fid_stats).")
    p.add_argument("--no-isc", action="store_true", help="Skip Inception Score.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Imported here so --help works without the fork installed.
    import torch_fidelity

    if args.fid_stats is not None:
        stats = args.fid_stats
    else:
        stats_dir = args.fid_stats_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, "fid_stats"
            )
        stats = os.path.join(stats_dir, DEFAULT_STATS[args.resolution])
    if not os.path.isfile(stats):
        raise FileNotFoundError(f"reference statistics not found: {stats}")
    if not os.path.isdir(args.sample_dir):
        raise NotADirectoryError(f"sample folder not found: {args.sample_dir}")

    n_png = len([f for f in os.listdir(args.sample_dir) if f.endswith(".png")])
    print(f"==> scoring {n_png} images in {args.sample_dir}")
    print(f"==> against {stats}")
    if n_png != 50_000:
        # Not fatal: useful for quick checks. But published numbers are 50k, and
        # FID is biased upward on smaller samples, so say so loudly.
        print(f"!!! {n_png} images, not 50000 -- not comparable with the paper")

    from torch_fidelity.defaults import DEFAULTS

    # The LTH14 fork's FID-only shortcut ignores ``fid_statistics_file`` and
    # tries to prepare ``input2=None``. Keep its combined FID+IS path for the
    # normal paper evaluation, but use the direct-statistics implementation
    # when --no-isc requests FID alone.
    if "fid_statistics_file" in DEFAULTS and not args.no_isc:
        metrics = torch_fidelity.calculate_metrics(
            input1=args.sample_dir,
            input2=None,
            fid_statistics_file=stats,
            cuda=True,
            isc=not args.no_isc,
            fid=True,
            kid=False,
            prc=False,
            verbose=args.verbose,
        )
    else:
        print("==> scoring against the .npz directly")
        metrics = score_against_statistics(
            args.sample_dir, stats, isc=not args.no_isc, verbose=args.verbose)

    fid = metrics["frechet_inception_distance"]
    print(f"\nFID  {fid:.4f}")
    if not args.no_isc:
        print(f"IS   {metrics['inception_score_mean']:.4f} "
              f"+/- {metrics['inception_score_std']:.4f}")


if __name__ == "__main__":
    main()
