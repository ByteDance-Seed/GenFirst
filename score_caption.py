# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Score generated COCO captions: CIDEr / BLEU-4 / METEOR / ROUGE-L / SPICE + CLIPScore.

References are the human COCO annotations (captions_train2017.json +
captions_val2017.json cover both the val2017 and the Karpathy test image sets).
PTBTokenizer and METEOR/SPICE need a JVM on PATH.

CLIPScore follows Hessel et al.: ``2.5 * max(cos(image, text), 0)``, which is
reference-free and therefore insensitive to the caption-length mismatch between
this model's training captions and COCO's human ones.

Usage:
    python score_caption.py --pred captions.jsonl --split val2017 \
        --coco-root /data/coco
    python score_caption.py --pred captions.jsonl --split karpathy-test \
        --coco-root /data/coco --karpathy-json /data/coco_karpathy/dataset_coco.json
"""

import argparse
import json
import os
import sys

# PTBTokenizer and SPICE shell out to bundled jars and write scratch files next
# to them, which fails for a root-owned site-packages install, so point
# PYCOCOEVALCAP_DIR at a writable copy of the package when that happens.
EVAL_PKGS = os.environ.get("PYCOCOEVALCAP_DIR", "")
if EVAL_PKGS and os.path.isdir(os.path.join(EVAL_PKGS, "pycocoevalcap")):
    sys.path.insert(0, EVAL_PKGS)


def load_references(coco_root):
    """``cocoid -> [human captions]`` across both 2017 annotation files.

    The Karpathy test images come from both, so both files are needed to score
    that split even though val2017 alone only needs one.
    """
    refs = {}
    for name in ("captions_train2017.json", "captions_val2017.json"):
        path = os.path.join(coco_root, "annotations", name)
        if not os.path.exists(path):
            continue
        ann = json.load(open(path))
        for a in ann["annotations"]:
            refs.setdefault(a["image_id"], []).append(a["caption"].strip())
    return refs


def split_ids(split, coco_root, karpathy_json):
    if split == "val2017":
        ann = json.load(open(os.path.join(coco_root, "annotations/captions_val2017.json")))
        return {im["id"] for im in ann["images"]}
    if not karpathy_json:
        raise SystemExit("--split karpathy-test needs --karpathy-json, i.e. the "
                         "dataset_coco.json of Karpathy's splits")
    kar = json.load(open(karpathy_json))
    return {im["cocoid"] for im in kar["images"] if im["split"] == "test"}


def coco_metrics(gts_raw, res_raw, skip_spice=False):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    tok = PTBTokenizer()
    gts = tok.tokenize({k: [{"caption": c} for c in v] for k, v in gts_raw.items()})
    res = tok.tokenize({k: [{"caption": v}] for k, v in res_raw.items()})

    out = {}
    scorers = [(Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
               (Meteor(), "METEOR"), (Rouge(), "ROUGE-L"), (Cider(), "CIDEr")]
    if not skip_spice:
        from pycocoevalcap.spice.spice import Spice
        scorers.append((Spice(), "SPICE"))
    for scorer, name in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
        except Exception as e:  # SPICE/METEOR need a JVM and can die on odd input
            print(f"  !! {name} failed: {type(e).__name__}: {e}")
            continue
        if isinstance(name, list):
            out.update({n: float(s) * 100 for n, s in zip(name, score)})
        else:
            out[name] = float(score) * 100
    return out


def clip_score(res_raw, image_index, clip_path, batch_size=64):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(clip_path).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_path)

    keys = [k for k in sorted(res_raw) if k in image_index]
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(keys), batch_size):
            chunk = keys[i : i + batch_size]
            imgs = [Image.open(image_index[k]).convert("RGB") for k in chunk]
            txts = [res_raw[k] or " " for k in chunk]
            inp = proc(text=txts, images=imgs, return_tensors="pt",
                       padding=True, truncation=True, max_length=77).to(device)
            ie = model.get_image_features(pixel_values=inp["pixel_values"])
            te = model.get_text_features(input_ids=inp["input_ids"],
                                         attention_mask=inp["attention_mask"])
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            cos = (ie * te).sum(-1).clamp(min=0)
            total += float((2.5 * cos).sum())
            n += len(chunk)
    # Reported x100 to sit on the same scale as the n-gram metrics; divide by
    # 100 to recover the 2.5*cos value papers quote in [0, 1].
    return 100 * total / max(n, 1)


def main(args):
    preds = {}
    with open(args.pred) as f:
        for line in f:
            r = json.loads(line)
            preds[int(r["cocoid"])] = r["caption"].strip()

    want = split_ids(args.split, args.coco_root, args.karpathy_json)
    have = want & set(preds)
    if not have:
        raise SystemExit(f"no predictions overlap split {args.split}")
    print(f"split={args.split}: {len(have)}/{len(want)} images have a prediction")

    refs = load_references(args.coco_root)
    missing_refs = [k for k in have if k not in refs]
    if missing_refs:
        print(f"  !! {len(missing_refs)} images have no human reference, dropping them")
        have -= set(missing_refs)

    res_raw = {k: preds[k] for k in have}
    gts_raw = {k: refs[k] for k in have}
    lens = sorted(len(v.split()) for v in res_raw.values())
    ref_lens = sorted(len(c.split()) for v in gts_raw.values() for c in v)
    print(f"  generated caption words: median {lens[len(lens)//2]}, "
          f"p90 {lens[int(.9*len(lens))]}, max {lens[-1]}")
    print(f"  human reference words:   median {ref_lens[len(ref_lens)//2]}")

    out = coco_metrics(gts_raw, res_raw, skip_spice=args.skip_spice)
    if not args.skip_clipscore:
        from eval_caption import build_image_index
        out["CLIPScore"] = clip_score(res_raw, build_image_index(args.coco_root),
                                      args.clip_path)

    print(f"\n============ captioning ({args.split}) ============")
    for k in ["CIDEr", "BLEU-4", "METEOR", "ROUGE-L", "SPICE", "CLIPScore",
              "BLEU-1", "BLEU-2", "BLEU-3"]:
        if k in out:
            print(f"  {k:<10}: {out[k]:.2f}")
    print("=" * 50)
    if args.out:
        json.dump({"split": args.split, "n_images": len(have), "pred": args.pred,
                   "metrics": out}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=str, required=True, help="JSONL from eval_caption.py")
    p.add_argument("--split", type=str, default="val2017",
                   choices=["val2017", "karpathy-test"])
    p.add_argument("--coco-root", type=str, required=True,
                   help="COCO root holding annotations/, train2017/ and val2017/")
    p.add_argument("--karpathy-json", type=str, default="",
                   help="dataset_coco.json of Karpathy's splits, for that split")
    p.add_argument("--clip-path", type=str,
                   default=os.environ.get("CLIP_PATH", "openai/clip-vit-large-patch14"),
                   help="CLIP for CLIPScore; a local directory or a hub id")
    p.add_argument("--skip-spice", action="store_true", help="SPICE is the slow one")
    p.add_argument("--skip-clipscore", action="store_true")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
