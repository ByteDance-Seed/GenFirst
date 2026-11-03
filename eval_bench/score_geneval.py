# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT license in LICENSE.

"""Standalone GenEval scorer (mmdet 3.x API + HuggingFace CLIP).

Upstream geneval/evaluation/evaluate_images.py targets mmdet 2.x (result[0] bbox
lists) + open_clip. This reimplements the SAME scoring logic against the mmdet 3.x
inference API (result.pred_instances) and HF CLIP, so it runs with the source-built
mmcv 2.1.0 + mmdet 3.3.0 on torch 2.5.

Consumes the folder layout produced by Prior_only/gen_for_bench.py --bench geneval:
  <imagedir>/<00000>/metadata.jsonl
  <imagedir>/<00000>/samples/0000.png ...

Outputs results.jsonl and prints the overall + per-tag GenEval scores.
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps
from mmdet.apis import inference_detector, init_detector
from transformers import CLIPModel, CLIPProcessor

TAGS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]


class ColorClassifier:
    def __init__(self, clip_path, device):
        self.device = device
        self.model = CLIPModel.from_pretrained(clip_path).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_path)
        self.cache = {}

    @torch.no_grad()
    def _text_emb(self, classname):
        if classname in self.cache:
            return self.cache[classname]
        templates = [f"a photo of a {{c}} {classname}",
                     f"a photo of a {{c}}-colored {classname}",
                     f"a photo of a {{c}} object"]
        embs = []
        for color in COLORS:
            texts = [t.format(c=color) for t in templates]
            tok = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            f = self.model.get_text_features(**tok)
            f = f / f.norm(p=2, dim=-1, keepdim=True)
            embs.append(f.mean(dim=0))
        clf = torch.stack(embs, dim=1)  # (D, num_colors)
        self.cache[classname] = clf
        return clf

    @torch.no_grad()
    def classify(self, image, objects, classname):
        clf = self._text_emb(classname)
        blank = Image.new("RGB", image.size, color="#999")
        crops = []
        for box, mask in objects:
            if mask is not None:
                comp = Image.composite(image, blank, Image.fromarray(mask))
            else:
                comp = image
            crops.append(comp.crop(box[:4]))
        px = self.processor(images=crops, return_tensors="pt").to(self.device)
        fe = self.model.get_image_features(**px)
        fe = fe / fe.norm(p=2, dim=-1, keepdim=True)
        sim = (fe @ clf).softmax(dim=-1)
        return [COLORS[i.item()] for i in sim.argmax(dim=1)]


def relative_position(obj_a, obj_b, position_threshold=0.1):
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / np.linalg.norm(offset)
    rel = set()
    if dx < -0.5: rel.add("left of")
    if dx > 0.5: rel.add("right of")
    if dy < -0.5: rel.add("above")
    if dy > 0.5: rel.add("below")
    return rel


def evaluate(image, objects, metadata, color_clf, position_threshold=0.1):
    correct = True
    reason = []
    matched_groups = []
    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found = objects.get(classname, [])[:req["count"]]
        if len(found) < req["count"]:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found)}")
        else:
            if "color" in req:
                colors = color_clf.classify(image, found, classname)
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reason.append(f"expected {req['color']} {classname}>={req['count']}")
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} {expected_rel}")
                else:
                    for obj in found:
                        for tobj in matched_groups[target_group]:
                            rels = relative_position(obj, tobj, position_threshold)
                            if expected_rel not in rels:
                                correct = matched = False
                                reason.append(f"expected {classname} {expected_rel} target")
                                break
                        if not matched:
                            break
        matched_groups.append(found if matched else None)
    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(objects.get(classname, [])) >= req["count"]:
            correct = False
            reason.append(f"expected {classname}<{req['count']}")
    return correct, "\n".join(reason)


@torch.no_grad()
def evaluate_image(detector, classnames, color_clf, filepath, metadata, args):
    result = inference_detector(detector, filepath)
    inst = result.pred_instances
    order = torch.argsort(inst.scores, descending=True)
    bboxes = inst.bboxes[order].cpu().numpy()
    scores = inst.scores[order].cpu().numpy()
    labels = inst.labels[order].cpu().numpy()
    masks = inst.masks[order].cpu().numpy() if "masks" in inst else [None] * len(order)

    conf = args.threshold if metadata["tag"] != "counting" else args.counting_threshold
    detected = {}
    for i in range(len(bboxes)):
        if scores[i] < conf:
            continue
        classname = classnames[labels[i]]
        if classname not in detected:
            detected[classname] = []
        if len(detected[classname]) >= args.max_objects:
            continue
        m = masks[i] if masks[i] is not None else None
        detected[classname].append((bboxes[i], m))

    image = ImageOps.exif_transpose(Image.open(filepath)).convert("RGB")
    is_correct, reason = evaluate(image, detected, metadata, color_clf, args.position_threshold)
    return {
        "filename": filepath, "tag": metadata["tag"], "prompt": metadata["prompt"],
        "correct": bool(is_correct), "reason": reason,
    }


def main(args):
    device = torch.device("cuda")
    detector = init_detector(args.model_config, args.model_ckpt, device=device)
    with open(args.object_names) as f:
        classnames = [l.strip() for l in f]
    color_clf = ColorClassifier(args.clip_path, device)

    results = []
    subdirs = sorted(d for d in os.listdir(args.imagedir)
                     if d.isdigit() and os.path.isdir(os.path.join(args.imagedir, d)))
    for si, sub in enumerate(subdirs):
        folder = os.path.join(args.imagedir, sub)
        with open(os.path.join(folder, "metadata.jsonl")) as fp:
            metadata = json.load(fp)
        sdir = os.path.join(folder, "samples")
        for name in sorted(os.listdir(sdir)):
            if not re.match(r"\d+\.png", name):
                continue
            r = evaluate_image(detector, classnames, color_clf,
                               os.path.join(sdir, name), metadata, args)
            results.append(r)
        if si % 50 == 0:
            print(f"  scored {si+1}/{len(subdirs)} prompts", flush=True)

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    with open(args.outfile, "w") as fp:
        for r in results:
            fp.write(json.dumps(r) + "\n")

    # summary
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n================ GenEval ================")
    print(f"images scored: {len(df)}")
    overall = df["correct"].mean()
    for tag in TAGS:
        sub = df[df["tag"] == tag]
        if len(sub):
            print(f"  {tag:15s}: {sub['correct'].mean()*100:.2f}  (n={len(sub)})")
    print(f"  {'OVERALL':15s}: {overall*100:.2f}")
    print("=========================================")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("imagedir")
    p.add_argument("--outfile", default=None)
    p.add_argument("--model-config", required=True)
    p.add_argument("--model-ckpt", required=True)
    p.add_argument("--object-names", required=True)
    p.add_argument("--clip-path", default="openai/clip-vit-large-patch14")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--counting-threshold", type=float, default=0.9)
    p.add_argument("--max-objects", type=int, default=16)
    p.add_argument("--position-threshold", type=float, default=0.1)
    a = p.parse_args()
    if a.outfile is None:
        a.outfile = os.path.join(a.imagedir, "results.jsonl")
    return a


if __name__ == "__main__":
    main(parse_args())
