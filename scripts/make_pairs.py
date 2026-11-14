# Settings and helpers
"""Build query-image / ground-truth pairs from DAVIS 2017 and split them into dev / test.

A pair = one query image (a crop of one object, taken from a RANDOM frame >= MIN_QUERY_FRAME)
         + the ground-truth boxes of that object in the rest of the video.

The search pipeline only needs: query_img + the video's frames.
The gt_boxes are for EVALUATION only (recall, precision, FPR, abstention). The search must never read them.

Usage (from the repo root):
    python make_pairs.py                          # uses DAVIS_ROOT env var, or ./data/DAVIS
    python make_pairs.py --davis-root D:/datasets/DAVIS
"""


import os, json, random
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm

DAVIS = "C:\\Users\\prati\\OneDrive\\Desktop\\Masters\\first_semester\\CMPE-252_AI_and_Data_Engineering\\final_project\\query-by-image-vos-and-tracking\\DAVIS-2017-trainval-480p\\DAVIS"
IMG_DIR  = f"{DAVIS}\\JPEGImages\\480p"
ANN_DIR  = f"{DAVIS}\\Annotations\\480p"
PAIR_DIR = f"{DAVIS}\\outputs\\pairs"

MIN_AREA           = 400   # an object counts as visible in a frame only if its mask has at least this many pixels
MIN_QUERY_FRAME    = 10    # the query frame is picked at random from frames >= this index
EXCLUDE_RADIUS     = 12    # frames within +-12 of the query frame are removed from the search range (near-duplicates)
QUERIES_PER_OBJECT = 1     # random query frames per object (raise it to average out the luck of one random pick)
SEED               = 0

def load_img(video, i): return np.array(Image.open(f"{IMG_DIR}/{video}/{i:05d}.jpg").convert("RGB"))

def load_ann(video, i): return np.array(Image.open(f"{ANN_DIR}/{video}/{i:05d}.png"))    # pixel value = object id

def mask_to_box(m):
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]          # x0, y0, x1, y1 # left bottom and right top

def masked_crop(img, mask, box, margin=0.10, bg=127):
    """Crop the object (plus 10% margin) and paint everything outside its mask gray."""
    H, W = mask.shape
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, y0 = int(max(0, x0 - margin * w)), int(max(0, y0 - margin * h))
    x1, y1 = int(min(W, x1 + margin * w)) + 1, int(min(H, y1 + margin * h)) + 1
    crop = img[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = bg   #sets background color for what is not mask
    return Image.fromarray(crop)



def make_pairs(video, split):
    n = len(os.listdir(f"{IMG_DIR}/{video}"))
    anns = [load_ann(video, i) for i in range(n)]

    obj_ids = sorted({int(o) for a in anns for o in np.unique(a)} - {0, 255})   # 0 = background, 255 = void  #as many oids as there are masks

    pairs, skipped = [], []

    for oid in obj_ids:
        # frames where this object is visible, with its ground-truth box
        boxes = {i: mask_to_box(a == oid) for i, a in enumerate(anns) if (a == oid).sum() >= MIN_AREA}

        eligible = [i for i in sorted(boxes) if i >= MIN_QUERY_FRAME]
        if not eligible:
            skipped.append((video, oid, f"not visible at or after frame {MIN_QUERY_FRAME}"));
            continue

        rng = random.Random(f"{SEED}/{video}/{oid}")  # same pick every run, even if other videos change

        for q in sorted(rng.sample(eligible, min(QUERIES_PER_OBJECT, len(eligible)))):
            lo, hi = q - EXCLUDE_RADIUS, q + EXCLUDE_RADIUS
            gt = {i: b for i, b in boxes.items() if not (lo <= i <= hi)}  # visible frames inside the search range
            if not gt:
                skipped.append((video, oid, f"query {q}: object only visible near the query frame"));
                continue

            qpath = f"{PAIR_DIR}/queries/{video}__{oid}__f{q}.jpg"
            masked_crop(load_img(video, q), anns[q] == oid, boxes[q]).save(qpath, quality=95)
            pairs.append({"pair_id": f"{video}__{oid}__f{q}", "split": split, "video": video, "obj_id": oid,
                          "query_frame": q, "query_img": qpath, "query_box": boxes[q],
                          "n_frames": n, "exclude_frames": [lo, hi],  # NOT searched
                          "gt_boxes": gt})  # evaluation only; frames not listed = object absent
    return pairs, skipped


def main():
    os.makedirs(f"{PAIR_DIR}\\queries", exist_ok=True)

    # Build the pairs and split them into dev / test
    splits = {"dev": open(f"{DAVIS}\\ImageSets\\2017\\train.txt").read().split(),  # 60 videos
              "test": open(f"{DAVIS}\\ImageSets\\2017\\val.txt").read().split()}  # 30 videos

    skipped_all = []
    for split, videos in splits.items():
        pairs = []
        for v in tqdm(videos, desc=split):
            p, s = make_pairs(v, split)
            pairs += p; skipped_all += s
        with open(f"{PAIR_DIR}/pairs_{split}.jsonl", "w") as f:
            for r in pairs: f.write(json.dumps(r) + "\n")
        print(split, ":", len(pairs), "pairs from", len(videos), "videos")
    print("skipped:", skipped_all)
# CHECK: this reads every annotation PNG from Drive, so it can take a few minutes.



if __name__ == "__main__":
    main()