# Step Reload and check a few pairs by eye
import numpy as np, json
from PIL import Image
import matplotlib.pyplot as plt, matplotlib.patches as patches


DAVIS = "C:\\Users\\prati\\OneDrive\\Desktop\\Masters\\first_semester\\CMPE-252_AI_and_Data_Engineering\\final_project\\query-by-image-vos-and-tracking\\DAVIS-2017-trainval-480p\\DAVIS"
PAIR_DIR = f"{DAVIS}\\outputs\\pairs"
IMG_DIR  = f"{DAVIS}\\JPEGImages\\480p"

def load_img(video, i): return np.array(Image.open(f"{IMG_DIR}/{video}/{i:05d}.jpg").convert("RGB"))

def load_pairs(split):
    out = [json.loads(l) for l in open(f"{PAIR_DIR}/pairs_{split}.jsonl")]
    for r in out: r["gt_boxes"] = {int(k): v for k, v in r["gt_boxes"].items()}   # JSON turns int keys into strings
    return out


def main():
    pairs_dev, pairs_test = load_pairs("dev"), load_pairs("test")
    print("dev:", len(pairs_dev), "| test:", len(pairs_test))

    assert not {p["video"] for p in pairs_dev} & {p["video"] for p in pairs_test}, "a video is in both splits!"

    for p in pairs_test[::max(1, len(pairs_test) // 3)][:3]:
        f = min(p["gt_boxes"]);
        x0, y0, x1, y1 = p["gt_boxes"][f]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2.2]})
        ax[0].imshow(Image.open(p["query_img"]));
        ax[0].set_title("query image");
        ax[0].axis("off")
        ax[1].imshow(load_img(p["video"], f));
        ax[1].axis("off")
        ax[1].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=3))
        ax[1].set_title(f"{p['pair_id']} | target frame {f}")
        plt.show()

# CHECK: does the query crop show the same object as the red box in the later frame?



if __name__ == "__main__":
    main()