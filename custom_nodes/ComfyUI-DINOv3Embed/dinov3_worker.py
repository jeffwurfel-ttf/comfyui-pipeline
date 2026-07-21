"""
dinov3_worker.py — runs INSIDE ComfyUI-DINOv3Embed/_env (transformers 4.57.x,
torch + scipy inherited from system via --system-site-packages).

Cropping lives HERE (D041/D042): each item is a FULL frame + a per-instance
MASK. The worker computes that mask's tight bbox (largest connected component),
crops image AND mask to it, and hands the crop to AutoImageProcessor for a
SINGLE resize. This is Phase 6's proven path and is per-instance independent by
construction — no shared window, no uniform sizing, no padding.

"Batch" = amortize the model load, NOT a tensor batch: the model loads ONCE,
then N SEQUENTIAL forward passes, each on its own tight crop. A mask batch of 1
is a batch of one, handled identically.

IN : --manifest {"model_dir": "...", "items": [{"image": frame_png, "mask": mask_png|null}, ...]}
OUT: JSON on STDOUT ONLY:
     {"model", "dim", "readouts":["cls","pool"],
      "instances":[{"index","cls":[..],"pool":[..],"masked","cropped",
                    "bbox":[x0,y0,x1,y1]|null,"patches_selected","patches_total"}]}
Vectors RAW (un-normalized); the matcher L2-normalizes. With a mask: pool = mean
of patch tokens whose grid cell is >=50% inside the (cropped) mask; without a
mask: global mean, no crop.
"""
import sys, os, json, argparse


def log(*a):
    print("[dinov3_worker]", *a, file=sys.stderr, flush=True)


def run(manifest_path):
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from scipy import ndimage

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    model_dir = manifest["model_dir"]
    items = manifest["items"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {model_dir} on {device} (ONE load for {len(items)} instance(s))")
    proc = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    patch = int(model.config.patch_size)

    rgb_cache = {}
    def load_rgb(p):
        if p not in rgb_cache:
            rgb_cache[p] = np.array(Image.open(p).convert("RGB"))
        return rgb_cache[p]

    def tight_bbox_cc(mask_bool):
        """Largest connected component's tight bbox (drops specks). Matches the
        Phase 6 reference exactly."""
        lbl, n = ndimage.label(mask_bool)
        if n == 0:
            return None
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        keep = int(np.argmax(sizes) + 1)
        ys, xs = np.where(lbl == keep)
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1), (lbl == keep)

    out_instances = []
    dim = None
    with torch.inference_mode():
        for i, it in enumerate(items):
            rgb = load_rgb(it["image"])                 # full frame
            mask_path = it.get("mask")
            bbox = None
            crop_maskarr = None
            if mask_path:
                m = np.array(Image.open(mask_path).convert("L")) > 127
                res = tight_bbox_cc(m)
                if res is not None:
                    x0, y0, x1, y1, ccmask = res
                    crop_rgb = rgb[y0:y1, x0:x1]
                    crop_maskarr = ccmask[y0:y1, x0:x1]
                    bbox = [x0, y0, x1, y1]
                    cropped = True
                else:                                    # empty mask: fall back to full frame
                    crop_rgb, cropped = rgb, False
            else:
                crop_rgb, cropped = rgb, False

            # SINGLE resize, per-instance, in AutoImageProcessor
            px = proc(images=Image.fromarray(crop_rgb.astype("uint8")),
                      return_tensors="pt")["pixel_values"].to(device)
            grid = px.shape[-1] // patch
            npatch = grid * grid
            res = model(px)                              # sequential forward, own crop
            cls = res.pooler_output[0].float().cpu()
            patches = res.last_hidden_state[0, -npatch:, :].float().cpu()
            dim = int(cls.shape[0])

            if crop_maskarr is not None:
                mg = np.array(Image.fromarray((crop_maskarr * 255).astype("uint8"))
                              .resize((grid, grid), Image.BILINEAR)).astype("float32") / 255.0
                sel = torch.tensor(mg.reshape(-1) >= 0.5)
                if int(sel.sum()) == 0:
                    sel = torch.zeros(npatch, dtype=torch.bool); sel[int(mg.argmax())] = True
                pool = patches[sel].mean(0); masked = True; nsel = int(sel.sum())
            else:
                pool = patches.mean(0); masked = False; nsel = npatch

            out_instances.append({
                "index": i, "cls": cls.tolist(), "pool": pool.tolist(),
                "masked": masked, "cropped": cropped, "bbox": bbox,
                "patches_selected": nsel, "patches_total": npatch,
            })
            log(f"item {i}: cropped={cropped} bbox={bbox} masked={masked} sel={nsel}/{npatch}")

    result = {
        "model": os.path.basename(model_dir.rstrip("/")),
        "dim": dim,
        "readouts": ["cls", "pool"],
        "instances": out_instances,
    }
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    run(args.manifest)


if __name__ == "__main__":
    main()
