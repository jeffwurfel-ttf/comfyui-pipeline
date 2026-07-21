"""
dinov3_worker.py — runs INSIDE ComfyUI-DINOv3Embed/_env (transformers 4.57.x,
torch inherited from system). Loads a DINOv3 ViT and emits two readouts per
input crop: the CLS token (pooler_output) and a mean-pool over patch tokens.

Contract (both directions are files/stdout — no GPU or /models needed to import
this module; model load happens only inside run()):
  IN : --manifest <path>  ->  {"model_dir": "...", "items": [{"image": png, "mask": png|null}, ...]}
  OUT: JSON on STDOUT ONLY (all diagnostics to stderr):
       {"model": name, "dim": D, "readouts": ["cls","pool"],
        "instances": [{"index": i, "cls": [..D..], "pool": [..D..],
                       "masked": bool, "patches_selected": n, "patches_total": g*g}]}

Vectors are RAW (un-normalized); the matcher L2-normalizes. If a mask is given,
pool = mean of patch tokens whose grid cell is >=50% inside the mask; else pool
= global mean over all patch tokens. Patch tokens are the last g*g tokens of
last_hidden_state (DINOv3 embed order is [cls, registers, patches]).
"""
import sys, os, json, argparse


def log(*a):
    print("[dinov3_worker]", *a, file=sys.stderr, flush=True)


def run(manifest_path):
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoModel, AutoImageProcessor

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    model_dir = manifest["model_dir"]
    items = manifest["items"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {model_dir} on {device}")
    proc = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    patch = int(model.config.patch_size)

    out_instances = []
    dim = None
    with torch.inference_mode():
        for i, it in enumerate(items):
            img = Image.open(it["image"]).convert("RGB")
            px = proc(images=img, return_tensors="pt")["pixel_values"].to(device)
            grid = px.shape[-1] // patch          # square token grid side
            npatch = grid * grid
            out = model(px)
            cls = out.pooler_output[0].float().cpu()
            patches = out.last_hidden_state[0, -npatch:, :].float().cpu()   # [g*g, D]
            dim = int(cls.shape[0])

            mask_path = it.get("mask")
            if mask_path:
                m = Image.open(mask_path).convert("L").resize((grid, grid), Image.BILINEAR)
                frac = np.asarray(m, dtype=np.float32).reshape(-1) / 255.0
                sel = torch.tensor(frac >= 0.5)
                if int(sel.sum()) == 0:            # safety, not tuning: keep the single most-covered cell
                    sel = torch.zeros(npatch, dtype=torch.bool)
                    sel[int(frac.argmax())] = True
                pool = patches[sel].mean(0)
                masked, nsel = True, int(sel.sum())
            else:
                pool = patches.mean(0)
                masked, nsel = False, npatch

            out_instances.append({
                "index": i,
                "cls": cls.tolist(),
                "pool": pool.tolist(),
                "masked": masked,
                "patches_selected": nsel,
                "patches_total": npatch,
            })
            log(f"item {i}: dim={dim} masked={masked} sel={nsel}/{npatch}")

    result = {
        "model": os.path.basename(model_dir.rstrip("/")),
        "dim": dim,
        "readouts": ["cls", "pool"],
        "instances": out_instances,
    }
    # STDOUT carries ONLY the JSON payload
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    run(args.manifest)


if __name__ == "__main__":
    main()
