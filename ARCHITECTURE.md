# ARCHITECTURE — comfyui-pipeline (gpu02)

Subsystem map for the GPU compute node. This is the cold-start orientation
document: what the box IS, verified against the box, not against memory.

**Authority order:** the running box > `.dev/AUDIT_gpu02.md` (ground-truth
audit, VERIFIED probes) > this file > `.dev/ROADMAP.md`. On any conflict, the
box wins — re-probe and correct this file, naming the drift.

**Provenance:** every section below carries a *Verified against* anchor — the
command or file that established it and the date. All anchors here are from the
2026-07-20 audit (`.dev/AUDIT_gpu02.md`); re-verify before relying on a value
that could have moved (VRAM, queue, image tags, package versions).

**Gateway handoff:** the Recognition Layer's cross-team deliverable —
capability/manifest, contracts, limits, measured numbers for a gateway engineer
— is `GATEWAY_HANDOFF_RECOGNITION.md` at the **repo root** (tracked, next to this
file). It is self-contained; register Tier 1 (SAM3) from it without reading the
workflow JSON.

---

## Status table

| Subsystem | State (2026-07-20) | Load-bearing gotcha |
|---|---|---|
| Host | Rocky 9.7, Threadripper 3970X (VM), RTX 4090 24GB, avx2 present | 16 vCPU of a 32-core part; runs as a guest VM |
| Storage | Docker + containerd on `/mnt/ssd`; `/home` 154G. 108G orphan at `/home/containerd-data` reclaimed 2026-07-21 (B011) | One image is 63–91GB; `/home` cannot hold it. Reclamation `du` needs sudo — non-sudo reads restricted dirs as empty |
| Container | `comfyui-pipeline` up since 2026-07-16, healthy; idle | `Created` (05-11) ≠ start time; use `StartedAt` |
| Images | `:latest` 788547d0 (63.6GB, Phase 6b/7) live; rollbacks pre-phase7=f92bc745, pre-hardening=58ccf95 | `build.sh` prune list omits `pre-phase7` — would delete it (B012) |
| ComfyUI | core 0.21.0 @ 428c3237 on 8188; queue/history empty | Production box — check `/queue` before running |
| Wrapper | 8189 `/health` green; `model_loaded:false` | `/health` schema is HEALTHCHECK-coupled (2 places) |
| Deps (main) | Py3.11, torch 2.4.1+cu121, transformers 5.8.0 | Never `pip install` into main env |
| Subprocess venvs | sam3dobjects/_env, SeedVR2/_env — real isolation | The precedent for any new conflicting dep |
| SAM3 (core) | 4 nodes live; checkpoint loads, golden set + video track + multi-prompt characterized (Phases 2-5) | `:N` cap needed; token limit is per-category not per-prompt; no category-attribution field — ordering-only, count-dependent |
| DINOv3 (Phase 6b/7) | BAKED: `DINOv3Embed` (subprocess `_env`, transformers 4.57.6, vitl16) + `SaveVectorJSON` live in `/object_info`; Phase 7 acceptance PASS | main env can't load DINOv3 (torch 2.4.1 × transformers 5.8.0 moe, D032) — node shells to `_env`. Emits CLS+pool (D037); STRING-JSON out (D038). Golden still is a stylization (D033/B007) |
| Custom nodes | 21 dirs; SeedVR2 + DINOv3Embed carry pinned `_env`s | CameraPack checked out twice (collision risk); only SeedVR2/DINOv3Embed self-isolate |
| Tooling | 8 scripts in `tools/`; all pass syntax | `tools/` is operator-owned; don't edit unasked |
| Control (guard hook) | `.claude/hooks/guard.py` PreToolUse, MODE=BUILD (6b/7); 20/20 vectors pass post-B010 fix | Crash = fail-open; partial corruption reports green — test BOTH branches (§Control layer) |
| Known drift | README/CLAUDE.md misstatements; 2 referenced docs absent | See §Known drift; box is authoritative |

---

## Host

- **OS/kernel:** Rocky Linux 9.7 (Blue Onyx), kernel `5.14.0-611.34.1.el9_7.x86_64`.
- **CPU:** AMD Ryzen Threadripper 3970X 32-Core; **16 vCPU** exposed, 1 thread/core.
  The `hypervisor` CPU flag is set — this is a **guest VM**, not bare metal.
- **avx2: PRESENT** (also avx, fma, f16c, bmi1/2, sha_ni; no AVX-512).
- **RAM:** 46 GiB + 56 GiB swap (swap unused at audit).
- **GPU:** NVIDIA RTX 4090, 24 GB. Driver `590.48.01`, CUDA `13.1`. A desktop
  session (Xorg + gnome-shell) is resident on the GPU — a GUI on a compute node.
- **Disks:** `/` 70G (24%), `/home` 154G (**74%, 41G free**), `/mnt/ssd` 1007G
  (47%, **515G free**).
- **Docker/containerd:** Docker Root `/mnt/ssd/docker`; containerd root
  `/mnt/ssd/containerd`; **nvidia runtime registered** (default runtime `runc`,
  nvidia opt-in per container). `daemon.json` carries the nvidia runtime — merge,
  never overwrite.

*Verified against:* `cat /etc/os-release`, `uname -r`, `lscpu`,
`grep avx2 /proc/cpuinfo`, `free -h`, `nvidia-smi`, `df -h`, `docker info`,
`/etc/docker/daemon.json`, `/etc/containerd/config.toml` — 2026-07-20
(`.dev/AUDIT_gpu02.md` §Part 1).

## Container

- **`comfyui-pipeline`**: running, **StartedAt `2026-07-16T21:16:29Z`**, health
  green. The `Created` field (2026-05-11) is the container-object date, not the
  process start — use `StartedAt`.
- **Live image:** `comfyui-pipeline:latest` = `788547d0f0fc` (63.6 GB, built
  2026-07-21, Phase 6b/7 — DINOv3Embed + VectorOut + DINOv3 `_env`). The +0.2 GB
  vs the prior 63.4 GB confirms the DINOv3 venv used `--system-site-packages` (a
  duplicate torch wheel would have been ~+6 GB).
- **Rollback, now two depths:** `pre-phase7` = `f92bc745493a` (63.4 GB, the
  pre-DINOv3 image) and `pre-hardening` = `8e5a0a7` = `58ccf95cf8bd` (91 GB,
  2026-05-08). Do not consume either without re-tagging `:latest` first.
  **`tools/build.sh`'s prune allowlist protects only `latest`/`pre-hardening`/
  `pre-rollback` — NOT `pre-phase7`**, so a `build.sh` run would delete the
  current return path. → BACKLOG B012.
- **Storage reclamation (2026-07-21):** the May Docker/containerd migration to
  `/mnt/ssd` left **108 GB orphaned at `/home/containerd-data`** — on the 154 GB
  `/home` it existed to relieve — invisible for 5 months because non-sudo `du`
  reports permission-restricted dirs as empty. Removed 2026-07-21. Run
  reclamation `du`/`df` with sudo and verify the OLD root is empty as an explicit
  migration step. → BACKLOG B011; gpu01 owes the same (B009).
- **ComfyUI** on 8188: core **0.21.0** @ git `428c3237` (2026-05-11). Queue and
  history **empty** at audit. This is a production box — check `/queue` first.
- **Wrapper** on 8189: `/health` green — `comfyui_responsive:true`,
  `disk_ok:true` (514 GB free on output path), `model_loaded:false`. The
  `/health` schema is duplicated in two HEALTHCHECK blocks (Dockerfile +
  docker-compose.rocky.yaml) — change all three together or none.

- **Service name ≠ container name:** `docker-compose.rocky.yaml` defines
  service key **`comfyui`** with `container_name: comfyui-pipeline`.
  `docker compose -f docker-compose.rocky.yaml restart <name>` takes the
  **service** name — the correct form is `restart comfyui`, NOT `restart
  comfyui-pipeline`. A doc referenced by prior planning (COMFY.md §10; absent
  from this repo, see §Known drift) documents the wrong form. → BACKLOG B005.

*Verified against:* `docker inspect`, `docker images`, `curl :8188/queue`,
`curl :8188/history`, `curl :8189/health`, `comfyui_version.py`,
`docker-compose.rocky.yaml:19,24` — 2026-07-20 (`.dev/AUDIT_gpu02.md` §Part 2).

## VRAM regimes

- **Idle (audit):** 441 MiB used / ~24 GB free. Composition: Xorg 14 MiB +
  gnome-shell 12 MiB + container python 386 MiB. `model_loaded:false`.
- **SAM3.1 fp16 checkpoint, measured, RESOLVED (Phase 2 + Phase 3 retest):**
  Phase 2 (no `conditioning` wired): `torch_vram_total` **1.0 GiB**. Phase 3
  (live `CLIPTextEncode("person")` wired into `conditioning`, same never-freed
  process): `torch_vram_total` grew to **1.844 GiB** (+0.844 GiB). Total delta
  from Phase 2's cold-idle baseline (24,791,089,152 B free) to Phase 3's
  post-run state (22,810,338,948 B free) is ~1.845 GiB, matching
  `torch_vram_total` almost exactly — internally consistent. **Confirmed:**
  the checkpoint's bundled text encoder (§SAM3 core surface) is not loaded to
  GPU until a real text prompt is wired through `conditioning`; the
  vision/detector backbone alone accounts for Phase 2's ~1.05 GiB. BACKLOG B006
  closed.
- **Two regimes will coexist at Phase 8** and must be tracked separately:
  1. **SAM3 via ComfyUI's model manager** — `/free` and the manager's eviction
     apply to it.
  2. **DINOv3 in a subprocess venv** (main env cannot load it — see Dependency
     baseline / D032), invoked out-of-process. ComfyUI's model manager does NOT
     see it; `/free` does not reclaim it. A leak here is a handoff finding, not
     an inline fix. Measured load footprint (SeedVR2 `_env`, 2026-07-21): vitb16
     ~327 MiB, vitl16 ~1156 MiB torch-allocated — trivially co-resident with
     SAM3's ~1.97 GiB on the 24 GB card.
- **Cautionary precedent:** the wan-video OOM. Measured ceilings and per-call
  runtime are **pending** and get filled from Phases 3, 4, and 8.

*Verified against:* `nvidia-smi`, `curl :8189/health` (idle numbers) — 2026-07-20.
Load-delta figures are PENDING (unmeasured until Phase 2+).

## Custom nodes

19 node dirs under `/app/ComfyUI/custom_nodes/`. Git-backed state:

- **Pinned (detached):** `ComfyUI-SeedVR2_VideoUpscaler` only — HEAD
  `4490bd1` (detached), upstream `numz/…`. **It is the sole git-pinned node.**
- **Tracking `main` (live upstream):** Frame-Interpolation, KJNodes, Manager,
  VideoHelperSuite, WanAnimatePreprocess, WanVideoWrapper, segment-anything-2,
  ProPainter, HyMotion, geometrypack, motioncapture, multiband, and CameraPack
  (both copies — see drift).
- **Vendored (no `.git`):** `ComfyUI-MultiPersonDetector`,
  `ComfyUI-SaveEXRCompressed`, `ComfyUI-VRAMPurge`, `comfyui-sam3dobjects`.
  These are source drops, **not** pinned checkouts.
- **Structured-data precedent:** `SaveBboxesJSON` in
  `ComfyUI-MultiPersonDetector/__init__.py:126` — `RETURN_TYPES=()`,
  `OUTPUT_NODE=True`, writes a JSON `STRING` to `output/` and returns
  `ui.files`. This is the shape the Phase 7 vector-out node clones.

*Verified against:* per-dir `git rev-parse` / `symbolic-ref` / `remote get-url`
inside the container, and `grep SaveBboxesJSON` — 2026-07-20
(`.dev/AUDIT_gpu02.md` §Part 4).

## SAM3 core surface

**SAM3 detection/segmentation ships in ComfyUI CORE, not as a custom node.**
Source: **`/app/ComfyUI/comfy_extras/nodes_sam3.py`** (533 lines). All four node
classes appear in `/object_info`:

- **`SAM3_Detect`** — `comfy_extras/nodes_sam3.py:88` (still/image path;
  per-instance masks via `individual_masks`).
- **`SAM3_VideoTrack`** — `nodes_sam3.py:260` (memory-based tracker).
- **`SAM3_TrackPreview`** — `nodes_sam3.py:315` (video preview, no tensor out).
- **`SAM3_TrackToMask`** — `nodes_sam3.py:473` (select tracked objects → MASK).

Key parameter — **`individual_masks`** (BOOLEAN, default False,
`nodes_sam3.py:107`, *"Output per-object masks instead of union"*):
- `False` → `torch.stack` → **`[B, H, W]`** (one union plane per input image).
- `True`  → `torch.cat`   → **`[Σ N_obj, H, W]`** (one plane per instance).
- Final assembly at `nodes_sam3.py:254`. **Landmine:** with `True` and `B>1`,
  frame boundaries are erased on the mask output (flat instance stack);
  per-frame grouping survives only on the bboxes output (`:238`). Do not do
  video by batching stills through `SAM3_Detect`.

**Checkpoint loading:** the nodes take a `MODEL` input and reach the net via
`model.model.diffusion_model` (`nodes_sam3.py:99,155`). SAM3 is recognized by
state-dict key sniffing (`comfy/model_detection.py:771`; SAM3 vs SAM3.1 classes
in `comfy/supported_models.py:1835/1882`). One checkpoint bundles the vision
model + text encoder, so a standard `CheckpointLoaderSimple` from
`models/checkpoints/` is the loader. **No auto-download** — manual placement
required. Backing modules present: `comfy/ldm/sam3/{detector,sam,tracker}.py`.

**Checkpoint staged (Phase 1 gate):** `sam3.1_multiplex_fp16.safetensors`
(1,745,546,848 B) at `/mnt/ssd/comfyui-models/checkpoints/`, confirmed loading
and executing (Phase 2) — see §VRAM regimes.

**Load-bearing gotcha, RESOLVED in Phase 3 (was open, see D024/D025):** SAM3's
prompt string supports a **`:N` count cap per class** (`"person:3"`, ComfyUI
SAM 3.1 docs; e.g. `"eye:2, window panels:4"` for multiple classes). A bare
class name with no `:N` **defaults to count 1** — this is documented model
behavior, not a detector fault. First measured: a bare `"person"` prompt
against a 3-person golden frame returned exactly 1 detection (the center-most
person), deterministically across two runs. Corrected to `"person:3"`:
returned 3 detections, deterministically across two runs, centers 613.1 /
928.0 / 1201.0 (cleanly separated, matching the 3 visible dancers), scores
0.9746 / 0.9707 / 0.9653. **The `:N` cap is not a discrete `SAM3_Detect` node
input** — it is parsed out of the free-text prompt string carried inside
`conditioning`, so `/object_info` alone will not surface it.

**Multi-category prompting, verified Phase 5 — attribution survives by
ordering only, not as an output field.** Source:
`comfy_extras/nodes_sam3.py:208-234` (per-category detection loop inside
`SAM3_Detect.execute`) and `comfy/text_encoders/sam3_clip.py` (tokenizer/
encoder wrapper building `sam3_multi_cond`).

- Every emitted bbox dict is `{x, y, width, height, score}` only
  (`nodes_sam3.py:229-234`) — **no category index or label field exists.**
- Ordering is **category-blocked**: `for ... in cond_list` (`:208`) iterates
  once per comma-separated category in prompt order; each category's own
  top-`:N` detections (score-sorted, `:223`) are appended as a contiguous
  block before moving to the next category. Never globally re-ranked across
  categories — confirmed with 2 and 8 simultaneous categories, every block
  landing exactly where predicted and matching the frame's actual content.
- **Block length is the category's ACTUAL detection count, not its `:N`
  cap** — `"wooden floor:2"` against 1 real floor region returned a
  1-length block, no padding, no marker. Attribution is only recoverable if
  the caller independently knows each category's real count; the output
  cannot self-describe its own block boundaries. This is the substantive gap
  from a fully labeled output — a **finding**, not a bug.
- **The 32-token limit is per CATEGORY SEGMENT, not per prompt**
  (`SAM3TokenizerWrapper.tokenize_with_weights`, `sam3_clip.py:51-63`,
  tokenizes each comma-separated segment independently). Measured boundary:
  ~30 real content tokens/segment before the standard tokenizer chunking adds
  a 2nd chunk (not an error — confirmed working end-to-end at 35 words/2
  chunks, graceful confidence dip, correct detection). **No cap found on the
  number of categories per prompt** — 8 categories/15 objects ran in one call
  with zero errors. Realistic 1-4 word entity labels are nowhere near the
  per-segment boundary; a 13-entity scene is not token-bounded.
- fal merged multi-subject prompts into one mask (why fal used one call per
  concept); **local SAM 3.1 does not merge** — confirmed distinct per-object
  detections across 8 simultaneous categories. The collapse-and-distribute
  logic in the project brief still stays, repurposed: not to prevent
  merging, but because the Recognition Layer must track each category's
  actual count itself to reconstruct attribution from an unlabeled,
  positionally-ordered output.

Full runtime evidence: `.dev/scratch/wf/sam3_phase5_multiprompt_results.md`.

**Video tracking, verified Phase 4 — landmine does NOT reproduce in
`SAM3_TrackToMask`:** `SAM3_VideoTrack` returns a `SAM3_TRACK_DATA` dict
(`packed_masks: [N_frames, N_obj, Hm, Wm//8]` uint8 bit-packed, `n_frames`,
`orig_size`, `scores` — a **per-object** list from `forward_video`, source
`comfy/ldm/sam3/tracker.py`). Object identity is index-stable: new tracks are
*appended* to the object axis (`_match_and_add_detections`), existing indices
are never reassigned. `SAM3_TrackToMask` has **no `individual_masks`
toggle** — `object_indices` (comma-separated STRING, default = all) selects/
unions object channels, and the `MASK` output is **unconditionally
`[N_frames, H, W]`**, regardless of selection. Per-frame grouping is the
tensor's native axis; per-instance grouping is a **query-time parameter**,
not a tensor axis — N separate calls are needed for N separated per-instance
stacks. This is architecturally different from `SAM3_Detect`'s
`individual_masks=True` landmine (which flattens instances into the leading
dim and erases frame boundaries for `B>1`) — `SAM3_TrackToMask` resolves it
by construction, not a workaround. The `:N` count-cap grammar (see above)
applies identically to `SAM3_VideoTrack`'s `conditioning` input —
`_extract_text_prompts` is the same helper both nodes call.

Measured (Phase 4, `golden_3dancers_128f.mp4`, prompt `"person:3"`,
`max_objects=4`, `detect_interval=1`): 3 tracked objects, stable identity
frame 1→30 (no index swap; confirmed no spurious 4th object via a cache-hit
`object_indices="3"` re-query returning all-zero across 128 frames).
`torch_vram_total`: +0.094 GiB for a 32-frame tracker run, +0.031 GiB more
for the full 128 frames (modest — per-frame tracker state is small relative
to model weights). 128-frame full run: 50.55s wall time.

*Verified against:* `curl :8188/object_info`, reads of `nodes_sam3.py`,
`model_detection.py`, `supported_models.py`, `find models -iname '*sam3*'` —
2026-07-20 (`.dev/AUDIT_gpu02.md` §Part 4, §Part 6, §Addendum). Phase 2/3/4
findings verified against `.dev/scratch/wf/sam3_load_smoke_results.md`,
`.dev/BLOCKED.md`, `.dev/scratch/wf/sam3_video_results.md`, and
`comfy/ldm/sam3/tracker.py` (`pack_masks`/`unpack_masks`/`forward_video`) —
2026-07-20.

## Dependency baseline

- **Main env:** Python 3.11.0rc1; torch 2.4.1+cu121, torchvision 0.19.1+cu121;
  numpy 1.26.4; opencv-python-headless 4.9.0.80; **transformers 5.8.0**;
  diffusers 0.38.0. No xformers in main. **Never `pip install` here.**
- **Subprocess venvs (the isolation precedent):**
  - `comfyui-sam3dobjects/_env` — Python 3.10.12, torch 2.4.1+**cu124**,
    transformers 5.8.0 (separate interpreter AND CUDA minor).
  - `ComfyUI-SeedVR2_VideoUpscaler/_env` — Python 3.11, torch 2.4.1+cu121,
    transformers **4.57.6**, diffusers 0.34.0, own opencv.
- **DINOv3 (Phase 6, PROVEN 2026-07-21 — D032):** the DINOv3 loader API is fine
  (`AutoModel`→`DINOv3ViTModel`, registered in 5.8.0 `modeling_auto.py:130`) —
  but the **main env CANNOT load it**: transformers 5.8.0 `modeling_utils` →
  `integrations.moe:250` registers a `torch.library.custom_op` that the main
  env's **torch 2.4.1** `infer_schema` rejects (stringized annotations). Nominal
  `torch>=2.4` is met but insufficient. This is the boundary break the note
  warned of — it landed on the whole `modeling_utils` import, not the DINOv3
  class. **The fix is transformers 4.57.x, NOT a torch bump** — both
  `sam3dobjects/_env` (5.8.0, cu124) and the main env (5.8.0, cu121) break;
  only **SeedVR2's `_env` (4.57.6)** loads DINOv3 (4.57.6 ≥ 4.56 and predates
  the 5.x moe change). Production path: a dedicated pinned `_env` baked via a
  Dockerfile block + rebuild (SeedVR2 precedent). Rescope routed to planner;
  see `.dev/BLOCKED.md`, D032/D033/D034. Weights staged under
  `/models/dinov3/{vitb16,vitl16}` (vitl16 selected, D034).

*Verified against:* `pip freeze` in main env and both `_env/` venvs —
2026-07-20 (`.dev/AUDIT_gpu02.md` §Part 3). Full freezes in `.dev/audit-data/`.

## Tooling

Eight scripts under `tools/` (operator toolkit — do not edit unasked). All pass
`bash -n` / `py_compile` (2026-07-20):

| Script | Purpose |
|---|---|
| `build.sh` | Build the image with a git-sha tag |
| `comfy_doctor.sh` | Workflow readiness diagnostic (`--fix` repairs symlinks) |
| `comfy_models.sh` | Unified model manager (status/download) |
| `comfy_onboard.sh` | Workflow onboarding |
| `comfy_preflight.sh` | Workflow preflight validator |
| `rollback.sh` | Point `:latest` at a tag + restart |
| `test_all_workflows.py` | Workflow test harness |
| `test_lifecycle.py` | Container lifecycle test harness |

No duplicate scripts. `build.sh` (tag/build) and `rollback.sh` (retag+restart)
are complementary, not overlapping. `assess_gpu02.sh` does **not** exist anywhere
on the box.

*Verified against:* `ls tools/`, per-script header + `bash -n`/`py_compile`,
`find / -name assess_gpu02.sh` — 2026-07-20 (`.dev/AUDIT_gpu02.md` §Part 5).

## Control layer — the guard hook

`.claude/hooks/guard.py` is a `PreToolUse` hook (runs under Python 3.9 on this
box). Two branches: **Write/Edit** (substring `WRITE_ALLOW` gated by `MODE`,
plus `WRITE_DENY` = guard.py/settings.json operator-only in every mode, and a
hard `..`-traversal block) and **Bash** (regex blocks: force/protected-branch
push, bulk `git add`, `rm -rf`, redirects into protected surfaces, docker
lifecycle, `pip install`, upstream installers, /etc/docker writes, VRAM
purge/interrupt, GPU reset). MODE widens for container-facing phases (6b/7) and
narrows back to SETUP after the rebuild. Verify any change to it with the
20-vector harness `.dev/scratch/phase6/guard_test.py` (both branches).

**Two load-bearing lessons from the B010 incident (2026-07-21):**

1. **Partial control corruption is worse than total failure.** A two-character
   corruption (`\s`→`\q` at lines 33/35) killed the *entire* Bash branch while
   `WRITE_ALLOW` still decoded correctly and passed 13/13 write vectors. A
   control that reports green on the half you happen to test is more dangerous
   than one that fails outright — it invites false confidence. **Test every
   branch of any guard, every time, not just the path you changed.**

2. **A guard crash FAILS OPEN.** A hook that raises exits code 1, which Claude
   Code treats as a *non-blocking* error — so the tool proceeds. Fail-open is
   deliberate for unreadable input (json.load is try/wrapped → exit 0), but it
   means **any runtime error anywhere in the hook silently disables every rule
   below it.** A bad regex on line 33 nullified lines 34-59. Keep hook bodies
   defensively simple; a syntax/regex fault is not a safe failure here.

## Known drift (box is authoritative)

- **`README.md:5`** describes the node as `10.10.210.80 (TTF-LAX-GPU01)`. The
  box is **`VFXNRocky-02` / `10.10.210.81` = gpu02**. README was written for the
  other node and never re-homed. → BACKLOG.
- **`CLAUDE.md:26`** calls SAM3D a *"pinned commit SHA"* precedent, but
  `comfyui-sam3dobjects` has **no `.git`** — vendored, isolated-by-venv, not
  pinned. Only SeedVR2 is git-pinned. → BACKLOG.
- **`COMFY.md` and `HARDENING_ROADMAP.md`** are referenced by prior planning but
  **do not exist in this repo** (only `README.md` + `CLAUDE.md` do). Prior
  `pip`-guard comments still cite `COMFY.md` as the anti-pattern source.
- **CameraPack duplicate checkout:** `ComfyUI-CameraPack` and
  `comfyui-camerapack` — same HEAD `60729ce`, same upstream. Possible node
  class-mapping collision (unproven). → BACKLOG.

*Verified against:* `README.md`, `CLAUDE.md`, `ls *.md`, `git`/`find` probes —
2026-07-20 (`.dev/AUDIT_gpu02.md` §Part 5, §Part 7).

---

*Cross-node note:* changes captured here landed on **gpu02 only**. gpu01
(10.10.210.80) is on a pre-hardening rollback and is NOT in sync; nothing here
has been propagated to it.
