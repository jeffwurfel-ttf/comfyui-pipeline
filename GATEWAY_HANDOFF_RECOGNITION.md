# Gateway Handoff — Recognition Layer (SAM3 + DINOv3)

**Audience:** a gateway engineer registering this capability who has never seen
this project. You should be able to register **Tier 1** from this document alone,
without opening a workflow JSON or asking us anything.

**Provenance:** every number here traces to a run on gpu02 (`comfyui-pipeline`,
image `e6b5eff17961`, ComfyUI core 0.21.0 @ `428c3237`), not to a plan. Phase
citations point at the evidence. Where something is unverified it is labelled.
Anything gateway-side (the capability-manifest schema, `comfyui_output_parser.py`,
`comfyui_cost`, `assert_model_approved`, `required_profile`) lives in the gateway
repo, not here — we describe our side of the contract and name the precedents
(`object3d`, `SAM3D`) you already have.

Two tiers because they have different readiness:

- **Tier 1 — SAM3 recognition:** fully specified, shippable today.
- **Tier 2 — DINOv3 identity:** proven to work; **no production threshold yet.**

---

# TIER 1 — SAM3 Recognition (shippable)

Open-vocabulary detection + segmentation from a text prompt. Runs entirely in
ComfyUI core (`comfy_extras/nodes_sam3.py`, 533 lines) — SAM3 is **not** a custom
node. One checkpoint bundles the vision model + text encoder.

## Capability name & manifest shape

Suggested name: **`recognition_sam3`**. Manifest follows the existing ComfyUI-
workflow capability pattern (cf. the `object3d`/SAM3D entries you already have):

```yaml
recognition_sam3:
  workflow: recognition_sam3_api.json      # API-format graph, exported by us
  required_profile: null                   # see below — do NOT preload a profile
  cost: comfyui_cost                        # time-based: comfyui_cost(generation_time_ms)
  # no assert_model_approved  (matches the object3d precedent)
  output: standard                          # ui.files in the output dir; NOT SAM3D-style flat paths
  inputs:
    image: IMAGE                            # or a video/frame batch for the video path
    prompt: STRING                          # MANDATORY :N grammar — see Input contract
    threshold: FLOAT  (default 0.5)
    individual_masks: BOOL (default true for per-instance)
```

### `required_profile: null` — and why
The workflow loads its own models (SAM3 checkpoint via `CheckpointLoaderSimple`;
DINOv3 via a subprocess venv in Tier 2). It must **not** be paired with a
preloaded model profile. Cautionary precedent: **the wan-video preload OOM** —
preloading a profile alongside a workflow that also loads weights has OOM'd this
24 GB card before. Let the workflow manage its own VRAM. `null` here means "no
gateway preload," not "no models."

## Input contract

- **`prompt` — the `:N` count-cap grammar is MANDATORY, not optional.** A bare
  class name returns **exactly ONE object**. `"person"` against a 3-person frame
  returns 1 detection (the center-most), deterministically. You must write
  `"person:3"`. This is documented SAM 3.1 behavior (the cap is parsed out of the
  free-text prompt, carried inside `conditioning`; it is **not** a discrete node
  input, so `/object_info` will not surface it). **This is the single most likely
  thing a caller gets wrong: it fails silently with a plausible single result.**
  Verified Phase 3 (D025): bare `"person"` → 1 box; `"person:3"` → 3 boxes,
  scores 0.9746 / 0.9707 / 0.9653, deterministic across two runs.
- **Multiple categories**: comma-separate, each with its own cap —
  `"person:3, chair:2, guitar:1"`.
- **Token limit is 32 tokens per CATEGORY SEGMENT, not per prompt** (each
  comma-separated segment is tokenized independently; `sam3_clip.py:51-63`).
  Measured boundary ~30 content tokens/segment before tokenizer chunking; still
  works past it (verified end-to-end at 35 words / 2 chunks, graceful confidence
  dip, correct detection). **No cap on the number of categories** — 8 categories
  / 15 objects ran in one call, zero errors (Phase 5). Realistic 1–4-word labels
  are nowhere near the per-segment limit.

## Output paths

- **Still path:** `SAM3_Detect(individual_masks=true)` → `MASK` **`[N_obj, H, W]`**
  (one plane per instance) plus `BOUNDING_BOX` (a per-frame-nested list of
  `{x, y, width, height, score}` dicts). Verified Phase 3.
- **Video path:** `SAM3_VideoTrack` → track data; `SAM3_TrackToMask` → `MASK`
  **`[N_frames, H, W]`** for a **selected** set of objects. Per-instance
  selection is a **query-time parameter** (`object_indices`, comma-separated
  STRING), **not a tensor axis** — the frame dimension is always the leading dim.
  N separate `SAM3_TrackToMask` calls give N separated per-instance stacks.
  Object identity is index-stable across frames (new tracks appended, never
  reassigned). Verified Phase 4: 3 tracked objects, stable identity frame 1→30.

## THE STRUCTURAL FINDING — read this before designing the contract

**ComfyUI graphs are static; instance counts are runtime-dynamic.** You do not
know N until SAM3 runs, but the graph's node wiring is fixed at author time.
This surfaces as what looks like two problems but is **one collision**:

- **Video** needs `object_indices` to select per-instance masks — but the
  indices to request depend on how many objects SAM3 found at runtime.
- **Multi-category** returns ordered blocks whose boundaries depend on each
  category's *actual* detection count — splitting them needs a "count oracle"
  that also only exists at runtime.

Both are the same thing: a static graph must service a runtime-dynamic instance
count. **This is the hardest thing the gateway contract has to accommodate and
it must not be discovered by you.** Practical accommodations (your call): the
caller supplies expected per-category counts; or the gateway regenerates/
parameterizes the graph per request; or it issues N single-instance calls.
(Note: the Tier 2 DINOv3 embed node absorbs this for the *embedding* step — it
takes the whole `[N,H,W]` mask stack and returns N vectors in one call — but the
SAM3 selection/splitting above is upstream of that and still faces the collision.)

## Category attribution — ordering only, never a field

- Every bbox dict is `{x, y, width, height, score}` — **no category index or
  label** (`nodes_sam3.py:229-234`).
- Attribution survives **by ordering only**: categories are emitted in prompt
  order, each category's top-`:N` detections as a contiguous score-sorted block,
  never globally re-ranked (verified with 2 and 8 categories, Phase 5).
- **Block length = the category's ACTUAL detection count, not its requested
  `:N`.** `"wooden floor:2"` against 1 real region returns a 1-length block, no
  padding, no marker. So **block boundaries are unknowable when any category
  undercounts** — the output cannot self-describe its splits.
- **Detectable case:** if the returned total == the sum of requested `:N`, then
  every block hit its cap and boundaries are exactly the `:N` values. Any
  shortfall means at least one category undercounted and the split is ambiguous.
- **Recommendation: one call per category.** On-prem the per-call cost that drove
  fal's batching no longer exists, and one-category-per-call makes each block's
  boundaries trivially known (block == that call's whole output). fal merged
  multi-subject prompts into one mask; local SAM 3.1 does **not** merge — so
  batching buys nothing here except the attribution ambiguity above.

## Output contract (what the parser receives)

Our nodes emit **standard ComfyUI output-dir files via `ui.files`** — the same
mechanism as `SaveImage`: masks are written by `MaskToImage → SaveImage` (PNG in
the output dir), and `/history` returns `outputs[node]["files"] =
[{"filename": ..., "subfolder": "", "type": "output"}]`. Verified live Phase 7
(the `ui.files → /history` echo was observed, not assumed).

**This does NOT need a SAM3D-style special case.** SAM3D needs one because it
emits *flat absolute file paths* through preview/path nodes
(`SAM3D_PreviewPointCloud(file_path)`, `Preview3D(model_file)` in
`workflows/sam3d_objects_api.json`) — paths to `.glb`/`.ply` outside the standard
output flow. Recognition emits ordinary output-dir files through the standard
`ui.files` echo, identical in shape to every image workflow the parser already
handles. If `comfyui_output_parser.py` handles `SaveImage`, it handles this.
(Answered from what our nodes emit + the SAM3D workflow's node types, not from
reading the gateway parser, which is out of this repo.)

## Cost

`comfyui_cost(generation_time_ms)` — time-based, matching the `object3d`
precedent. **No `assert_model_approved`** (object3d omits it; there is no gated
per-generation model approval in this path — the checkpoint is placed once).

## Measured VRAM & runtime (cite, don't estimate)

- **Checkpoint:** `sam3.1_multiplex_fp16.safetensors`, 1,745,546,848 B (~1.75 GB
  fp16), placed at `/models/checkpoints/` (bind-mounted, survives recreate). **No
  auto-download** — manual placement required; `CheckpointLoaderSimple` is the
  loader (Phase 1/2).
- **Still detect (Phase 3):** loads + executes; SAM3 resident footprint
  `torch_vram_total` ~1.88 GiB on the 24 GB card (Phase 8 idle read).
- **Video track (Phase 4, `golden_3dancers_128f.mp4`, `person:3`,
  `max_objects=4`, `detect_interval=1`):** +0.094 GiB tracker state at 32 frames,
  +0.031 GiB more for the full 128; **128-frame run 50.55 s** wall.
- Card: RTX 4090 24 GB; ~21 GiB free at idle after the checkpoint loads.

---

# TIER 2 — DINOv3 Identity (proven; NO threshold yet)

Per-instance appearance embeddings for identity/gallery matching. Runs as a
baked custom node (`DINOv3Embed`) that shells to an isolated subprocess venv
(transformers 4.57.6) because the main env's transformers 5.8.0 cannot load
DINOv3 under torch 2.4.1. **Not shippable as a thresholded matcher yet — see
below.**

## What is proven

- **Model:** DINOv3 **ViT-L/16** (`facebook/dinov3-vitl16-pretrain-lvd1689m`,
  gated Meta license, granted; permissive for studio use). Embedding **dim
  1024**.
- **Two readouts per instance:** CLS token and masked mean-pool (mean of patch
  tokens inside the instance mask). Both emitted; raw (un-normalized) — the
  matcher L2-normalizes.
- **Per-instance independent BY CONSTRUCTION.** Cropping is inside the node: it
  takes a full frame + SAM3's `[N,H,W]` mask stack, crops each instance to its
  own tight bbox, one single resize each. **An instance's vector does not depend
  on its co-detections.** Verified live through ComfyUI (Phase 8): instance 0
  embedded alone vs in a batch of 3 → **bit-identical** (cosine 1.000000,
  max|Δ| 0.0). Repeats are bit-identical (deterministic float32 forward).
- **Node output:** STRING (JSON) `{model, dim, instances:[{index, cls[1024],
  pool[1024], masked, bbox, patches_selected, patches_total}]}`, saved by
  `SaveVectorJSON` as a standard output-dir `.json` (same `ui.files` contract as
  Tier 1). Index N of the vector list ↔ index N of the mask stack (verified by
  cosine, Phase 8).

## Performance

- **~3.7 s model load + ~0.04 s per additional instance.** One model load per
  node CALL, regardless of N (measured Phase 8: 1 instance 3.73 s / 1 load;
  3 instances 3.81 s / 1 load). Pass the whole mask stack in one call — N
  separate calls pay the ~3.7 s load N times.

## Two memory regimes (state the tradeoff)

- **SAM3 is resident and ComfyUI-managed** — it lives in ComfyUI's model
  manager, so `/free` reclaims it.
- **DINOv3 is ephemeral** — the subprocess loads the model, embeds, and **exits
  per call**; VRAM is reclaimed by the OS on exit (verified Phase 8: 0 lingering
  workers after a run). **No leak, and `/free` does not apply to it.** Tradeoff:
  no residency to leak, but the ~3.7 s load recurs on every call (a persistent
  worker would trade that for held VRAM — a future optimization, not shipped).

## CONTRACT BOUNDARY — identity does NOT survive style transfer

**First-class limit, not a footnote — Coyote is a generative editor.** DINOv3
identity bindings established on source footage will **not** carry to a
generatively restyled take. Measured (Phase 6, D033/D036): the same subject
across a cartoon-render ↔ photoreal-video domain gap scores cosine **0.52–0.60**
— **below** different-subject same-domain similarity. Enrollment must happen in
the **target visual domain**; anything that restyles the pixels (stylization,
heavy grade, generative edit) requires re-enrollment in that new domain.

## NO threshold is stated — explicitly, and why

We do **not** publish a match threshold, on purpose:
- The only margins we measured come from **three similar dancers in matched
  studio lighting** — adversarial relative to the actual scope (props, vehicles,
  CG characters), so they understate real-world separability in the wrong
  direction to trust as a cutoff.
- Gallery matching decides by **argmax over enrollments**, not by an absolute
  cutoff, so a pairwise margin does not measure retrieval accuracy.

**What would settle it (pending operator input gate):** N distinct production
assets, each captured/rendered under **≥2 conditions**, scored as **retrieval
accuracy** (does argmax return the correct enrollment?). Until that exists, ship
Tier 2 as "embeddings available," not "matching guaranteed."

## Dancer numbers — documented WORST CASE for same-category distractors

Labelled as worst case, not as a spec. Within a single visual domain (photoreal
video), ViT-L/16 masked-pool cosine (Phase 6, D035):

| pair | cosine |
|---|---|
| same instance, different frame (A0·A30) | 0.872 |
| different instance, same category (A0·C0) | 0.802 |
| different instance, same category (B0·C0) | 0.792 |
| different instance, same category (A0·B0) | 0.678 |
| **margin (same-instance − best different-instance)** | **+0.071** |

Read: two *different* people in matched lighting reach ~0.80 cosine; the
same-instance-across-frames margin over them is only ~0.07. This is why no
threshold is stated — the distractor case is genuinely hard and the sample is
adversarial.

---

# OPEN QUESTIONS (gateway team owns)

1. How will the static-graph / dynamic-count collision be resolved for the video
   and multi-category paths — caller-supplied counts, per-request graph
   generation, or N single-instance calls? (Tier 1 §Structural finding.)
2. Do you want one-call-per-category enforced at the gateway, or exposed as a
   caller option? (Tier 1 §Attribution.)
3. Who sources the retrieval-accuracy validation set (N assets × ≥2 conditions)
   that gates a published DINOv3 threshold? (Tier 2 §No threshold.)
4. For restyled/generative outputs, will the gateway require re-enrollment in the
   target domain, or refuse cross-domain matches outright? (Tier 2 §Contract
   boundary.)
5. Is the ephemeral per-call ~3.7 s DINOv3 load acceptable at your throughput, or
   is a persistent embedding worker warranted? (Tier 2 §Memory regimes.)
```
