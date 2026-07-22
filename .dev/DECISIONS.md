# DECISIONS — Recognition Layer, container phase (gpu02)

Decision log. Newest first. Format: `## D### — date — one-line`, then the
reasoning and the alternative not taken. Append-only; supersede with a new
entry, never rewrite an old one. Seeded 2026-07-20 (Phase 1). D001–D018 predate
this repo's log (upstream planning framework).

---

## D046 — 2026-07-22 — D045 findings landed in the handoff doc; guard widened; vector check 22/22
Operator added `GATEWAY_HANDOFF_RECOGNITION.md` to guard WRITE_ALLOW permanently
(sha `2a54fb3886cce8eb`) — living tracked deliverable, same category as
ARCHITECTURE.md; its omission was a planner oversight from when the doc lived in
.dev/. Re-ran the vector check (`.dev/scratch/phase6/guard_test.py`) with a new
ALLOW vector for the doc: **22/22 PASS**, both branches live, all seals hold.
Applied both staged D045 changes from
`.dev/scratch/phase9/handoff_D045_additions.md`: the vocabulary bullet (end of
§Input contract) and the §Reading scores + §Absence has no signal sections
(before §Output paths), incl. the enumerate-then-ground architectural constraint.
Checked for the fal-era "bare category nouns only" rule per instruction to
replace-not-append: it never carried into the doc — Tier 1's only "bare"
mentions are the `:N` count-cap behavior (a different, still-correct claim) — so
the new vocabulary bullet stands as the sole vocabulary rule and no superseded
text remains anywhere in the doc. Also removed a dangling ``` fence at EOF
(pre-existing artifact of the .dev/ → root move). Alternative not taken:
appending the correction alongside a restated fal rule — rejected per operator
instruction; superseded claims do not stand. D045's "doc edit BLOCKED" note is
resolved by this entry. gpu02 only; the doc is git-carried so gpu01 gets it on
its eventual sync (B009).

## D045 — 2026-07-21 — SAM3 vocabulary probe: score is noun-fit not correctness; SAM always answers (absence has no signal)
Five findings from `.dev/toolkit/recognize.py` runs today, all for
GATEWAY_HANDOFF_RECOGNITION.md Tier 1:
1. **Score measures noun fit, not correctness.** Verified-correct detections
   ranged 0.63–0.983 — any confidence filter discards good results.
2. **Specificity raises score on identical pixels:** `"ball"` 0.63 →
   `"basketball"` 0.784, same detection (center_x 2026 vs 2024).
3. **Attributive adjectives work:** `"blue hat"` 0.983 (top of set). CONTRADICTS
   the fal-era "bare category nouns only" rule → revised: adjective+noun works;
   relational/prepositional phrases do not (`"dancer in white shirt"` = 0 on fal).
4. **Misspellings tolerate:** `"thermas"` found the thermos (0.678).
5. **SAM ALWAYS ANSWERS (the critical one).** `"elephant"` on an elephant-free
   frame returned a box at 0.694 — HIGHER than two correct detections (thermos
   0.678, ball 0.63). No in-band absence signal; no threshold separates
   hallucination from correct. `:N` is a ceiling not a target (`elephant:3` → 2);
   hallucinated boxes can duplicate (2 boxes 2 px apart, 1552/1554); `:N` is inert
   on real detections (`person:5` → identical 3 at 0.975/0.971/0.965), so the
   count-oracle attribution scheme is safe; score drift appeared only on the
   hallucination (0.694 @ :1 vs 0.667 @ :3).

**Architectural constraint:** SAM 3 MUST NOT be exposed as free-text noun search.
Enumerate-then-ground (a VLM e.g. Gemini enumerates what is present; SAM only
grounds nouns from that enumeration) is the hallucination gate, not a labeling
convenience. Free-text noun entry by an artist has no protection against a
confident box on nothing.

Doc edit BLOCKED: GATEWAY_HANDOFF_RECOGNITION.md is at repo root, not in SETUP's
WRITE_ALLOW (it moved out of .dev/ in bfc1ee5). Ready-to-apply additions staged
at `.dev/scratch/phase9/handoff_D045_additions.md`; a WRITE_ALLOW add for the
doc is requested.

## D044 — 2026-07-21 — Phase 9 COMPLETE: gateway handoff doc written; project container phase closed
`.dev/GATEWAY_HANDOFF_RECOGNITION.md` written as the deliverable — two tiers
(Tier 1 SAM3, shippable; Tier 2 DINOv3, proven/no-threshold), self-contained so
a gateway engineer registers Tier 1 without reading workflow JSON or asking us.
Every number traces to a phase (3/4/5/6/8), unverified items labelled (no
threshold, and why). Structural finding (static graph vs dynamic instance count)
stated once as one collision. Output contract answered from our nodes' `ui.files`
behavior + the SAM3D workflow's path-node types (no SAM3D-style special case
needed), not by reading the gateway parser (out of repo). Doc lives in .dev/
(gitignored) per instruction. Operator's `tools/build.sh` B012 fix committed
(b2a913f) so gpu01's port gets it from git. Guard is SETUP permanently
(0c5d455f88d73c13). No further rebuilds; the container phase is done.

## D043 — 2026-07-21 — Phase 8 acceptance re-confirmed LIVE through ComfyUI; all four bit-exact
Committed 6fa5e89, rebuilt (image e6b5eff17961). Ran the real graph
`SAM3_Detect("person:3",individual_masks) → DINOv3Embed(image,mask) →
SaveVectorJSON` (+ a VHS_SelectMasks("0") branch for batch-1). Through ComfyUI's
tensor handling — where format assumptions surface — all four hold bit-exactly:
CONTINUITY vs Phase 6 ref cos 1.0000/max|Δ|0.0; INDEPENDENCE instance-0 batch-1
vs batch-3 cos 1.000000/max|Δ|0.0; CORRESPONDENCE diagonal 1.0 index N↔N;
PERFORMANCE each DINOv3Embed call = 1 subprocess = 1 load (node A: 1 load/3
instances, node B: 1 load/1), prompt 9.16s. The host-harness result held through
the real path. Phase 8 COMPLETE. This is the last rebuild; guard narrows to
SETUP permanently. Phase 9 is a document (writes confined to .dev/). Evidence:
`.dev/scratch/phase8/crop_inside_results.md`.

## D042 — 2026-07-21 — Cropping moved INSIDE DINOv3Embed (fixes D041); acceptance bit-exact
Replaces the disqualified crop node (D041). DINOv3Embed now takes IMAGE (full
frame) + MASK (SAM3's N-instance stack); the worker computes each mask's tight
bbox (largest CC) and crops image+mask to it, one SINGLE resize in
AutoImageProcessor — Phase 6's path, per-instance independent by construction.
"Batch" = amortize the model load (ONE load, N SEQUENTIAL forwards on their own
tight crops), NOT a tensor batch — no padding, no shared window, no uniform
sizing, so no co-detection dependence. A single frame broadcasts across N masks;
MASK batch of 1 == N=1, no special case.

Acceptance (baked _env, transformers 4.57.6 + scipy, vs Phase 6 crop-outside
reference): (1) CONTINUITY cos 1.0000 / max|Δ| 0.0 both readouts — bit-identical,
so Phase 6's margins hold exactly through the node; (2) INDEPENDENCE instance-0
alone vs in a batch of 3 bit-identical (max|Δ| 0.0) — the property the crop node
could not give; (3) CORRESPONDENCE diagonal 1.0, off-diag 0.77-0.81, index N↔N;
(4) PERFORMANCE 1 load / 3.81s for 3 instances vs 1 load / 3.73s for 1 (load
~3.7s dominates; +0.04s per extra in-call forward) — 3-in-one-call = 1 load vs 3
separate calls = 3 loads, reload-per-call fixed. Graph simplifies to
SAM3_Detect → DINOv3Embed(image,mask) → SaveVectorJSON (no crop/repeat nodes).
Dockerfile unchanged (its COPY block already brings the package; only contents
changed). Evidence: `.dev/scratch/phase8/crop_inside_results.md`. Needs a rebuild.

## D041 — 2026-07-21 — ImageCropByMaskAndResize DISQUALIFIED: crop window is batch-derived, no clean override
Planner's crop-decision check. Q1 (can the window be pinned, not batch-derived?):
NO. The window is `max_w=max([w...]) , max_h=max([h...])` over the batch
(image_nodes.py:4065-4067); every instance is cropped to that shared window.
`max_crop_resolution` is a per-instance ceiling, not a window pin — below it the
window tracks the largest co-detected instance, so an instance's crop (and
embedding) depends on its co-detections, which breaks argmax gallery matching by
construction. The sole batch-independence override, `min_crop_resolution ==
max_crop_resolution == V`, forces a square V×V window and (to satisfy Q2, V≥819)
becomes ≥819² with the instance buried in context — a degenerate work-around,
not a clean pin. Q2 (raise the [128,512] clamp above 819): YES in isolation
(`max_crop_resolution` range 0..16384; `base_resolution` is the separate resize
target), but MOOT given Q1. Per the rule (EITHER not fixable → stop, do not work
around): STOPPED, node disqualified, routed to planner. A batch-independent,
tight, aspect-consistent per-instance crop matching Phase 6 needs a different
node or a purpose-built one (tight bbox + single resize → would need a rebuild).
Evidence: `.dev/scratch/phase8/crop_param_analysis.md`.

## D040 — 2026-07-21 — Phase 8 STOPPED at crop boundary: ImageCropByMaskAndResize materially changes the embedding (continuity divergence)
Planner's ADDED CRITERION caught it. `ImageCropByMaskAndResize` clamps crops to
max 512 (truncates the 819-tall center dancer), forces a uniform max-window
across the batch (adds context to smaller dancers), and double-resizes (lanczos
→512 then AutoImageProcessor bicubic→224). Continuity (workflow-crop vs Phase-6
tight-bbox-crop, same 3 SAM3 masks, diagonal cosine): CLS ~0.87, POOL ~0.95 —
NOT near 1.0. The 0.05–0.13 discrepancy is comparable to or larger than Phase
6's discrimination margin (+0.03…+0.07, D035), so the crop path perturbs the
embedding on the scale of the identity signal. **Phase 6's margin analysis does
NOT describe production through this crop node.** Per the planner's rule, this
is a finding not a failure: reported and STOPPED at the crop boundary, did NOT
choose a crop strategy. Routed to planner.

Bonus results while there (all positive): the full graph runs with no rebuild
(survey held); SAM3 bboxes reproduce Phase 3 bit-identically; instance
correspondence is CLEAN (diagonal dominates, no off-by-one) — ROADMAP
correspondence is satisfiable once a crop path is fixed. VRAM: SAM3 resident
1.88 GiB (managed, /free-able); DINOv3 worker is ephemeral (exits per call, 0
lingering procs, VRAM reclaimed — no leak, reload-per-call cost is a Phase 9
note). Evidence: `.dev/scratch/phase8/phase8_continuity_results.md`.

## D039 — 2026-07-21 — Phase 7 PASS post-rebuild; INITIAL_INSTRUCTIONS assumption 2 (ui.files echo) VERIFIED, no longer INFERRED
On the rebuilt image (`788547d0f0fc`), a live `DINOv3Embed → SaveVectorJSON`
prompt (`1413ea35…`, status success) closed the last unverified audit
assumption: the `SaveBboxesJSON`-shaped `{"ui":{"files":[…]}}` return DOES echo
into `/history` `outputs[node]["files"]` — observed live for both sink nodes,
not inferred. All Phase 7 criteria pass: nodes in `/object_info` (cited from
operator), `.json` recoverable from `/history`, both readouts (cls+pool, dim
1024) present per D037, MASK path exercised with (122/196) and without (196/196),
and floats round-trip to precision — the independent recompute from the baked
`_env` was **bit-identical** (max|Δ|=0.0, deterministic float32 forward).
Evidence: `.dev/scratch/phase7/phase7_results.md`. The SAM3→crop→DINOv3→vectors
joint is proven at the node level; Phase 8 can wire it end to end.

## D038 — 2026-07-21 — Embed-node output type: STRING-carrying-JSON, not a custom EMBEDDING type
Both options were defensible (planner flag 3). Chose STRING (matching
SaveBboxesJSON's `("STRING", {"forceInput": True})` exactly). Reasons: (1)
eliminates the cross-package type-string coupling — a custom "EMBEDDING" type
must be declared identically in BOTH packages, and a mismatch surfaces only
after the rebuild, the most expensive place to find it; (2) maximum precedent
fidelity with the proven Phase 7 template; (3) VectorOut stays genuinely generic
(saves any STRING-JSON, not just embeddings). Cost accepted: less graph-level
type safety — mitigated by a documented JSON schema. Contract:
`DINOv3Embed` emits `("STRING",)` named `embeddings_json`; `SaveVectorJSON`
consumes `("STRING", {"forceInput": True})`. Both agree on STRING — no custom
type string exists to drift. JSON payload: `{model, dim, instances:[{index, cls:
[…], pool:[…], masked:bool, patches_selected, patches_total}]}`, vectors RAW
(un-normalized; matcher L2-normalizes). Node emits BOTH readouts per D037; MASK
optional (present → masked pool over foreground patches; absent → global mean
pool).

## D037 — 2026-07-21 — Planner rulings on the masked-pooling result; Phase 6b+7 combined
1. **The dancer margin is NOT a project gate.** Reasons for the record: gallery
   matching is argmax over enrollments, not thresholded pairwise margin, so the
   margin doesn't measure retrieval accuracy; N=1 (one clip, one frame pair,
   three subjects); Phase 4 already showed SAM3 handles within-shot persistence
   on this footage, so DINOv3's real jobs (cross-shot matching, gallery ID,
   re-acquire after tracker loss) were never exercised; and three similar men in
   matched lighting is adversarial vs the brief's scope (props, vehicles, CG
   characters). The ~0.80 different-instance number becomes a documented
   worst-case same-category distractor, not a gate.
2. **ADOPT masked pooling — for background invariance, not discriminability.**
   Same-instance-across-frames rose sharply (vitl16 0.836→0.872, vitb16
   0.822→0.914); the real use case is one asset across shots with unrelated
   backgrounds, which the dancer test understates.
3. **The node emits BOTH readouts** (CLS and masked-pool); **MASK is an optional
   input.** One extra vector per instance keeps the choice reversible on real
   data.
4. **vitl16 confirmed; variant question closed** (D034/D035 stand).
5. **Phase 9 needs a representative gallery test before any threshold:** N
   distinct production assets, each under ≥2 conditions, scored as RETRIEVAL
   ACCURACY (argmax returns the correct enrollment). Asset sourcing is an
   operator input gate; it does NOT block 6b or 7.

Execution: proceed to **Phase 6b combined with Phase 7** under ONE guard
widening and ONE rebuild. Build both node packages + the venv setup host-side,
verify what's verifiable without the image, THEN bring one rebuild proposal that
tags the running `:latest` (`f92bc745`) as `:pre-phase7` first. `58ccf95`
(8e5a0a7 == pre-hardening) is NOT a valid return path — do not spend it.

## D036 — 2026-07-21 — Recognition Layer contract boundary (Phase 9, promoted from B007): DINOv3 identity does NOT survive style transfer
Stated boundary of the Recognition Layer's contract, not an enrollment-hygiene
note. Phase 6 evidence: the same dancer scores NEGATIVE identity margin across
the cartoon-still ↔ photoreal-video domain gap (D033) — cosine ~0.52–0.60
same-person cross-domain, BELOW different-person same-domain. Therefore identity
bindings established on source footage will NOT carry to a generatively restyled
take. Any pipeline that enrolls on one visual style and matches on another
(restyle, stylization, heavy grade) must re-enroll in the target domain. Carry
into Phase 9 planning as a hard contract limit. B007 promoted to this.

## D035 — 2026-07-21 — Masked-pooling probe: margin widens but the background-confound MECHANISM is falsified; embedding-readout choice routed to planner
Planner hypothesis: the modest CLS margin and high different-instance similarity
(~0.82) are a background confound; masked pooling (mean of foreground patch
tokens) should drop different-instance similarity and widen the margin. Result
(raw bbox crops, background kept; one method, no tuning): margin DID widen
(vitb16 +0.008→+0.052; vitl16 +0.044→+0.071) BUT different-instance similarity
ROSE, not fell (vitb16 diff-max 0.814→0.862; vitl16 0.793→0.802). The widening
came from same-instance-across-frames rising most (background is cross-frame
noise), not from separating different people — who still sit at ~0.80–0.86 on
person-only patches. So background is NOT what makes unrelated subjects look
alike; the confound theory as a mechanism is not supported. Per the planner's
own criteria this is the ambiguous middle (margin moved, but the diagnostic
quantity did not behave as predicted), so the production-readout decision
(CLS vs masked-pool + MASK node input) and the sufficiency call are the
planner's, not self-decided. Variant vitl16 reaffirmed under the new numbers
(D034 holds; masked margin +0.071 vs +0.052 and lower different-instance
confusion). Method used: `pooler_output` vs mean of the last grid² post-norm
patch tokens selected by ≥0.5 mask coverage on the 14×14 grid.

## D034 — 2026-07-21 — DINOv3 variant selected: ViT-L/16, on measured within-video discrimination margin
Both variants downloaded and tested. Corrected within-video gate margins
(same-instance-across-frames minus best different-instance-same-category):
vitl16 +0.063 (squish 224²) / +0.067 (pad-square); vitb16 +0.027 / +0.054.
vitl16 wins and is more preprocessing-robust; vitb16's +0.027 is not "clearly
above" (A0·C0 0.820 vs A0·A30 0.847 — too tight). License identical (D031) and
VRAM (1.16 GiB vs 0.33 GiB) is trivially co-resident with SAM3 on 24 GB, so the
choice is on discrimination quality, not size, exactly as the brief directed.
Consequence flagged: enrolled reference vectors are variant-specific — switching
later invalidates the whole gallery, so this is expensive to reverse. dim=1024.
CAVEAT: margins are modest; final go/no-go on DINOv3-for-identity is a planner
call (see D033 and the results file).

## D033 — 2026-07-21 — Data-integrity finding: the golden still is a stylization, not a frame of the video → specified gate was cross-domain
`golden_3dancers_1920x1080.png` (Phase 3 still; source of A0/B0/C0) is a
cel-shaded CARTOON render. `golden_3dancers_128f.mp4` (source of A30) is
PHOTOREALISTIC footage of the same dancers/poses. Verified by eye and by
still-vs-video-frame0 MAD 19.29. The planner's STEP 5 assumed both share a
visual domain; they do not. So the specified test (still A0/B0/C0 vs video A30)
measures a cartoon→photo domain gap, producing a NEGATIVE margin that is NOT a
DINOv3 identity failure. Corrected by rebuilding all four crops from the
photoreal video (Phase 4 obj0/1/2 masks) — margin then flips POSITIVE for both
variants and both preprocessing modes. This finding also corrects Phase 4's
attribution of the same MAD gap to "H.264 re-encoding" (BACKLOG). Routes to
planner: it bears on which visual domain identity enrollment actually operates
in (photoreal footage, per this evidence — the stylized still is a SAM3
detection asset, not an identity-enrollment asset).

## D032 — 2026-07-21 — DINOv3 main-env load BLOCKED (torch 2.4.1 × transformers 5.8.0); rescope to subprocess venv, proven pair 4.57.6/2.4.1
The DINOv3 loader API itself is fine — `AutoModel`→`DINOv3ViTModel` is
registered in transformers 5.8.0 (`modeling_auto.py:130`), same call shape as
4.x. The break is lower: main-env `from_pretrained` fails importing
`transformers.modeling_utils` → `integrations.moe:250`, which registers a
`torch.library.custom_op` whose stringized annotations the main env's torch
2.4.1 `infer_schema` cannot resolve. Nominal `torch>=2.4` is met but the moe
integration needs newer torch in practice. This is the documented Phase 6
FAILURE MODE → subprocess venv (SeedVR2 precedent, NOT SAM3D). Proven on the
box: SeedVR2's `_env` (transformers 4.57.6 + torch 2.4.1) loads DINOv3 and runs
the full gate — 4.57.6 ≥ DINOv3's documented 4.56 and predates the 5.x moe
break. Rescope, not a retry: a dedicated pinned `_env` baked via a Dockerfile
block + rebuild is required, foldable into the single Phase 6+7 rebuild the
planner flagged. Did NOT pip install into the main env. Weights already staged
under /models, so the venv needs only the runtime.

## D031 — 2026-07-21 — DINOv3 HF access GRANTED; license read closed as permissive for studio use
Meta granted access to user `jeff-wurfel-ttf` for the gated
`facebook/dinov3-*-pretrain-lvd1689m` repos, clearing the D028 block. Phase 6
resumes at STEP 2 (variant selection). License read (relayed by planner, closed):
permissive for commercial studio use; attribution is owed ONLY on third-party
distribution of the model itself — not on internal use, embeddings, or derived
vectors. No blocker to baking weights into the on-box pipeline or shipping the
gallery of enrolled vectors. Supersedes D028's block (D028 remains the record of
why the gate existed).

## D030 — 2026-07-21 — Phase 7 prerequisite settled: custom_nodes/ is BAKED, not bind-mounted → phase ends in a rebuild OPERATOR GATE
The audit left this INFERRED / doc-conflicted. Now settled from three agreeing
sources (running box is authoritative): (1) `docker-compose.rocky.yaml` volumes
list only `/models`, `./workflows:ro`, output, input — no custom_nodes volume;
(2) `Dockerfile` populates `/app/ComfyUI/custom_nodes/*` at build time via `git
clone` + `COPY` — the SaveBboxesJSON precedent package itself is baked at
`Dockerfile:157-159`; (3) live `docker inspect` mounts + in-container `mount`
show custom_nodes is NOT a mount, it is served from the image layer.

Consequence: a node created host-side in `custom_nodes/` is invisible to the
running container until `docker build` + `--force-recreate`. There is no
incremental bind-mount path. Phase 7 therefore ends at a rebuild OPERATOR GATE,
exactly as the ROADMAP Phase 7 prerequisite anticipated.

Rebuild-safety accounting (verified on the box): running image is
`comfyui-pipeline:latest` = `f92bc745` (63.4 GB, current production). The ONE
rollback is `comfyui-pipeline:8e5a0a7` = `comfyui-pipeline:pre-hardening` =
`58ccf95` (91 GB, older, separate ID). A `docker build -t comfyui-pipeline:latest`
reassigns `:latest` to the NEW image and orphans `f92bc745`. So the rebuild
proposal MUST first tag the current running latest (e.g. `:pre-phase7`) to
preserve a return path to today's working state; the 58ccf95 rollback tags are
untouched by a rebuild but must not be relied on as the phase-7 rollback (they
predate every post-hardening change). "Do not spend it" holds.

## D029 — 2026-07-21 — Phase 7 dependency corrected: it does NOT depend on Phase 6 (planner, relayed)
Recorded from the planner (its primary log is upstream; captured here so this
repo's DECISIONS stays a coherent ground truth). The ROADMAP listed Phase 7 as
depending on Phase 6; that was a planner error. The vector-out node's contract
is float-in / JSON-out and is indifferent to the source of the floats. With
Phase 6 BLOCKED on a human HF license grant (D028), an unvalidated DINOv3
vector is the wrong test payload anyway. SAM3 bbox/score floats are proven,
reproducible, and have established ground truth (Phase 3 golden set), so they
are the better Phase 7 payload. Phase 7 proceeds now on SAM3 data.

## D028 — 2026-07-21 — Phase 6 gated at STEP 1: DINOv3 HF access denied, routes to planner
STEP 1 (ACCESS) is a hard gate that can end the phase, and it did. The token in
the container (`whoami` → `jeff-wurfel-ttf`) authenticates and reads repo
metadata, but an authenticated HEAD on the weight file of every
`facebook/dinov3-*-pretrain-lvd1689m` variant returns `GatedRepoError 403`
("not in the authorized list"; `gated=manual`). Confirmed it is not
token-scope (metadata + whoami succeed), not transient (deterministic across
four variants), not a wrong repo id (all metadata resolves) — the wall is the
Meta license grant.

Followed the brief literally: wrote BLOCKED.md and stopped rather than reaching
for a workaround. The brief forecloses workarounds explicitly — few-shot VLM
identity was evaluated and rejected before the project began, so there is no
fallback embedding path to substitute. Did not proceed to STEP 2–5; probing the
transformers-5.x loader API (STEP 3, the other documented Phase 6 failure route
→ subprocess venv) is downstream of an access grant and would be wasted work
now.

Alternative not taken: request a guard mode flip to BUILD to start staging
weights under `/models`. Rejected — there are no weights to stage without
access, the whole gate was reachable inside SETUP, and the phase brief said not
to request a mode change absent an actual block. The block here is upstream of
any write.

## D027 — 2026-07-20 — Phase 5: attribution survives only by ordering, not as a field; token limit is per-category not per-prompt
Read `nodes_sam3.py:208-234` and `comfy/text_encoders/sam3_clip.py` before
running anything, per the phase's explicit instruction. Two corrections to
the operator's framing, both source-grounded then runtime-confirmed:
1. **No output field carries category attribution** (`frame_bbox_dicts`
   entries are `{x,y,width,height,score}` only). Ordering is category-blocked
   (`cond_list` loop, `nodes_sam3.py:208`, per-category top-N sort at `:223`),
   never globally re-ranked — confirmed with 2 and 8 simultaneous categories,
   every block landing exactly where source predicts. But block length is
   the category's ACTUAL detection count, not its `:N` cap — a
   `"wooden floor:2"` request against 1 real floor region returned a
   1-length block, no padding. Attribution is mechanically recoverable but
   only with an external count oracle; the output cannot self-describe its
   own block boundaries.
2. **The 32-token limit is per comma-separated category segment, not per
   prompt** (`SAM3TokenizerWrapper.tokenize_with_weights`,
   `sam3_clip.py:51-63`, tokenizes each segment independently). Direct
   tokenizer introspection (no model load) found the real boundary at ~30
   content tokens/segment, past which the standard chunking mechanism adds a
   2nd chunk rather than erroring or truncating — confirmed working end to
   end at runtime (35-word single-category phrase, 2 chunks, correct
   detection, no error). **No cap found on the number of categories per
   prompt** — an 8-category, 15-object prompt ran in one call with no error.
   The operator's "4-5 categories" concern does not hold; corrected for
   Phase 9.
Per "DO NOT tune prompts to produce a nicer answer": the control (`"person:3"`
alone) was run unchanged first and reproduced Phase 3 bit-for-bit before any
multi-category test, confirming no drift before adding variables. The
collapse-and-distribute logic from the brief stays — not because SAM 3.1
merges categories the way fal did (it doesn't, confirmed), but because the
Recognition Layer must independently track per-category actual counts to
reconstruct attribution from an unlabeled, positionally-ordered output.

## D026 — 2026-07-20 — Phase 4 methodology: per-object TrackToMask calls, not a single flat tensor
`SAM3_TrackToMask` has no `individual_masks` toggle (unlike `SAM3_Detect`) —
`object_indices` selects/unions specific tracked objects per call, and the
output is always `[N_frames, H, W]` (frame-indexed, never instance-stacked).
Read from source (`nodes_sam3.py:473-509`) before building anything, per the
running pattern of dumping contracts first. To get 3 separately identified,
frame-indexed mask stacks, called the node 3 times (`object_indices="0"`,
`"1"`, `"2"`) rather than looking for a single-call flat-stack equivalent —
none exists, and none is needed, since the frame dimension survives natively.
Also verified no spurious 4th track was created (`max_objects=4` allowed one
spare slot): re-submitted the identical graph with an added
`object_indices="3"` query; nodes 1-4 cache-hit (confirmed via
`execution_cached`, same `track_data`), and the query returned all-zero
across all 128 frames. Bonus frame-correspondence task (frame.png's source
frame index) done via `ffmpeg`-extracted frames + per-frame mean-absolute-
pixel-diff against the golden still, not via any new ComfyUI node — outside
the guarded surfaces, cheap, time-boxed to the requested ~10 minutes.

## D025 — 2026-07-20 — Phase 3 unblocked: planner spec error, not a model/wiring fault
Root cause of the BLOCKED.md finding: SAM3's prompt contract supports a
`:N` count cap per class (e.g. `"eye:2, window panels:4"`, per ComfyUI SAM 3.1
docs). A bare `"person"` carries no count and defaults to 1 — the single
0.9746 detection in the D024 run was **correct behavior**, not a detector
regression. The Phase 3 ROADMAP spec (bare `"person"` expected to yield 3
instances) was wrong; the error is the planner's, not the executor's or the
model's. Corrected prompt for the re-run: `"person:3"`. Before re-running,
the full `/object_info/SAM3_Detect` contract gets dumped and reported — the
audit only ever surfaced `MODEL`/`individual_masks`, and that incomplete read
of the contract is what let the wrong spec stand unquestioned. Gate re-
registered (HARD: 3 instances, `[3,H,W]`, separable centers matching the 3
visible dancers, stable ordering across 2 runs; SOFT: scores >= 0.90, fal's
0.92-0.93 band superseded — local 0.9746 on a single instance suggests SAM 3.1
scores higher on this frame than SAM 3 did). Still forbidden: adjusting
resolution/threshold, or trying counts other than 3 to find a pass. Noted for
later phases: SAM3 prompts have a 32-token limit.

## D024 — 2026-07-20 — Phase 3 stopped after 2 attempts; routed to planner, not self-resolved
Both attempts (identical params, per the "do not adjust to pass" rule)
returned 1 detected instance against a golden frame with 3 clearly separated,
visually-confirmed dancers. Deterministic — bit-identical bbox/score both
runs. This is a HARD-criteria fail (wrong instance count) per D023, not a
near-miss and not attributable to a graph-wiring bug found on inspection
(conditioning wired per `SAM3_Detect`'s documented contract; `node_errors: {}`
both runs; mask confirmed single connected component, not a merge). Did not
attempt a third run with a different prompt/threshold — explicitly forbidden.
Wrote `.dev/BLOCKED.md` with full measured evidence and stopped; did not
proceed to Phase 4/5. Alternative not taken: guessing that a richer prompt or
point-seeded conditioning would fix it and re-running — that would be
adjusting inputs to force a pass, the exact thing the gate forbids.

## D023 — 2026-07-20 — Phase 3 gate split HARD/SOFT for SAM 3 vs 3.1 version delta (operator amendment)
ROADMAP Phase 3's original acceptance ("every score in 0.92-0.97") assumed the
same SAM version as the fal ground truth. The local checkpoint is SAM 3.1; the
golden set (prompt "person", sam-3/image, 2026-07-16) came from SAM 3. A
version delta in scores is expected and is not itself a failure. Operator
split the gate:
- HARD (any failure is a FAIL): exactly 3 instances; mask tensor `[3, H, W]`;
  3 boxes with centers cleanly separable left to right; ordering stable
  across two runs at the same seed.
- SOFT (log, don't fail): scores within 0.92-0.93 (fal's 0.93/0.93/0.92 band).
  Outside that band with all HARD criteria passing is recorded as a SAM 3 vs
  3.1 delta with actual values quoted, not treated as a regression.
- FAIL regardless of scores: wrong instance count, merged masks, or
  inseparable centers.
Do not adjust prompt, threshold, or resolution to force a pass. Two attempts,
then `BLOCKED.md`. Supersedes the unqualified score-band acceptance in
ROADMAP.md's original Phase 3 section without rewriting it (append-only log).

## D022 — 2026-07-20 — Phase 2 smoke test wired no conditioning into SAM3_Detect
The ROADMAP deliverable names `CheckpointLoaderSimple -> SAM3_Detect ->
PreviewImage` with no `CLIPTextEncode`/conditioning node. Kept literal: Phase 2
proves load + execution only, not detection accuracy (that is Phase 3's job,
with the golden frame and `"person"` prompt). Result: the union mask output is
correctly all-zero (zero detections, nothing to detect against) and the
measured VRAM delta (~1.05 GiB) is smaller than the checkpoint's on-disk size
(~1.63 GiB), plausibly because the unwired text-encoder path never loaded to
GPU. Filed as BACKLOG B006 rather than assumed or chased down inline — out of
Phase 2's scope, and Phase 6/8 already track VRAM regimes for the joint
SAM3+DINOv3 workflow.

## D021 — 2026-07-20 — Executor refused a known guard loophole rather than using it
A bash redirect would have bypassed the Write-tool path check. The executor
identified it, declined it, and escalated. Recorded because it is evidence the
prose-plus-hook layering works where the hook alone cannot, and because the
alternative would have been an unreviewable silent bypass.

## D020 — 2026-07-20 — Phase 1 escalated to Opus
ARCHITECTURE.md is the cold-start ground-truth artifact. A degraded version
fails silently and poisons every later session. The hooks and .gitignore are
mechanical; the document is not.

## D019 — 2026-07-20 — Guard mode taxonomy corrected; Phase 1 is setup, not validation
Planner error caught by the executor. AUDIT-through-Phase-5 conflated validation
writes (correctly .dev/-only) with all writes (Phase 1 builds the operating
layer and lands outside it by definition). Mode renamed SETUP, WRITE_ALLOW
widened to .claude/, .gitignore, ARCHITECTURE.md. WRITE_DENY added so the guard
cannot widen itself in any mode — widening .claude/ without it would have made
the control self-modifiable. Partial bash-redirect guard added for the five
protected surfaces; general shell parsing remains out of reach by design.
