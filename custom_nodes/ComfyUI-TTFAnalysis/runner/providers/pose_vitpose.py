"""
2D pose — ViTPose. SPECIFICATION ONLY. NOT IMPLEMENTED, NOT REGISTERED.

This file exists so the shape of a main-env provider is visible while the
blocker is still open. It deliberately contains no implementation and calls
register() nowhere, so importing it is a no-op and registry.load_all() adds
nothing.

WHY IT IS BLOCKED
-----------------
The deployed pose path runs `yolov10m.onnx` as its person detector, which is
AGPL-3.0 (verified from THU-MIG/yolov10/LICENSE; parent ultralytics likewise).
AGPL §13 triggers on network interaction and a private gateway is network
interaction, so this cannot ship in client deliverables as wired. It is
selected by five deployed workflows — see BACKLOG B018.

Second, unrelated blocker: the deployed ViTPose ONNX itself comes from
`JunkyByte/easy_ViTPose`, whose HF card declares NO licence at all. Upstream
ViTPose code is Apache-2.0 (Open-MMLab), but those redistributed weights carry
no terms — UNRESOLVED, not clean. `Kijai/vitpose_comfy` declares Apache-2.0 and
is the cheap fix. See BACKLOG B021.

The YOLOX port that would unblock the first has not landed, and it is not a
file swap: `WanAnimatePreprocess/models/onnx_models.py` `Yolo.postprocess`
decodes only Ultralytics-family head layouts, so YOLOX needs a new branch.

THE SHAPE, WHEN IT LANDS
------------------------
    register(Provider(
        name="pose2d",
        env="main",              # <- the reason dispatch has a 'main' path.
                                 #    ViTPose is already wired in the main
                                 #    ComfyUI env and has no dependency
                                 #    conflict; it must NOT be duplicated into
                                 #    _env just to satisfy this runner.
        depends_on=(),
        schema=Schema((
            Dataset("pose_kp",    ("T", "P", 133, 3), "float16"),  # x, y, conf
            Dataset("pose_bbox",  ("T", "P", 4),      "float16"),
            Dataset("pose_count", ("T",),             "uint8"),
        )),
        cost=Cost(vram_mb=..., window=1, gpu=True),
        display=Display(
            dataset="pose_kp",
            readout_kind="keypoints",
            range_fn=None,                  # keypoints are already in pixels
            proxies=(ProxySpec("pose_skeleton", _draw_skeleton),),
        ),
        license=LicenseRow(
            model="ViTPose-L wholebody + <detector>",
            code_license="Apache-2.0 (Open-MMLab)",
            weight_license=...,   # MUST be resolved; JunkyByte declares none
            gated=False,
            verdict=...,          # BLOCKED until the detector is replaced
            source_url=...,
            date_checked=...,
        ),
        run=_run,
    ))

Note the LicenseRow above cannot currently be completed, and registration
asserts on an incomplete one. That is the intended behaviour: this provider
should be unable to register until its licence question is answered.

`P` is a ragged axis (persons per frame). Whoever implements this needs to pick
a fixed cap and record it, or move to a variable-length dataset — the current
Schema has no notion of ragged and would need extending. Flagging it here so it
is a design decision rather than a surprise.
"""
