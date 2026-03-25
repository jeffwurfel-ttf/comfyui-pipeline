#!/usr/bin/env python3
"""
ComfyUI Workflow Test Runner

Runs all registered workflows with test inputs and reports results.
Run directly on the ComfyUI server.

Usage:
    python3 test_all_workflows.py
    python3 test_all_workflows.py --skip wan_t2v,wan_i2v   # skip slow ones
    python3 test_all_workflows.py --only esrgan,rife        # test specific ones
"""

import json
import os
import sys
import time
import argparse
import urllib.request
import urllib.error
import cv2
import numpy as np

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8188")
WORKFLOWS_DIR = os.environ.get("WORKFLOWS_DIR", "workflows")
INPUT_DIR = "/app/ComfyUI/input"
OUTPUT_DIR = "/app/ComfyUI/output"


# ============================================================================
# Test Definitions — each workflow gets input overrides and a timeout
# ============================================================================

TESTS = {
    "esrgan_upscale": {
        "json": "esrgan_upscale.json",
        "overrides": {"1": {"image": "test_frame.png"}},
        "timeout": 120,
        "description": "ESRGAN 4x upscale",
        "setup": "extract_frame",
    },
    "rife_interpolation": {
        "json": "rife_interpolation_api.json",
        "overrides": {"1": {"video": "video.mp4"}},
        "timeout": 120,
        "description": "RIFE frame interpolation (2x)",
    },
    "wan_t2v": {
        "json": "wan_t2v.json",
        "overrides": {},
        "timeout": 600,
        "description": "Wan 2.2 Text-to-Video",
    },
    "wan_i2v": {
        "json": "wan_i2v.json",
        "overrides": {"5": {"image": "test_frame.png"}},
        "timeout": 600,
        "description": "Wan 2.2 Image-to-Video",
        "setup": "extract_frame",
    },
    "wan_fun_inpaint": {
        "json": "wan22_fun_inpaint_wrapper.json",
        "overrides": {
            "7": {"image": "frame_start.png"},
            "8": {"image": "frame_end.png"},
            "9": {"width": 960, "height": 544, "num_frames": 5},
        },
        "timeout": 600,
        "description": "Wan Fun Inpaint (diffusion interpolation)",
        "setup": "extract_start_end",
    },
    "mask_generation": {
        "json": "MaskGeneration_api.json",
        "overrides": {},
        "timeout": 180,
        "description": "SAM2 mask generation",
        "skip_reason": "Requires tracked bboxes JSON from detection step",
    },
    "multi_person_detection": {
        "json": "MultiPersonDetection_api.json",
        "overrides": {"1": {"video": "video.mp4"}},
        "timeout": 120,
        "description": "Multi-person detection (YOLO + tracking)",
    },
    "character_swap_masked": {
        "json": "TTFCharacterSwapMasked_api.json",
        "overrides": {},
        "timeout": 900,
        "description": "Character swap with external mask",
        "skip_reason": "Requires reference image + mask video — run via replace_characters.py",
    },
    "motion_capture": {
        "json": "motion_capture_api.json",
        "overrides": {
            "10": {"file": "video.mp4"},
            "11": {"file": "mask_video.mp4"},
            "15": {"video": "video.mp4"},
        },
        "timeout": 300,
        "description": "GVHMR + ViTPose skeleton overlay + BVH",
        "requires": ["video.mp4", "mask_video.mp4"],
    },
    "sam3d_objects": {
        "json": "sam3d_objects_api.json",
        "overrides": {"1": {"image": "test_frame.png"}},
        "timeout": 2400,
        "description": "SAM3D single object → 3D mesh",
        "setup": "extract_frame",
        "skip_reason": "Takes 10-40 min — run separately with test_object_to_3d.py",
    },
    "gvhmr_static": {
        "json": "GVHMR_api.json",
        "overrides": {"14": {"file": "video.mp4"}, "15": {"file": "mask_video.mp4"}},
        "timeout": 300,
        "description": "GVHMR pose estimation (static camera)",
        "requires": ["video.mp4", "mask_video.mp4"],
        "needs_output_node": {
            "99": {
                "class_type": "SaveStringKJ",
                "inputs": {"string": ["16", 0], "filename_prefix": "gvhmr_test", "output_folder": "output"},
                "_meta": {"title": "Save NPZ Path"},
            }
        },
    },
    "smpl_to_bvh": {
        "json": "smpl_to_bvh_pipeline_api.json",
        "overrides": {},
        "timeout": 30,
        "description": "SMPL NPZ → BVH conversion",
        "skip_reason": "Requires NPZ from GVHMR — tested as part of motion_capture pipeline",
    },
    "sdxl_txt2img": {
        "json": "sdxl_txt2img.json",
        "overrides": {},
        "timeout": 120,
        "description": "SDXL text-to-image",
    },
    "flux_txt2img": {
        "json": "flux_txt2img.json",
        "overrides": {},
        "timeout": 120,
        "description": "Flux text-to-image",
    },
}


# ============================================================================
# Setup Helpers
# ============================================================================

def ensure_test_frame():
    """Extract a test frame from video.mp4 if not already present."""
    path = f"{INPUT_DIR}/test_frame.png"
    if os.path.exists(path):
        return True
    try:
        cap = cv2.VideoCapture(f"{INPUT_DIR}/video.mp4")
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(path, frame)
            print(f"  [setup] Extracted test_frame.png")
            return True
    except Exception as e:
        print(f"  [setup] Failed to extract test frame: {e}")
    return False


def ensure_start_end_frames():
    """Extract start and end frames from video.mp4."""
    start_path = f"{INPUT_DIR}/frame_start.png"
    end_path = f"{INPUT_DIR}/frame_end.png"
    if os.path.exists(start_path) and os.path.exists(end_path):
        return True
    try:
        cap = cv2.VideoCapture(f"{INPUT_DIR}/video.mp4")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(start_path, frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(end_path, frame)
        cap.release()
        print(f"  [setup] Extracted frame_start.png + frame_end.png")
        return True
    except Exception as e:
        print(f"  [setup] Failed to extract frames: {e}")
    return False


# ============================================================================
# ComfyUI API Helpers
# ============================================================================

def queue_prompt(workflow: dict) -> str:
    """Submit workflow to ComfyUI, return prompt_id."""
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            err_data = json.loads(body)
            details = err_data.get("error", {}).get("message", "")
            node_errors = err_data.get("node_errors", {})
            msgs = [details] if details else []
            for nid, nerr in node_errors.items():
                for e_item in nerr.get("errors", []):
                    msgs.append(f"Node {nid}: {e_item.get('message', '')} — {e_item.get('details', '')}")
            raise RuntimeError(f"HTTP {e.code}: {'; '.join(msgs) if msgs else body[:200]}")
        except (json.JSONDecodeError, RuntimeError):
            if isinstance(sys.exc_info()[1], RuntimeError):
                raise
            raise RuntimeError(f"HTTP {e.code}: {body[:200]}")
    result = json.loads(resp.read())
    if "error" in result:
        raise RuntimeError(f"Queue error: {result['error']}")
    if "node_errors" in result and result["node_errors"]:
        errors = result["node_errors"]
        msgs = []
        for nid, err in errors.items():
            if isinstance(err, dict):
                for e in err.get("errors", []):
                    msgs.append(f"Node {nid}: {e.get('message', '')} — {e.get('details', '')}")
        raise RuntimeError(f"Validation errors: {'; '.join(msgs)}")
    return result["prompt_id"]


def wait_for_result(prompt_id: str, timeout: int) -> dict:
    """Poll until workflow completes or fails."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=10
            )
            data = json.loads(resp.read())
            if prompt_id in data:
                history = data[prompt_id]
                status = history.get("status", {})
                if status.get("completed"):
                    return {"status": "success", "outputs": history.get("outputs", {})}
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    error_msg = "Unknown error"
                    for m in msgs:
                        if m[0] == "execution_error":
                            error_msg = m[1].get("exception_message", "")[:300]
                            break
                    return {"status": "error", "error": error_msg}
        except Exception:
            pass
        time.sleep(2)
    return {"status": "timeout", "error": f"Timed out after {timeout}s"}


def check_vram():
    """Get current VRAM usage."""
    try:
        resp = urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
        data = json.loads(resp.read())
        devices = data.get("devices", [])
        if devices:
            gpu = devices[0]
            total = gpu.get("vram_total", 0) / (1024 ** 2)
            free = gpu.get("vram_free", 0) / (1024 ** 2)
            return round(total - free), round(total)
    except Exception:
        pass
    return 0, 0


def purge_vram():
    """Call wrapper's VRAM purge endpoint."""
    try:
        req = urllib.request.Request(
            "http://localhost:8189/vram/purge",
            data=b"",
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read())
        return result.get("success", False), result.get("method", "unknown")
    except Exception as e:
        return False, str(e)


# ============================================================================
# Main Test Runner
# ============================================================================

def run_test(name: str, test_def: dict) -> dict:
    """Run a single workflow test."""
    json_file = os.path.join(WORKFLOWS_DIR, test_def["json"])

    # Check workflow file exists
    if not os.path.exists(json_file):
        return {"status": "skip", "reason": f"JSON not found: {test_def['json']}"}

    # Check skip reason
    if "skip_reason" in test_def:
        return {"status": "skip", "reason": test_def["skip_reason"]}

    # Check required input files
    for req_file in test_def.get("requires", []):
        if not os.path.exists(f"{INPUT_DIR}/{req_file}"):
            return {"status": "skip", "reason": f"Missing input: {req_file}"}

    # Run setup
    setup = test_def.get("setup")
    if setup == "extract_frame":
        if not ensure_test_frame():
            return {"status": "skip", "reason": "Could not extract test frame"}
    elif setup == "extract_start_end":
        if not ensure_start_end_frames():
            return {"status": "skip", "reason": "Could not extract start/end frames"}

    # Load workflow
    with open(json_file) as f:
        wf = json.load(f)

    # Apply overrides
    for node_id, overrides in test_def.get("overrides", {}).items():
        if node_id in wf:
            wf[node_id]["inputs"].update(overrides)

    # Add output node if needed
    if "needs_output_node" in test_def:
        wf.update(test_def["needs_output_node"])

    # Check VRAM before
    vram_before, vram_total = check_vram()

    # Submit
    start = time.time()
    try:
        prompt_id = queue_prompt(wf)
    except RuntimeError as e:
        return {"status": "error", "error": str(e), "time": 0}

    # Wait
    result = wait_for_result(prompt_id, test_def.get("timeout", 300))
    elapsed = time.time() - start

    # Check VRAM after
    vram_after, _ = check_vram()

    result["time"] = round(elapsed, 1)
    result["vram_before"] = vram_before
    result["vram_after"] = vram_after

    # Count outputs
    if result.get("outputs"):
        out_count = 0
        for nid, node_out in result["outputs"].items():
            for key in ["images", "gifs", "files", "text"]:
                items = node_out.get(key, [])
                if isinstance(items, list):
                    out_count += len(items)
        result["output_count"] = out_count

    return result


def main():
    parser = argparse.ArgumentParser(description="Test all ComfyUI workflows")
    parser.add_argument("--only", help="Comma-separated list of tests to run")
    parser.add_argument("--skip", help="Comma-separated list of tests to skip")
    parser.add_argument("--no-purge", action="store_true", help="Don't purge VRAM between GPU tests")
    parser.add_argument("--list", action="store_true", help="Just list available tests")
    args = parser.parse_args()

    if args.list:
        for name, t in TESTS.items():
            skip = " [SKIP]" if "skip_reason" in t else ""
            print(f"  {name:30s} {t['description']}{skip}")
        return 0

    # Filter tests
    test_names = list(TESTS.keys())
    if args.only:
        only = set(args.only.split(","))
        test_names = [n for n in test_names if n in only]
    if args.skip:
        skip = set(args.skip.split(","))
        test_names = [n for n in test_names if n not in skip]

    # GPU workflows that need VRAM purge between runs
    gpu_workflows = {
        "esrgan_upscale", "rife_interpolation",
        "wan_t2v", "wan_i2v", "wan_fun_inpaint", "motion_capture",
        "gvhmr_static", "character_swap_masked", "sam3d_objects",
        "sdxl_txt2img", "flux_txt2img", "multi_person_detection",
    }

    print("=" * 70)
    print("  ComfyUI Workflow Test Runner")
    print("=" * 70)
    vram_used, vram_total = check_vram()
    print(f"  Server: {COMFYUI_URL}")
    print(f"  VRAM:   {vram_used}MB / {vram_total}MB")
    print(f"  Tests:  {len(test_names)}")
    print("=" * 70)
    print()

    results = {}
    for i, name in enumerate(test_names):
        test_def = TESTS[name]
        desc = test_def["description"]
        print(f"[{i+1}/{len(test_names)}] {name}: {desc}")

        # Purge VRAM before GPU workflows
        if name in gpu_workflows and not args.no_purge:
            vram_used, _ = check_vram()
            if vram_used > 800:
                print(f"  Purging VRAM ({vram_used}MB)...")
                ok, method = purge_vram()
                if ok:
                    print(f"  VRAM purged ({method})")
                else:
                    print(f"  VRAM purge warning: {method}")
                time.sleep(2)

        result = run_test(name, test_def)
        results[name] = result

        status = result["status"]
        if status == "success":
            outputs = result.get("output_count", 0)
            elapsed = result.get("time", 0)
            print(f"  ✓ PASS ({elapsed}s, {outputs} outputs, VRAM: {result.get('vram_after', '?')}MB)")
        elif status == "skip":
            print(f"  ⊘ SKIP: {result.get('reason', '')}")
        elif status == "error":
            print(f"  ✗ FAIL: {result.get('error', '')[:200]}")
        elif status == "timeout":
            print(f"  ✗ TIMEOUT: {result.get('error', '')}")
        print()

    # Summary
    passed = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] in ("error", "timeout"))
    skipped = sum(1 for r in results.values() if r["status"] == "skip")

    print("=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)

    if failed:
        print("\n  FAILURES:")
        for name, r in results.items():
            if r["status"] in ("error", "timeout"):
                print(f"    {name}: {r.get('error', '')[:200]}")

    # Save report
    report_path = "test_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())