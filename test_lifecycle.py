#!/usr/bin/env python3
"""
ComfyUI Container Lifecycle Test

Validates the container conforms to the node manager protocol:
  - Health check works when unloaded
  - Generate returns 503 when no model loaded
  - Model load works (warmup workflow runs)
  - Health/status report correct state after load
  - Generate produces an image
  - Reload same model is a fast no-op
  - Unload frees VRAM
  - State is clean after unload

Usage:
    python test_lifecycle.py                             # SDXL (default)
    python test_lifecycle.py --profile flux-dev           # Flux Dev
    python test_lifecycle.py --url http://10.0.1.5:8189   # Remote
    python test_lifecycle.py --skip-generate              # Skip gen (faster)
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)


PASS = 0
FAIL = 0
SKIP = 0


def check(desc, expected, actual):
    global PASS, FAIL
    if expected == actual:
        print(f"  \u2713 {desc}")
        PASS += 1
    else:
        print(f"  \u2717 {desc} (expected: {expected}, got: {actual})")
        FAIL += 1


def check_true(desc, condition):
    global PASS, FAIL
    if condition:
        print(f"  \u2713 {desc}")
        PASS += 1
    else:
        print(f"  \u2717 {desc}")
        FAIL += 1


def section(title):
    print(f"\n{title}")
    print("-" * 50)


# =============================================================================
# Tests
# =============================================================================

def test_health_unloaded(url):
    section("1. Health check (unloaded)")
    try:
        r = requests.get(f"{url}/health", timeout=10)
        check("HTTP 200", 200, r.status_code)
        d = r.json()
        check("status is healthy", "healthy", d.get("status"))
        check("model_loaded is False", False, d.get("model_loaded"))
        check("model_name is None", None, d.get("model_name"))
        check_true("vram_total_mb > 0", d.get("vram_total_mb", 0) > 0)
        check("comfyui connected", "connected", d.get("comfyui"))
        print(f"    GPU: {d.get('gpu_name')}")
        print(f"    VRAM: {d.get('vram_used_mb')}MB / {d.get('vram_total_mb')}MB")
        return True
    except requests.ConnectionError:
        print(f"  \u2717 Cannot connect to {url}")
        print(f"    Is the container running? Try: docker-compose up -d")
        return False
    except Exception as e:
        print(f"  \u2717 Error: {e}")
        return False


def test_generate_503(url):
    section("2. Generate without model (expect 503)")
    try:
        r = requests.post(
            f"{url}/generate",
            json={"prompt": "test", "width": 64, "height": 64, "num_steps": 1},
            timeout=10,
        )
        check("HTTP 503", 503, r.status_code)
    except Exception as e:
        print(f"  \u2717 Error: {e}")


def test_status_unloaded(url):
    section("3. Model status (unloaded)")
    try:
        r = requests.get(f"{url}/model/status", timeout=10)
        check("HTTP 200", 200, r.status_code)
        d = r.json()
        check("state is unloaded", "unloaded", d.get("state"))
        check("model_name is None", None, d.get("model_name"))
    except Exception as e:
        print(f"  \u2717 Error: {e}")


def test_list_profiles(url):
    section("4. List profiles")
    try:
        r = requests.get(f"{url}/model/profiles", timeout=10)
        check("HTTP 200", 200, r.status_code)
        d = r.json()
        profiles = d.get("profiles", [])
        check_true("has profiles", len(profiles) > 0)
        names = [p["name"] for p in profiles]
        print(f"    Available: {', '.join(names)}")
        check("no current profile", None, d.get("current"))
    except Exception as e:
        print(f"  \u2717 Error: {e}")


def test_load(url, profile):
    section(f"5. Load model (profile: {profile})")
    try:
        print(f"    Loading '{profile}'... (may take 30-120s)")
        start = time.time()
        r = requests.post(
            f"{url}/model/load",
            json={"model": profile},
            timeout=360,
        )
        elapsed = time.time() - start
        print(f"    Completed in {elapsed:.1f}s")

        check("HTTP 200", 200, r.status_code)
        if r.status_code != 200:
            print(f"    Response: {r.text[:500]}")
            return False

        d = r.json()
        check("success", True, d.get("success"))
        check("model matches", profile, d.get("model"))
        check_true("vram_used_mb > 0", d.get("vram_used_mb", 0) > 0)
        check_true("load_time_ms >= 0", d.get("load_time_ms", -1) >= 0)

        print(f"    VRAM used: {d.get('vram_used_mb')} MB")
        print(f"    Load time: {d.get('load_time_ms')} ms")
        return d.get("success", False)

    except requests.Timeout:
        print(f"  \u2717 Load timed out after 360s")
        return False
    except Exception as e:
        print(f"  \u2717 Error: {e}")
        return False


def test_health_loaded(url, profile):
    section("6. Health check (loaded)")
    try:
        r = requests.get(f"{url}/health", timeout=10)
        check("HTTP 200", 200, r.status_code)
        d = r.json()
        check("model_loaded is True", True, d.get("model_loaded"))
        check("model_name matches", profile, d.get("model_name"))
    except Exception as e:
        print(f"  \u2717 Error: {e}")


def test_status_loaded(url, profile):
    section("7. Model status (loaded)")
    try:
        r = requests.get(f"{url}/model/status", timeout=10)
        check("HTTP 200", 200, r.status_code)
        d = r.json()
        check("state is loaded", "loaded", d.get("state"))
        check("profile matches", profile, d.get("profile"))
        check_true("last_used not None", d.get("last_used") is not None)
        check_true("loaded_at not None", d.get("loaded_at") is not None)
        check_true("vram_used_mb > 0", d.get("vram_used_mb", 0) > 0)
    except Exception as e:
        print(f"  \u2717 Error: {e}")


def test_reload_noop(url, profile):
    section("8. Reload same model (should no-op)")
    try:
        start = time.time()
        r = requests.post(
            f"{url}/model/load", json={"model": profile}, timeout=30)
        elapsed = time.time() - start
        check("HTTP 200", 200, r.status_code)
        d = r.json()
        check("success", True, d.get("success"))
        check_true("fast (<5s = no-op)", elapsed < 5)
        check_true("already_loaded flag", d.get("already_loaded", False))
        print(f"    Took {elapsed:.2f}s")
    except Exception as e:
        print(f"  \u2717 Error: {e}")


def test_generate(url, profile):
    section("9. Generate image")

    payload = {
        "prompt": "a red cube on white background, simple, 3d render",
        "negative_prompt": "blurry, text, watermark",
        "width": 512,
        "height": 512,
        "num_steps": 8,
        "guidance_scale": 5.0,
        "seed": 42,
    }

    try:
        print(f"    Generating {payload['width']}x{payload['height']} "
              f"@ {payload['num_steps']} steps...")
        start = time.time()
        r = requests.post(f"{url}/generate", json=payload, timeout=300)
        elapsed = time.time() - start
        print(f"    Completed in {elapsed:.1f}s")

        check("HTTP 200", 200, r.status_code)
        d = r.json()
        check("success", True, d.get("success"))

        images = d.get("images", [])
        check_true("has image data", len(images) > 0)

        if images:
            b64 = images[0].get("base64", "")
            check_true("image base64 not empty", len(b64) > 100)

            # Save test image
            try:
                img_bytes = base64.b64decode(b64)
                out = Path("test_output.png")
                out.write_bytes(img_bytes)
                print(f"    Saved: {out} ({len(img_bytes):,} bytes)")
            except Exception:
                pass

            print(f"    Seed: {images[0].get('seed')}")

        print(f"    Latency: {d.get('latency_ms')} ms")

    except Exception as e:
        print(f"  \u2717 Error: {e}")


def test_unload(url):
    section("10. Unload model")
    try:
        r = requests.post(f"{url}/model/unload", timeout=30)
        check("HTTP 200", 200, r.status_code)
        d = r.json()
        check("success", True, d.get("success"))
        check_true("vram_freed_mb >= 0", d.get("vram_freed_mb", -1) >= 0)
        print(f"    VRAM freed: {d.get('vram_freed_mb')} MB")
    except Exception as e:
        print(f"  \u2717 Error: {e}")


def test_verify_unloaded(url):
    section("11. Verify unloaded state")
    try:
        r = requests.get(f"{url}/model/status", timeout=10)
        d = r.json()
        check("state is unloaded", "unloaded", d.get("state"))
        check("model_name is None", None, d.get("model_name"))

        r = requests.post(
            f"{url}/generate", json={"prompt": "test"}, timeout=10)
        check("generate returns 503", 503, r.status_code)
    except Exception as e:
        print(f"  \u2717 Error: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test ComfyUI container lifecycle + generation")
    parser.add_argument(
        "--url", default="http://localhost:8189",
        help="Wrapper API URL (default: http://localhost:8189)")
    parser.add_argument(
        "--profile", default="sdxl",
        help="Profile to test (default: sdxl)")
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip image generation (faster)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ComfyUI Container Lifecycle Test")
    print("=" * 60)
    print(f"  Target:  {args.url}")
    print(f"  Profile: {args.profile}")
    print("=" * 60)

    # 1-3: Unloaded state
    if not test_health_unloaded(args.url):
        print(f"\nABORTED: Cannot connect to {args.url}")
        sys.exit(1)

    test_generate_503(args.url)
    test_status_unloaded(args.url)

    # 4: Profiles
    test_list_profiles(args.url)

    # 5: Load
    if not test_load(args.url, args.profile):
        print(f"\nABORTED: Model load failed. Check logs:")
        print(f"  docker-compose logs -f comfyui")
        sys.exit(1)

    # 6-8: Loaded state
    test_health_loaded(args.url, args.profile)
    test_status_loaded(args.url, args.profile)
    test_reload_noop(args.url, args.profile)

    # 9: Generate
    if args.skip_generate:
        section("9. Generate image")
        global SKIP
        SKIP += 1
        print("  - Skipped (--skip-generate)")
    else:
        test_generate(args.url, args.profile)

    # 10-11: Unload
    test_unload(args.url)
    test_verify_unloaded(args.url)

    # Summary
    total = PASS + FAIL
    print(f"\n{'=' * 60}")
    if FAIL == 0:
        print(f"  ALL TESTS PASSED ({PASS}/{total})")
    else:
        print(f"  {FAIL} TESTS FAILED ({PASS}/{total} passed)")
    if SKIP:
        print(f"  ({SKIP} skipped)")
    print(f"{'=' * 60}")

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()