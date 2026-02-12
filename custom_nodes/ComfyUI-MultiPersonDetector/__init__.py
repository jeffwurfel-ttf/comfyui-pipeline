"""
Multi-Person Detection & Tracking Nodes for ComfyUI

Includes:
- MultiPersonBboxDetector: Detect all people in video frames
- SaveBboxesJSON: Save detections to JSON
- LoadBboxesJSON: Load detections from JSON
- FilterBboxesByPerson: Extract single person from tracked data
- LoadTrackedBboxesForPerson: Load tracked bboxes for SAM2 segmentation
- LoadTrackedBboxesInfo: Get info about tracked people

Installation:
1. Create folder: ComfyUI/custom_nodes/ComfyUI-MultiPersonDetector/
2. Copy this file as: __init__.py
3. Restart ComfyUI
"""

import json
import os
import numpy as np

try:
    import folder_paths
    from comfy.utils import ProgressBar
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    folder_paths = None
    ProgressBar = None


class MultiPersonBboxDetector:
    """
    Detects bounding boxes for ALL people in each frame.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Minimum confidence for detection"
                }),
                "max_people": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Maximum number of people to detect per frame"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("bboxes_json", "num_people_detected",)
    FUNCTION = "detect"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Detects bounding boxes for ALL people in each frame."

    def detect(self, model, images, confidence_threshold=0.5, max_people=10):
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) is required")

        detector = model["yolo"]
        B, H, W, C = images.shape
        shape = np.array([H, W])[None]
        images_np = images.numpy()
        detector.reinit()
        pbar = ProgressBar(B) if ProgressBar else None
        all_frame_bboxes = []
        max_people_in_any_frame = 0

        for frame_idx, img in enumerate(images_np):
            img_resized = cv2.resize(img, (640, 640))
            img_input = img_resized.transpose(2, 0, 1)[None]
            detections = detector(img_input, shape, single_person=False)[0]
            frame_bboxes = []
            
            for det in detections:
                bbox = det.get("bbox")
                if bbox is None:
                    continue
                conf = bbox[4] if len(bbox) > 4 else 1.0
                if conf < confidence_threshold:
                    continue
                x1, y1, x2, y2 = bbox[:4]
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                frame_bboxes.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "confidence": float(conf)
                })
                if len(frame_bboxes) >= max_people:
                    break

            frame_bboxes.sort(key=lambda x: x["confidence"], reverse=True)
            all_frame_bboxes.append(frame_bboxes)
            max_people_in_any_frame = max(max_people_in_any_frame, len(frame_bboxes))
            if pbar:
                pbar.update_absolute(frame_idx + 1)
            if frame_idx % 50 == 0:
                print(f"[MultiPersonDetector] Frame {frame_idx}/{B}, detected {len(frame_bboxes)} people")

        detector.cleanup()
        output = {
            "num_frames": B,
            "frame_width": W,
            "frame_height": H,
            "max_people_detected": max_people_in_any_frame,
            "frames": all_frame_bboxes
        }
        bboxes_json = json.dumps(output, indent=2)
        print(f"[MultiPersonDetector] Complete: {B} frames, max {max_people_in_any_frame} people")
        return (bboxes_json, max_people_in_any_frame,)


class SaveBboxesJSON:
    """Saves bboxes JSON to a file."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes_json": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "bboxes"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "WanAnimatePreprocess"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves bboxes JSON to the output folder."

    def save(self, bboxes_json, filename_prefix="bboxes"):
        if folder_paths:
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
            os.makedirs(output_dir, exist_ok=True)

        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:04d}.json"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1

        with open(filepath, 'w') as f:
            f.write(bboxes_json)

        print(f"[SaveBboxesJSON] Saved to: {filepath}")
        return {"ui": {"files": [{"filename": filename, "subfolder": "", "type": "output"}]}}


class LoadBboxesJSON:
    """Loads bboxes JSON from a file."""

    @classmethod
    def INPUT_TYPES(s):
        if folder_paths:
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        else:
            files = []
        return {
            "required": {
                "json_file": (sorted(files) if files else ["none"], {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bboxes_json",)
    FUNCTION = "load"
    CATEGORY = "WanAnimatePreprocess"

    @classmethod
    def IS_CHANGED(s, json_file):
        return float("nan")

    def load(self, json_file):
        if folder_paths:
            input_dir = folder_paths.get_input_directory()
        else:
            input_dir = os.path.join(os.path.dirname(__file__), "..", "..", "input")
        filepath = os.path.join(input_dir, json_file)
        with open(filepath, 'r') as f:
            bboxes_json = f.read()
        print(f"[LoadBboxesJSON] Loaded from: {filepath}")
        return (bboxes_json,)


class FilterBboxesByPerson:
    """Filters bboxes JSON to only include a specific tracked person."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tracked_bboxes_json": ("STRING", {"forceInput": True}),
                "person_id": ("INT", {"default": 0, "min": 0, "max": 50}),
            },
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("person_bboxes",)
    FUNCTION = "filter"
    CATEGORY = "WanAnimatePreprocess"

    def filter(self, tracked_bboxes_json, person_id=0):
        data = json.loads(tracked_bboxes_json)
        people = data.get("people", {})
        person_key = str(person_id)
        if person_key not in people:
            raise ValueError(f"Person ID {person_id} not found. Available: {list(people.keys())}")

        person_data = people[person_key]
        frames = person_data.get("frames", {})
        bboxes = []
        num_frames = data.get("num_frames", max(int(k) for k in frames.keys()) + 1)

        for frame_idx in range(num_frames):
            frame_key = str(frame_idx)
            if frame_key in frames:
                bbox = frames[frame_key]
                # Wrap in list for SAM2 format
                bboxes.append([(int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"]))])
            else:
                if bboxes:
                    bboxes.append(bboxes[-1])
                else:
                    bboxes.append([(0, 0, 0, 0)])
        return (bboxes,)


class LoadTrackedBboxesForPerson:
    """
    Loads tracked bboxes JSON and outputs bboxes for a specific person.
    Output format matches what Sam2Segmentation expects.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_file": ("STRING", {
                    "default": "tracked_bboxes.json",
                    "tooltip": "JSON filename in ComfyUI input folder"
                }),
                "person_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "tooltip": "Which person to extract (0-indexed)"
                }),
                "num_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Number of frames (0 = auto from JSON)"
                }),
            },
        }

    RETURN_TYPES = ("BBOX", "INT", "INT")
    RETURN_NAMES = ("bboxes", "frame_count", "person_count")
    FUNCTION = "load"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Loads tracked bboxes for a specific person from JSON."

    @classmethod
    def IS_CHANGED(s, json_file, person_id, num_frames):
        return float("nan")

    def load(self, json_file, person_id=0, num_frames=0):
        if folder_paths:
            input_dir = folder_paths.get_input_directory()
        else:
            input_dir = os.path.join(os.path.dirname(__file__), "..", "..", "input")

        filepath = os.path.join(input_dir, json_file)
        with open(filepath, 'r') as f:
            data = json.load(f)

        total_frames = num_frames if num_frames > 0 else data.get("num_frames", 0)
        num_people = data.get("num_people", len(data.get("people", {})))
        people = data.get("people", {})
        person_key = str(person_id)

        if person_key not in people:
            raise ValueError(f"Person ID {person_id} not found. Available: {list(people.keys())}")

        person_data = people[person_key]
        frames_dict = person_data.get("frames", {})
        first_frame = person_data.get("first_frame", 0)

        # Get the first valid bbox to use for frames before person appears
        first_valid_bbox = None
        if str(first_frame) in frames_dict:
            bbox = frames_dict[str(first_frame)]
            first_valid_bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]

        # Build bbox list - one per frame, wrapped in list for SAM2 format
        # SAM2 expects: [[[x1,y1,x2,y2]], [[x1,y1,x2,y2]], ...] - list of frames, each with list of bboxes
        bboxes = []
        for frame_idx in range(total_frames):
            frame_key = str(frame_idx)
            if frame_key in frames_dict:
                bbox = frames_dict[frame_key]
                bboxes.append([[bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]])
            else:
                # Use first valid bbox for preceding frames, or last bbox for subsequent
                if bboxes:
                    bboxes.append(bboxes[-1])  # Repeat last
                elif first_valid_bbox:
                    bboxes.append([first_valid_bbox])  # Wrap in list
                else:
                    bboxes.append([[0, 0, 100, 100]])  # Fallback

        print(f"[LoadTrackedBboxes] Loaded {len(bboxes)} frames for person {person_id}")
        print(f"  First bbox: {bboxes[0] if bboxes else 'none'}")
        return (bboxes, total_frames, num_people)


class LoadTrackedBboxesInfo:
    """Loads tracked bboxes JSON and returns info about tracked people."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_file": ("STRING", {
                    "default": "tracked_bboxes.json",
                    "tooltip": "JSON filename in ComfyUI input folder"
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("num_people", "num_frames", "frame_width", "frame_height", "info_text")
    FUNCTION = "load_info"
    CATEGORY = "WanAnimatePreprocess"

    @classmethod
    def IS_CHANGED(s, json_file):
        return float("nan")

    def load_info(self, json_file):
        if folder_paths:
            input_dir = folder_paths.get_input_directory()
        else:
            input_dir = os.path.join(os.path.dirname(__file__), "..", "..", "input")

        filepath = os.path.join(input_dir, json_file)
        with open(filepath, 'r') as f:
            data = json.load(f)

        num_frames = data.get("num_frames", 0)
        frame_width = data.get("frame_width", 0)
        frame_height = data.get("frame_height", 0)
        num_people = data.get("num_people", len(data.get("people", {})))

        info_lines = [f"Tracked {num_people} people across {num_frames} frames"]
        info_lines.append(f"Resolution: {frame_width}x{frame_height}")
        for pid, pdata in data.get("people", {}).items():
            info_lines.append(f"  Person {pid}: frames {pdata.get('first_frame')}-{pdata.get('last_frame')} ({pdata.get('frame_count')} frames)")

        info_text = "\n".join(info_lines)
        print(f"[LoadTrackedBboxesInfo] {info_text}")
        return (num_people, num_frames, frame_width, frame_height, info_text)


# Node registration
NODE_CLASS_MAPPINGS = {
    "MultiPersonBboxDetector": MultiPersonBboxDetector,
    "SaveBboxesJSON": SaveBboxesJSON,
    "LoadBboxesJSON": LoadBboxesJSON,
    "FilterBboxesByPerson": FilterBboxesByPerson,
    "LoadTrackedBboxesForPerson": LoadTrackedBboxesForPerson,
    "LoadTrackedBboxesInfo": LoadTrackedBboxesInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiPersonBboxDetector": "Multi-Person Bbox Detector",
    "SaveBboxesJSON": "Save Bboxes JSON",
    "LoadBboxesJSON": "Load Bboxes JSON",
    "FilterBboxesByPerson": "Filter Bboxes By Person",
    "LoadTrackedBboxesForPerson": "Load Tracked Bboxes (Person)",
    "LoadTrackedBboxesInfo": "Load Tracked Bboxes Info",
}

print("[ComfyUI-MultiPersonDetector] Loaded successfully - 6 nodes available")