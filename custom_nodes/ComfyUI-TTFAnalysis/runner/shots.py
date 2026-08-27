"""
Shot boundaries — PySceneDetect (BSD-3-Clause), CPU.

Runs FIRST and everything downstream is per-shot. Depth scale, flow, and derived
normals all break across a cut: a whole cut-sequence run produces output that is
subtly wrong rather than obviously wrong, which is worse.
"""


def detect(video_path, threshold, n_frames):
    from scenedetect import ContentDetector, SceneManager, open_video
    v = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(v, show_progress=False)
    sl = sm.get_scene_list()
    if not sl:                                    # no cuts -> one shot
        return [(0, n_frames)]
    shots = [(s[0].get_frames(), min(s[1].get_frames(), n_frames)) for s in sl]
    if shots[0][0] > 0:
        shots.insert(0, (0, shots[0][0]))
    return [(a, b) for a, b in shots if b > a]
