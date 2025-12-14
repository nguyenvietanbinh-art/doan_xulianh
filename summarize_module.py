# summarize_module.py
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

SAMPLE_RATE = 1
HIST_DIFF_THRESH = 0.5
MOTION_DIFF_THRESH = 25
ROI_MIN_AREA = 500
KEYFRAMES_PER_SHOT = 2
SUMMARY_FRAME_DURATION = 1
OUTPUT_FPS = 15

__all__ = ["summarize_video"]

def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame.copy())
        ok, frame = cap.read()
    cap.release()
    return frames, fps

def hist_diff(a, b, bins=32):
    ha = cv2.calcHist([a], [0,1,2], None, [bins,bins,bins], [0,180,0,256,0,256])
    hb = cv2.calcHist([b], [0,1,2], None, [bins,bins,bins], [0,180,0,256,0,256])
    ha = cv2.normalize(ha, ha).flatten()
    hb = cv2.normalize(hb, hb).flatten()
    corr = cv2.compareHist(ha.astype('float32'), hb.astype('float32'), cv2.HISTCMP_CORREL)
    return 1.0 - corr

def detect_shot_boundaries(frames, thresh=HIST_DIFF_THRESH):
    if len(frames) == 0:
        return []
    boundaries = [0]
    for i in range(1, len(frames)):
        try:
            d = hist_diff(cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2HSV),
                          cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV))
        except Exception:
            d = 0
        if d > thresh:
            boundaries.append(i)
    boundaries.append(len(frames))
    shots = []
    for i in range(len(boundaries)-1):
        shots.append((boundaries[i], boundaries[i+1]))
    return shots

def compute_motion_heatmaps(frames, blur_ksize=5):
    if len(frames) == 0:
        return []
    masks = [np.zeros(frames[0].shape[:2], dtype=np.uint8)]
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g1 = cv2.GaussianBlur(g1, (blur_ksize, blur_ksize), 0)
        g2 = cv2.GaussianBlur(g2, (blur_ksize, blur_ksize), 0)
        diff = cv2.absdiff(g1, g2)
        _, th = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        masks.append(th)
    return masks

def score_frames_by_roi(masks):
    if len(masks) == 0:
        return []
    return [int(np.sum(m > 0)) for m in masks]

def frame_feature_hist(frame, bins=32):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0,1,2], None, [bins,bins,bins], [0,180,0,256,0,256])
    return cv2.normalize(h, h).flatten()

def select_keyframes_in_shot(frames, indices, scores, k=KEYFRAMES_PER_SHOT):
    if len(indices) == 0:
        return []
    feats = np.array([frame_feature_hist(frames[i]) for i in indices])
    if len(indices) <= k:
        # return top-k by motion score (deterministic)
        sorted_by_score = sorted(indices, key=lambda i: scores[i] if i < len(scores) else 0, reverse=True)
        return sorted_by_score[:k]
    # cluster
    kmeans = KMeans(n_clusters=k, random_state=0).fit(feats)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    chosen = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(feats[members] - centers[c], axis=1)
        pick = members[np.argmin(dists)]
        chosen.append(indices[pick])
    return sorted(chosen)

def create_summary_video(frames, keyframe_indices, out_path,
                         fps_out=OUTPUT_FPS,
                         seconds_per_key=SUMMARY_FRAME_DURATION):
    # fallback nếu frames rỗng
    if len(frames) == 0:
        raise ValueError("Input video has no frames.")

    # fallback nếu không có keyframe
    if len(keyframe_indices) == 0:
        print("⚠ No keyframes found — using first frame as fallback.")
        keyframe_indices = [0]

    # lấy kích thước frame đầu tiên
    h, w = frames[0].shape[:2]

    # ép chiều rộng / chiều cao thành số chẵn (bắt buộc cho many codecs)
    if w % 2 == 1: w -= 1
    if h % 2 == 1: h -= 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps_out, (w, h))

    frames_per_key = max(1, int(fps_out * seconds_per_key))

    for idx in keyframe_indices:
        # safety index clamp
        if idx < 0:
            idx = 0
        if idx >= len(frames):
            idx = len(frames) - 1
        frame = frames[idx]
        frame = cv2.resize(frame, (w, h))
        for _ in range(frames_per_key):
            writer.write(frame)

    writer.release()

    # verify
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise IOError(f"Output video not written or empty: {out_path}")
    return out_path

def compute_compression_ratio(orig_fps, orig_n_frames, summary_seconds):
    if orig_fps <= 0 or orig_n_frames <= 0:
        return None
    return summary_seconds / (orig_n_frames / orig_fps)

def compute_recall(pred_keyframe_indices, ground_truth_intervals, fps):
    if not ground_truth_intervals:
        return None
    if not pred_keyframe_indices:
        return 0.0
    pred_times = [i / fps for i in pred_keyframe_indices]
    covered = 0
    for (s,e) in ground_truth_intervals:
        if any((t >= s and t <= e) for t in pred_times):
            covered += 1
    return covered / len(ground_truth_intervals)

def _resolve_input_path(video_input):
    """
    Accept either:
    - a string path
    - a gradio upload object (dict with 'name' or 'data')
    Return a filesystem path string.
    """
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        # new gradio sometimes returns {'name': '/tmp/..', 'data': '/tmp/...'}
        return video_input.get("name") or video_input.get("data")
    # fallback: try to convert
    try:
        return str(video_input)
    except Exception:
        return None

def summarize_video(video_input, output_path="summary_output.mp4", gt_intervals=None):
    """
    Main function to call.
    - video_input: path string or gradio upload dict
    - output_path: where to save summary mp4
    Returns a dict:
      {
        "output_path": str,
        "keyframes": [indices],
        "compression_ratio": float,
        "recall": float or None
      }
    """
    input_path = _resolve_input_path(video_input)
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    print("Reading frames...")
    frames, fps = read_video_frames(input_path)
    n = len(frames)
    print(f"Frames: {n}, FPS: {fps}")

    if n == 0:
        raise ValueError("No frames read from input video.")

    print("Detecting shots...")
    shots = detect_shot_boundaries(frames)
    print(f"Detected {len(shots)} shots")

    print("Computing motion masks...")
    masks = compute_motion_heatmaps(frames)
    scores = score_frames_by_roi(masks)

    keyframes = []
    for start, end in shots:
        indices = list(range(start, end))
        if len(indices) == 0:
            continue
        chosen = select_keyframes_in_shot(frames, indices, scores, k=KEYFRAMES_PER_SHOT)
        # fallback fill
        if len(chosen) < KEYFRAMES_PER_SHOT:
            sorted_by_score = sorted(indices, key=lambda i: scores[i] if i < len(scores) else 0, reverse=True)
            for cand in sorted_by_score:
                if cand not in chosen:
                    chosen.append(cand)
                if len(chosen) >= KEYFRAMES_PER_SHOT:
                    break
        keyframes.extend(chosen)

    keyframes = sorted(set(keyframes))
    print(f"Selected {len(keyframes)} keyframes")

    # create summary video
    out = create_summary_video(frames, keyframes, output_path, fps_out=OUTPUT_FPS, seconds_per_key=SUMMARY_FRAME_DURATION)

    summary_seconds = len(keyframes) * SUMMARY_FRAME_DURATION
    comp_ratio = compute_compression_ratio(fps, n, summary_seconds)
    recall = None
    if gt_intervals is not None:
        recall = compute_recall(keyframes, gt_intervals, fps)

    result = {
        "output_path": out,
        "keyframes": keyframes,
        "compression_ratio": comp_ratio,
        "recall": recall
    }
    print("Summarization done:", result)
    return result
