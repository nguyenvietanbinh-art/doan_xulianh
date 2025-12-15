import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

# --- CẤU HÌNH ---
CLIP_DURATION = 2.0       
HIST_DIFF_THRESH = 0.5    
MOTION_DIFF_THRESH = 25
KEYFRAMES_PER_SHOT = 1   
ROI_MIN_AREA = 500
OUTPUT_FPS = 30          

__all__ = ["summarize_video"]

def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame) 
        ok, frame = cap.read()
    cap.release()
    return frames, fps

def hist_diff(a, b, bins=32):
    ha = cv2.calcHist([a], [0,1,2], None, [bins,bins,bins], [0,180,0,256,0,256])
    hb = cv2.calcHist([b], [0,1,2], None, [bins,bins,bins], [0,180,0,256,0,256])
    ha = cv2.normalize(ha, ha).flatten()
    hb = cv2.normalize(hb, hb).flatten()
    return 1.0 - cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)

def detect_shot_boundaries(frames, thresh=HIST_DIFF_THRESH):
    if not frames: return []
    boundaries = [0]
    for i in range(1, len(frames)):
        try:
            d = hist_diff(cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2HSV),
                          cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV))
        except: d = 0
        if d > thresh:
            boundaries.append(i)
    boundaries.append(len(frames))
    return [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

def compute_motion_heatmaps(frames, blur_ksize=5):
    if not frames: return []
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
    return [int(np.sum(m > 0)) for m in masks]

def frame_feature_hist(frame, bins=32):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0,1,2], None, [bins,bins,bins], [0,180,0,256,0,256])
    return cv2.normalize(h, h).flatten()

def select_keyframes_using_clustering(frames, indices, scores, k=1):
    if not indices: return []
    
    if len(indices) <= k:
        return sorted(indices, key=lambda i: scores[i], reverse=True)[:k]

    feats = np.array([frame_feature_hist(frames[i]) for i in indices])
    
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(feats)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
    except:
        return sorted(indices, key=lambda i: scores[i], reverse=True)[:k]

    chosen = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0: continue
        
        member_indices = [indices[m] for m in members]
        
        best_in_cluster = max(member_indices, key=lambda i: scores[i])
        chosen.append(best_in_cluster)
        
    return sorted(chosen)

def create_summary_video_clips(frames, keyframes, out_path, fps_out=30):
    if not frames or not keyframes:
        raise ValueError("No frames to write.")

    h, w = frames[0].shape[:2]
    if w % 2 == 1: w -= 1
    if h % 2 == 1: h -= 1

    frames_per_clip = int(CLIP_DURATION * fps_out)
    half_clip = frames_per_clip // 2
    ranges = []

    for idx in keyframes:
        start = max(0, idx - half_clip)
        end = min(len(frames), idx + half_clip)
        ranges.append((start, end))

    merged_ranges = []
    if ranges:
        ranges.sort()
        curr_start, curr_end = ranges[0]
        for next_start, next_end in ranges[1:]:
            if next_start < curr_end: 
                curr_end = max(curr_end, next_end)
            else:
                merged_ranges.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_ranges.append((curr_start, curr_end))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(out_path, fourcc, fps_out, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps_out, (w, h))

    total_frames_written = 0
    for start, end in merged_ranges:
        for i in range(start, end):
            frame = cv2.resize(frames[i], (w, h))
            writer.write(frame)
            total_frames_written += 1

    writer.release()
    return out_path, total_frames_written

def compute_recall(pred_indices, gt_intervals, fps):
    if not gt_intervals or not pred_indices: return 0.0
    covered = 0
    pred_times = [i / fps for i in pred_indices]
    for (s, e) in gt_intervals:
        if any(s <= t <= e for t in pred_times):
            covered += 1
    return covered / len(gt_intervals)

def summarize_video(video_input, output_path="summary_output.mp4", gt_intervals=None):
    if isinstance(video_input, dict):
        input_path = video_input.get("name") or video_input.get("data")
    else:
        input_path = str(video_input)

    if not os.path.exists(input_path):
        raise FileNotFoundError("Input video not found")

    print(f"Reading frames from: {input_path}")
    frames, fps = read_video_frames(input_path)
    total_frames = len(frames)
    
    print("1. Detecting Shots...")
    shots = detect_shot_boundaries(frames)
    
    print("2. Computing Motion ROI...")
    masks = compute_motion_heatmaps(frames)
    scores = score_frames_by_roi(masks)
    
    print("3. Clustering Keyframes...")
    all_keyframes = []
    for start, end in shots:
        indices = list(range(start, end))
        if not indices: continue
        chosen = select_keyframes_using_clustering(frames, indices, scores, k=KEYFRAMES_PER_SHOT)
        all_keyframes.extend(chosen)
    
    all_keyframes = sorted(list(set(all_keyframes)))
    print(f"Selected {len(all_keyframes)} keyframes.")

    print("4. Generating Summary Video (Skimming)...")
    out_file, summary_frames_count = create_summary_video_clips(frames, all_keyframes, output_path, fps_out=OUTPUT_FPS)
    
    orig_duration = total_frames / fps
    sum_duration = summary_frames_count / OUTPUT_FPS
    comp_ratio = orig_duration / sum_duration if sum_duration > 0 else 0
    
    recall = None
    if gt_intervals:
        recall = compute_recall(all_keyframes, gt_intervals, fps)

    result = {
        "output_path": out_file,
        "keyframes": all_keyframes,
        "compression_ratio": comp_ratio,
        "recall": recall
    }
    
    print(f"DONE! Ratio: {comp_ratio:.2f}x")
    return result