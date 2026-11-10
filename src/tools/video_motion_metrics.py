import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_video_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames were decoded from {video_path}")
    return frames, float(fps)


def center_crop(frame: np.ndarray, crop_ratio: float) -> np.ndarray:
    if crop_ratio >= 0.999:
        return frame
    if not (0.0 < crop_ratio <= 1.0):
        raise ValueError("crop_ratio must be in (0, 1].")
    h, w = frame.shape[:2]
    new_h = max(1, int(round(h * crop_ratio)))
    new_w = max(1, int(round(w * crop_ratio)))
    y0 = max(0, (h - new_h) // 2)
    x0 = max(0, (w - new_w) // 2)
    return frame[y0:y0 + new_h, x0:x0 + new_w]


def preprocess_frames(
    frames: List[np.ndarray],
    size_wh: Tuple[int, int],
    crop_ratio: float,
) -> List[np.ndarray]:
    width, height = size_wh
    out: List[np.ndarray] = []
    for frame in frames:
        cur = center_crop(frame, crop_ratio)
        cur = cv2.resize(cur, (width, height), interpolation=cv2.INTER_AREA)
        out.append(cur)
    return out


def uniform_resample_by_time(
    frames: List[np.ndarray],
    fps: float,
    target_len: int,
    target_fps: float,
) -> List[np.ndarray]:
    if target_len <= 0:
        raise ValueError("target_len must be positive.")

    src_n = len(frames)
    src_duration = max(src_n - 1, 0) / fps if src_n > 1 else 0.0

    resampled: List[np.ndarray] = []
    for i in range(target_len):
        t = 0.0 if target_len == 1 else i / target_fps
        if src_duration <= 1e-8:
            idx = 0
        else:
            alpha = min(max(t / src_duration, 0.0), 1.0)
            idx = int(round(alpha * (src_n - 1)))
        idx = min(max(idx, 0), src_n - 1)
        resampled.append(frames[idx])
    return resampled


def rgb_to_gray_float(frame_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def compute_farneback_flow(gray_prev: np.ndarray, gray_next: np.ndarray) -> np.ndarray:
    prev_u8 = np.clip(gray_prev * 255.0, 0, 255).astype(np.uint8)
    next_u8 = np.clip(gray_next * 255.0, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_u8,
        next=next_u8,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


def evaluate_pair(
    gt_video: str,
    pred_video: str,
    *,
    target_fps: float = 10.0,
    resize_width: int = 256,
    crop_ratio: float = 1.0,
) -> Dict:
    gt_frames, gt_fps = read_video_frames(gt_video)
    pred_frames, pred_fps = read_video_frames(pred_video)

    gt_h, gt_w = gt_frames[0].shape[:2]
    target_width = int(resize_width)
    target_height = int(round(gt_h * (target_width / gt_w)))

    gt_duration = len(gt_frames) / gt_fps
    pred_duration = len(pred_frames) / pred_fps
    common_duration = min(gt_duration, pred_duration)
    target_len = max(2, int(common_duration * target_fps))

    gt_frames = preprocess_frames(gt_frames, (target_width, target_height), crop_ratio)
    pred_frames = preprocess_frames(pred_frames, (target_width, target_height), crop_ratio)

    gt_frames = uniform_resample_by_time(gt_frames, gt_fps, target_len, target_fps)
    pred_frames = uniform_resample_by_time(pred_frames, pred_fps, target_len, target_fps)

    gt_float = [f.astype(np.float32) / 255.0 for f in gt_frames]
    pred_float = [f.astype(np.float32) / 255.0 for f in pred_frames]
    gt_gray = [rgb_to_gray_float(f) for f in gt_frames]
    pred_gray = [rgb_to_gray_float(f) for f in pred_frames]

    per_frame_l1 = []
    per_frame_psnr = []
    per_frame_ssim = []

    for g, p in zip(gt_float, pred_float):
        l1 = np.mean(np.abs(g - p))
        psnr = peak_signal_noise_ratio(g, p, data_range=1.0)
        ssim = structural_similarity(g, p, channel_axis=-1, data_range=1.0)

        per_frame_l1.append(float(l1))
        per_frame_psnr.append(float(psnr))
        per_frame_ssim.append(float(ssim))

    temporal_l1_list = []
    flow_epe_list = []
    flow_mag_l1_list = []

    diag = float(np.sqrt(target_width ** 2 + target_height ** 2))

    for t in range(1, target_len):
        d_gt = gt_float[t] - gt_float[t - 1]
        d_pred = pred_float[t] - pred_float[t - 1]
        temporal_l1 = np.mean(np.abs(d_gt - d_pred))
        temporal_l1_list.append(float(temporal_l1))

        flow_gt = compute_farneback_flow(gt_gray[t - 1], gt_gray[t])
        flow_pred = compute_farneback_flow(pred_gray[t - 1], pred_gray[t])

        flow_diff = flow_gt - flow_pred
        epe = np.mean(np.linalg.norm(flow_diff, axis=-1))
        flow_epe_list.append(float(epe))

        mag_gt = np.linalg.norm(flow_gt, axis=-1)
        mag_pred = np.linalg.norm(flow_pred, axis=-1)
        flow_mag_l1 = np.mean(np.abs(mag_gt - mag_pred))
        flow_mag_l1_list.append(float(flow_mag_l1))

    mean_l1 = float(np.mean(per_frame_l1))
    mean_psnr = float(np.mean(per_frame_psnr))
    mean_ssim = float(np.mean(per_frame_ssim))
    mean_temporal_l1 = float(np.mean(temporal_l1_list))
    mean_flow_epe = float(np.mean(flow_epe_list))
    mean_flow_mag_l1 = float(np.mean(flow_mag_l1_list))

    flow_epe_norm = mean_flow_epe / diag
    flow_mag_l1_norm = mean_flow_mag_l1 / diag

    appearance_diff = 0.5 * mean_l1 + 0.5 * (1.0 - mean_ssim)
    motion_diff = 0.5 * mean_temporal_l1 + 0.5 * flow_epe_norm
    overall_diff = 0.4 * appearance_diff + 0.6 * motion_diff

    return {
        "gt_video": str(gt_video),
        "pred_video": str(pred_video),
        "settings": {
            "target_fps": float(target_fps),
            "resize_width": int(target_width),
            "resize_height": int(target_height),
            "crop_ratio": float(crop_ratio),
            "num_aligned_frames": int(target_len),
            "gt_original_fps": float(gt_fps),
            "pred_original_fps": float(pred_fps),
            "common_duration_sec": float(common_duration),
        },
        "metrics": {
            "l1": mean_l1,
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "temporal_l1": mean_temporal_l1,
            "flow_epe": mean_flow_epe,
            "flow_epe_norm": flow_epe_norm,
            "flow_mag_l1": mean_flow_mag_l1,
            "flow_mag_l1_norm": flow_mag_l1_norm,
            "appearance_diff": appearance_diff,
            "motion_diff": motion_diff,
            "overall_diff": overall_diff,
        },
    }


def main():
    parser = argparse.ArgumentParser("Video-to-video motion-aware evaluation")
    parser.add_argument("--gt_video", required=True, type=str)
    parser.add_argument("--pred_video", required=True, type=str)
    parser.add_argument("--target_fps", default=10.0, type=float)
    parser.add_argument("--resize_width", default=256, type=int)
    parser.add_argument("--crop_ratio", default=1.0, type=float)
    parser.add_argument("--out_json", default="", type=str)
    args = parser.parse_args()

    result = evaluate_pair(
        gt_video=args.gt_video,
        pred_video=args.pred_video,
        target_fps=args.target_fps,
        resize_width=args.resize_width,
        crop_ratio=args.crop_ratio,
    )

    print(json.dumps(result, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
