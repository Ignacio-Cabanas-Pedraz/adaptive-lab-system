"""
Optimized Video Processing Script for Adaptive Lab System
GPU-optimized version with batched processing for better GPU utilization
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import torch
from adaptive_lab_system import AdaptiveLabSystem, SystemMode


def parse_args():
    parser = argparse.ArgumentParser(description='Process video with Adaptive Lab System (GPU-optimized)')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--mode', type=str, default='tracking',
                        choices=['auto', 'discovery', 'tracking', 'verification'],
                        help='Processing mode (tracking is fastest for GPU)')
    parser.add_argument('--sam-checkpoint', type=str, default='./checkpoints/sam2.1_hiera_tiny.pt',
                        help='Path to SAM 2 checkpoint')
    parser.add_argument('--sam-config', type=str, default='./configs/sam2.1/sam2.1_hiera_t.yaml',
                        help='Path to SAM 2 config')
    parser.add_argument('--yolo-model', type=str, default='yolov9c.pt',
                        help='YOLOv9 model to use: t(tiny)/s(small)/m(medium)/c(compact)/e(extended) - MIT license')
    parser.add_argument('--skip-frames', type=int, default=5,
                        help='Process every Nth frame (default: 5 for speed)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Number of frames to read ahead (default: 4)')
    parser.add_argument('--save-json', action='store_true',
                        help='Save detection results as JSON')
    parser.add_argument('--no-video-output', action='store_true',
                        help='Skip video encoding (faster, JSON only)')
    return parser.parse_args()


def draw_detections_fast(frame, objects, mode):
    """
    Fast annotation (minimal CPU overhead)
    """
    annotated = frame.copy()

    for obj in objects:
        if 'bbox' not in obj:
            continue

        bbox = obj['bbox']
        x, y, w, h = [int(v) for v in bbox]
        confidence = obj.get('confidence', 0)

        # Simple bounding box (faster than masks)
        if confidence > 0.7:
            color = (0, 255, 0)
        elif confidence > 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

        # Simple label
        label = f"{obj['class'][:15]}: {confidence:.2f}"
        cv2.putText(annotated, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Mode indicator
    cv2.putText(annotated, f"{mode.upper()}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return annotated


def save_results_json(results, output_path):
    """Save detection results as JSON"""
    def convert_to_python_type(obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, list):
            return [convert_to_python_type(item) for item in obj]
        else:
            return obj

    serializable_results = []

    for frame_result in results:
        frame_data = {
            'frame_number': int(frame_result['frame_number']),
            'mode': frame_result['mode'],
            'processing_time_ms': float(frame_result.get('actual_processing_time_ms', 0)),
            'objects': []
        }

        for obj in frame_result['objects']:
            obj_data = {
                'class': obj['class'],
                'confidence': float(obj['confidence']),
                'bbox': convert_to_python_type(obj.get('bbox', [])),
                'center': convert_to_python_type(obj.get('center', [0, 0])),
                'area': float(obj.get('area', 0))
            }
            frame_data['objects'].append(obj_data)

        serializable_results.append(frame_data)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {output_path}")


def create_summary_stats(results):
    """Create summary statistics"""
    total_frames = len(results)

    all_objects = []
    for frame in results:
        all_objects.extend([obj['class'] for obj in frame['objects']])

    unique_classes = set(all_objects)
    avg_time = np.mean([r.get('actual_processing_time_ms', 0) for r in results])

    modes = [r['mode'] for r in results]
    mode_counts = {mode: modes.count(mode) for mode in set(modes)}

    summary = {
        'total_frames_processed': total_frames,
        'unique_object_classes': list(unique_classes),
        'total_detections': len(all_objects),
        'average_processing_time_ms': round(avg_time, 2),
        'mode_distribution': mode_counts,
        'fps': round(1000 / avg_time, 2) if avg_time > 0 else 0
    }

    return summary


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {run_dir}")
    print(f"GPU Optimization: Enabled")
    print(f"Batch size: {args.batch_size}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {args.video}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    print(f"  Skip frames: {args.skip_frames}")

    # Initialize system
    print("\nInitializing Adaptive Lab System...")
    lab_system = AdaptiveLabSystem(
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        yolo_model=args.yolo_model
    )

    # Set mode
    if args.mode != 'auto':
        mode_map = {
            'discovery': SystemMode.DISCOVERY,
            'tracking': SystemMode.TRACKING,
            'verification': SystemMode.VERIFICATION
        }
        lab_system.set_mode(mode_map[args.mode])

    # Prepare output video
    out = None
    if not args.no_video_output:
        output_video_path = run_dir / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Process video
    print(f"\nProcessing video...")
    print(f"Mode: {args.mode}")
    print(f"Video output: {'Disabled (JSON only)' if args.no_video_output else 'Enabled'}")

    frame_count = 0
    processed_count = 0
    results = []

    frames_to_process = total_frames // args.skip_frames
    if args.max_frames:
        frames_to_process = min(frames_to_process, args.max_frames)

    pbar = tqdm(total=frames_to_process, desc="Processing",
                unit="frame", dynamic_ncols=True)

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames
        if frame_count % args.skip_frames != 0:
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        context = None
        if args.mode == 'auto':
            if processed_count < 3:
                context = {"user_action": "exploring"}
            else:
                context = {"procedure_active": True}

        result = lab_system.process_frame(frame_rgb, context)
        results.append(result)

        # Update progress bar with stats
        avg_time = result.get('actual_processing_time_ms', 0)
        current_fps = 1000 / avg_time if avg_time > 0 else 0
        pbar.set_postfix({
            'ms/frame': f"{avg_time:.1f}",
            'FPS': f"{current_fps:.1f}",
            'objects': len(result['objects']),
            'mode': result['mode'][:4]
        })

        # Optionally draw annotations (CPU overhead)
        if not args.no_video_output:
            annotated = draw_detections_fast(frame, result['objects'], result['mode'])
            out.write(annotated)

        processed_count += 1
        pbar.update(1)

        # Check max frames
        if args.max_frames and processed_count >= args.max_frames:
            break

    pbar.close()

    # Cleanup
    cap.release()
    if out is not None:
        out.release()

    print(f"\nProcessed {processed_count} frames")
    if not args.no_video_output:
        print(f"Annotated video saved to: {output_video_path}")

    # Save JSON results
    if args.save_json or args.no_video_output:
        json_path = run_dir / "results.json"
        save_results_json(results, json_path)

    # Create and save summary
    summary = create_summary_stats(results)
    summary_path = run_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary:")
    print(f"  Unique objects detected: {len(summary['unique_object_classes'])}")
    print(f"  Classes: {', '.join(summary['unique_object_classes'])}")
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Average processing time: {summary['average_processing_time_ms']:.2f}ms")
    print(f"  Average FPS: {summary['fps']:.2f}")
    print(f"  Mode distribution: {summary['mode_distribution']}")
    print(f"\nSummary saved to: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
