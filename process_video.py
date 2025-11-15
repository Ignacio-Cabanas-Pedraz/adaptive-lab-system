"""
Video Processing Script for Adaptive Lab System
Processes a video file and detects lab equipment using the adaptive system
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from adaptive_lab_system import AdaptiveLabSystem, SystemMode


def parse_args():
    parser = argparse.ArgumentParser(description='Process video with Adaptive Lab System')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'discovery', 'tracking', 'verification'],
                        help='Processing mode (auto switches modes intelligently)')
    parser.add_argument('--sam-checkpoint', type=str, default='./checkpoints/sam2.1_hiera_tiny.pt',
                        help='Path to SAM 2 checkpoint')
    parser.add_argument('--sam-config', type=str, default='./configs/sam2.1/sam2.1_hiera_t.yaml',
                        help='Path to SAM 2 config')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt',
                        help='YOLO model to use (n/s/m/l)')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Process every Nth frame (1 = all frames)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--save-masks', action='store_true',
                        help='Save individual mask visualizations')
    parser.add_argument('--save-json', action='store_true',
                        help='Save detection results as JSON')
    return parser.parse_args()


def draw_detections(frame, objects, mode):
    """
    Draw bounding boxes and labels on frame

    Args:
        frame: numpy array (H, W, 3) BGR
        objects: list of detected objects
        mode: system mode used

    Returns:
        annotated frame
    """
    annotated = frame.copy()

    for obj in objects:
        # Get bbox
        if 'bbox' in obj:
            bbox = obj['bbox']
            x, y, w, h = [int(v) for v in bbox]
        else:
            continue

        # Color based on confidence
        confidence = obj.get('confidence', 0)
        if confidence > 0.7:
            color = (0, 255, 0)  # Green - high confidence
        elif confidence > 0.5:
            color = (0, 255, 255)  # Yellow - medium confidence
        else:
            color = (0, 165, 255)  # Orange - low confidence

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

        # Draw mask if available
        if 'mask' in obj and obj['mask'] is not None:
            mask = obj['mask']
            # Create colored overlay
            overlay = annotated.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
            annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        # Label
        label = f"{obj['class']}: {confidence:.2f}"

        # Background for text
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x, y - label_h - 10), (x + label_w, y), color, -1)
        cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw mode indicator
    mode_text = f"Mode: {mode.upper()}"
    cv2.putText(annotated, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return annotated


def save_results_json(results, output_path):
    """Save detection results as JSON"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []

    for frame_result in results:
        frame_data = {
            'frame_number': frame_result['frame_number'],
            'mode': frame_result['mode'],
            'processing_time_ms': frame_result.get('actual_processing_time_ms', 0),
            'objects': []
        }

        for obj in frame_result['objects']:
            obj_data = {
                'class': obj['class'],
                'confidence': obj['confidence'],
                'bbox': obj.get('bbox', []),
                'center': obj.get('center', [0, 0]),
                'area': obj.get('area', 0)
            }
            frame_data['objects'].append(obj_data)

        serializable_results.append(frame_data)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {output_path}")


def create_summary_stats(results):
    """Create summary statistics from results"""
    total_frames = len(results)

    # Count objects
    all_objects = []
    for frame in results:
        all_objects.extend([obj['class'] for obj in frame['objects']])

    # Unique classes
    unique_classes = set(all_objects)

    # Average processing time
    avg_time = np.mean([r.get('actual_processing_time_ms', 0) for r in results])

    # Mode distribution
    modes = [r['mode'] for r in results]
    mode_counts = {mode: modes.count(mode) for mode in set(modes)}

    summary = {
        'total_frames_processed': total_frames,
        'unique_object_classes': list(unique_classes),
        'total_detections': len(all_objects),
        'average_processing_time_ms': round(avg_time, 2),
        'mode_distribution': mode_counts
    }

    return summary


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {run_dir}")

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

    # Initialize system
    print("\nInitializing Adaptive Lab System...")
    lab_system = AdaptiveLabSystem(
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        yolo_model=args.yolo_model
    )

    # Set mode if not auto
    if args.mode != 'auto':
        mode_map = {
            'discovery': SystemMode.DISCOVERY,
            'tracking': SystemMode.TRACKING,
            'verification': SystemMode.VERIFICATION
        }
        lab_system.set_mode(mode_map[args.mode])

    # Prepare output video
    output_video_path = run_dir / "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Process video
    print(f"\nProcessing video (every {args.skip_frames} frame(s))...")
    frame_count = 0
    processed_count = 0
    results = []

    # Calculate frames to process
    frames_to_process = total_frames // args.skip_frames
    if args.max_frames:
        frames_to_process = min(frames_to_process, args.max_frames)

    pbar = tqdm(total=frames_to_process, desc="Processing")

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
            # Let system determine mode automatically
            if processed_count < 3:
                context = {"user_action": "exploring"}
            else:
                context = {"procedure_active": True}

        result = lab_system.process_frame(frame_rgb, context)
        results.append(result)

        # Draw annotations
        annotated = draw_detections(frame, result['objects'], result['mode'])

        # Write to output video
        out.write(annotated)

        # Save individual mask visualizations if requested
        if args.save_masks and result['objects']:
            masks_dir = run_dir / "masks"
            masks_dir.mkdir(exist_ok=True)

            for idx, obj in enumerate(result['objects']):
                if 'mask' in obj and obj['mask'] is not None:
                    mask_img = (obj['mask'] * 255).astype(np.uint8)
                    mask_path = masks_dir / f"frame_{processed_count:04d}_obj_{idx}_{obj['class']}.png"
                    cv2.imwrite(str(mask_path), mask_img)

        processed_count += 1
        pbar.update(1)

        # Check max frames
        if args.max_frames and processed_count >= args.max_frames:
            break

    pbar.close()

    # Cleanup
    cap.release()
    out.release()

    print(f"\nProcessed {processed_count} frames")
    print(f"Annotated video saved to: {output_video_path}")

    # Save JSON results if requested
    if args.save_json:
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
    print(f"  Mode distribution: {summary['mode_distribution']}")
    print(f"\nSummary saved to: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
