#!/usr/bin/env python3
"""
YOLO Detection Script

Detect objects in images, videos, or webcam streams.

Usage:
    python scripts/detect.py --source image.jpg --model yolov8n.pt
    python scripts/detect.py --source video.mp4 --model yolov8n.pt --save
    python scripts/detect.py --source 0 --model yolov8n.pt  # Webcam
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import YOLODetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    
    parser.add_argument('--source', type=str, required=True,
                       help='Image/video path or webcam index (0)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Model path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--save', action='store_true',
                       help='Save detection results')
    parser.add_argument('--save-dir', type=str, default='runs/detect',
                       help='Directory to save results')
    parser.add_argument('--show', action='store_true',
                       help='Display results')
    parser.add_argument('--classes', type=int, nargs='+',
                       help='Filter by class IDs')
    parser.add_argument('--max-det', type=int, default=300,
                       help='Maximum detections per image')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main detection function."""
    args = parse_args()
    
    print("="*70)
    print("YOLO OBJECT DETECTION")
    print("="*70)
    print(f"Source: {args.source}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = YOLODetector(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        verbose=args.verbose
    )
    
    # Detect objects
    results = detector.detect(
        source=args.source,
        save=args.save,
        save_dir=args.save_dir,
        show=args.show,
        classes=args.classes,
        max_det=args.max_det
    )
    
    # Print summary
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    
    if len(results) == 1:
        result = results[0]
        print(f"Detected {result['num_detections']} objects")
        
        # Count by class
        class_counts = {}
        for det in result['detections']:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nDetections by class:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    else:
        total_detections = sum(r['num_detections'] for r in results)
        print(f"Processed {len(results)} images/frames")
        print(f"Total detections: {total_detections}")
        print(f"Average per image: {total_detections/len(results):.1f}")
    
    print("="*70 + "\n")
    
    if args.save:
        print(f"✓ Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
