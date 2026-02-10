#!/usr/bin/env python3
"""
YOLO Training Script

Train custom YOLO models on your dataset.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --model yolov8n.pt --data data/dataset.yaml --epochs 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainer import YOLOTrainer, train_from_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLO Model')
    
    parser.add_argument('--config', type=str,
                       help='Path to training config YAML')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Model variant or path to weights')
    parser.add_argument('--data', type=str, default='data/dataset.yaml',
                       help='Path to dataset YAML')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of workers')
    parser.add_argument('--optimizer', type=str, default='auto',
                       help='Optimizer (SGD/Adam/AdamW/auto)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Train from config if provided
    if args.config:
        print(f"Training from config: {args.config}\n")
        results = train_from_config(args.config)
        return
    
    # Initialize trainer
    trainer = YOLOTrainer(
        model_name=args.model,
        data_yaml=args.data,
        project=args.project,
        name=args.name,
        device=args.device
    )
    
    # Train model
    results = trainer.train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        resume=args.resume,
        patience=args.patience,
        workers=args.workers,
        optimizer=args.optimizer
    )
    
    # Validate
    print("\nRunning validation...")
    val_results = trainer.validate()
    
    print("\n✓ Training completed successfully!")
    print(f"✓ Best model saved to: {trainer.project}/{trainer.name}/weights/best.pt")


if __name__ == "__main__":
    main()
