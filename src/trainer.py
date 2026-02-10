"""
YOLO Model Trainer

Comprehensive training pipeline for custom YOLO models.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch


class YOLOTrainer:
    """
    Trainer class for YOLO models.
    
    Provides methods for training, validation, and fine-tuning.
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        data_yaml: str = 'data/dataset.yaml',
        project: str = 'runs/train',
        name: str = 'exp',
        device: str = 'auto'
    ):
        """
        Initialize trainer.
        
        Args:
            model_name: YOLO model variant or path to weights
            data_yaml: Path to dataset configuration
            project: Project directory for outputs
            name: Experiment name
            device: Device for training ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.project = project
        self.name = name
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self.model = YOLO(model_name)
        
        print(f"✓ YOLOTrainer initialized")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Dataset: {data_yaml}")
    
    def train(
        self,
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = 0.01,
        resume: bool = False,
        patience: int = 50,
        save_period: int = -1,
        workers: int = 8,
        optimizer: str = 'auto',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train YOLO model.
        
        Args:
            epochs: Number of training epochs
            batch: Batch size
            imgsz: Input image size
            lr0: Initial learning rate
            resume: Resume from last checkpoint
            patience: Early stopping patience
            save_period: Save checkpoint every n epochs (-1 to disable)
            workers: Number of data loading workers
            optimizer: Optimizer ('SGD', 'Adam', 'AdamW', 'auto')
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        print("\n" + "="*70)
        print("STARTING YOLO TRAINING")
        print("="*70)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch}")
        print(f"Image size: {imgsz}")
        print(f"Device: {self.device}")
        print("="*70 + "\n")
        
        # Train model
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            device=self.device,
            project=self.project,
            name=self.name,
            resume=resume,
            patience=patience,
            save_period=save_period,
            workers=workers,
            optimizer=optimizer,
            verbose=True,
            **kwargs
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Results saved to: {self.model.trainer.save_dir}")
        print("="*70 + "\n")
        
        return results
    
    def validate(
        self,
        model_path: Optional[str] = None,
        split: str = 'val'
    ) -> Dict[str, Any]:
        """
        Validate model on dataset.
        
        Args:
            model_path: Path to model weights (uses current model if None)
            split: Dataset split to validate on ('train', 'val', 'test')
            
        Returns:
            Validation metrics
        """
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        print("\n" + "="*70)
        print("VALIDATING MODEL")
        print("="*70)
        
        results = model.val(
            data=self.data_yaml,
            split=split,
            device=self.device
        )
        
        print("\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
        print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
        print(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}")
        print("="*70 + "\n")
        
        return results.results_dict
    
    def fine_tune(
        self,
        pretrained_weights: str,
        epochs: int = 50,
        freeze_layers: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fine-tune pre-trained model on custom dataset.
        
        Args:
            pretrained_weights: Path to pre-trained weights
            epochs: Number of fine-tuning epochs
            freeze_layers: Number of layers to freeze from backbone
            **kwargs: Additional training arguments
            
        Returns:
            Training results
        """
        # Load pre-trained model
        self.model = YOLO(pretrained_weights)
        
        # Freeze layers
        if freeze_layers > 0:
            print(f"Freezing first {freeze_layers} layers...")
            for i, (name, param) in enumerate(self.model.model.named_parameters()):
                if i < freeze_layers:
                    param.requires_grad = False
        
        # Train with lower learning rate
        return self.train(
            epochs=epochs,
            lr0=kwargs.get('lr0', 0.001),  # Lower learning rate for fine-tuning
            **kwargs
        )
    
    def hyperparameter_tuning(
        self,
        iterations: int = 10,
        **kwargs
    ):
        """
        Perform hyperparameter tuning.
        
        Args:
            iterations: Number of tuning iterations
            **kwargs: Fixed training parameters
        """
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING")
        print("="*70)
        print(f"Iterations: {iterations}")
        print("="*70 + "\n")
        
        results = self.model.tune(
            data=self.data_yaml,
            iterations=iterations,
            device=self.device,
            **kwargs
        )
        
        print("\n" + "="*70)
        print("TUNING COMPLETED")
        print("="*70)
        
        return results
    
    def export_model(
        self,
        format: str = 'onnx',
        imgsz: int = 640
    ) -> str:
        """
        Export trained model.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'engine', 'tflite')
            imgsz: Input image size
            
        Returns:
            Path to exported model
        """
        print(f"\nExporting model to {format}...")
        
        export_path = self.model.export(
            format=format,
            imgsz=imgsz
        )
        
        print(f"✓ Model exported to: {export_path}")
        
        return export_path


def train_from_config(config_path: str):
    """
    Train model from YAML configuration file.
    
    Args:
        config_path: Path to training configuration
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = YOLOTrainer(
        model_name=config['model'],
        data_yaml=config['data'],
        project=config.get('project', 'runs/train'),
        name=config.get('name', 'exp'),
        device=config.get('device', 'auto')
    )
    
    # Train model
    results = trainer.train(
        epochs=config.get('epochs', 100),
        batch=config.get('batch', 16),
        imgsz=config.get('imgsz', 640),
        lr0=config.get('lr0', 0.01),
        patience=config.get('patience', 50),
        workers=config.get('workers', 8),
        optimizer=config.get('optimizer', 'auto')
    )
    
    # Validate
    val_results = trainer.validate()
    
    # Export if specified
    if config.get('export', {}).get('enabled', False):
        export_format = config['export'].get('format', 'onnx')
        trainer.export_model(format=export_format)
    
    return results


def main():
    """Example usage."""
    # Initialize trainer
    trainer = YOLOTrainer(
        model_name='yolov8n.pt',
        data_yaml='data/dataset.yaml'
    )
    
    # Train model
    results = trainer.train(
        epochs=100,
        batch=16,
        imgsz=640
    )
    
    # Validate
    val_results = trainer.validate()
    
    # Export
    trainer.export_model(format='onnx')


if __name__ == "__main__":
    main()
