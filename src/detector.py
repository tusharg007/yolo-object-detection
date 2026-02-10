"""
YOLO Object Detector

High-performance object detection with YOLOv8.
Supports real-time detection, batch processing, and custom models.
"""

import os
from typing import List, Union, Optional, Dict, Any
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO


class YOLODetector:
    """
    Main YOLO detector class for object detection tasks.
    
    Attributes:
        model: YOLO model instance
        device: Device for inference ('cuda' or 'cpu')
        conf_threshold: Confidence threshold for detection
        iou_threshold: IoU threshold for NMS
    """
    
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        verbose: bool = False
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device for inference ('cuda', 'cpu', or 'auto')
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
            verbose: Whether to print verbose output
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model
        self.model = self._load_model()
        
        # Get class names
        self.class_names = self.model.names
        
        if verbose:
            print(f"✓ YOLODetector initialized")
            print(f"  Model: {model_path}")
            print(f"  Device: {self.device}")
            print(f"  Classes: {len(self.class_names)}")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model."""
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def detect(
        self,
        source: Union[str, np.ndarray, List[str]],
        save: bool = False,
        save_dir: str = 'runs/detect',
        show: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in image(s) or video.
        
        Args:
            source: Image path, numpy array, video path, or list of paths
            save: Whether to save results
            save_dir: Directory to save results
            show: Whether to display results
            conf: Confidence threshold (overrides default)
            iou: IoU threshold (overrides default)
            classes: Filter by class IDs
            max_det: Maximum detections per image
            
        Returns:
            List of detection results
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        
        # Run inference
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            classes=classes,
            max_det=max_det,
            device=self.device,
            verbose=self.verbose,
            save=save,
            project=save_dir if save else None,
            show=show
        )
        
        # Parse results
        parsed_results = []
        for result in results:
            parsed_results.append(self._parse_result(result))
        
        return parsed_results
    
    def _parse_result(self, result) -> Dict[str, Any]:
        """Parse YOLO result into dictionary format."""
        boxes = result.boxes
        
        detections = []
        for i in range(len(boxes)):
            detection = {
                'bbox': boxes.xyxy[i].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                'confidence': float(boxes.conf[i].cpu().numpy()),
                'class_id': int(boxes.cls[i].cpu().numpy()),
                'class_name': self.class_names[int(boxes.cls[i].cpu().numpy())]
            }
            detections.append(detection)
        
        return {
            'image_path': result.path,
            'image_shape': result.orig_shape,
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def detect_image(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        draw_boxes: bool = True
    ) -> Dict[str, Any]:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to image
            save_path: Path to save annotated image
            draw_boxes: Whether to draw bounding boxes
            
        Returns:
            Detection results with annotated image
        """
        results = self.detect(image_path)[0]
        
        if draw_boxes:
            image = cv2.imread(image_path)
            annotated_image = self.draw_detections(image, results['detections'])
            results['annotated_image'] = annotated_image
            
            if save_path:
                cv2.imwrite(save_path, annotated_image)
        
        return results
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
        skip_frames: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in video.
        
        Args:
            video_path: Path to video file or camera index (0 for webcam)
            output_path: Path to save output video
            show: Whether to display video
            skip_frames: Process every nth frame
            
        Returns:
            List of detection results for each frame
        """
        cap = cv2.VideoCapture(video_path if isinstance(video_path, str) else int(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_results = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if specified
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    continue
                
                # Detect objects
                results = self.detect(frame)[0]
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, results['detections'])
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if show:
                    cv2.imshow('YOLO Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                all_results.append(results)
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        
        return all_results
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image
            detections: List of detections
            thickness: Box thickness
            font_scale: Font scale for labels
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            conf = det['confidence']
            class_name = det['class_name']
            
            # Generate color based on class
            color = self._get_color(det['class_id'])
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return annotated
    
    def _get_color(self, class_id: int) -> tuple:
        """Generate consistent color for each class."""
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
    
    def benchmark(
        self,
        image_size: tuple = (640, 640),
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            image_size: Input image size
            num_runs: Number of inference runs
            
        Returns:
            Performance metrics
        """
        import time
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            self.detect(dummy_image)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.detect(dummy_image)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'mean_time': float(times.mean()),
            'std_time': float(times.std()),
            'min_time': float(times.min()),
            'max_time': float(times.max()),
            'fps': float(1.0 / times.mean())
        }
    
    def export_model(
        self,
        format: str = 'onnx',
        output_path: Optional[str] = None
    ) -> str:
        """
        Export model to different formats.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'coreml', 'engine', 'tflite')
            output_path: Output path for exported model
            
        Returns:
            Path to exported model
        """
        export_path = self.model.export(format=format)
        
        if output_path and export_path != output_path:
            import shutil
            shutil.move(export_path, output_path)
            export_path = output_path
        
        print(f"✓ Model exported to {export_path}")
        return export_path


def main():
    """Example usage."""
    # Initialize detector
    detector = YOLODetector('yolov8n.pt', verbose=True)
    
    # Detect in image
    results = detector.detect_image('example.jpg', save_path='output.jpg')
    print(f"Detected {results['num_detections']} objects")
    
    # Benchmark
    metrics = detector.benchmark()
    print(f"Average FPS: {metrics['fps']:.2f}")


if __name__ == "__main__":
    main()
