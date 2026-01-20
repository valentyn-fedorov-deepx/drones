"""Export YOLOv8 model to TensorRT engine format."""
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    # Path to the PyTorch model
    model_path = Path("models/drones_hf/yolov11x/weight/best.pt")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model from {model_path}")
    model = YOLO(str(model_path))
    
    # Export to TensorRT engine
    # The engine will be saved in the same directory as the .pt file
    print("Exporting to TensorRT engine format...")
    print("This may take several minutes...")
    
    engine_path = model.export(
        format="engine",
        device="0",  # GPU 0
        half=True,   # FP16 for better performance
        imgsz=640,   # Inference image size
        simplify=True,
        workspace=4,  # Max workspace size in GB
    )
    
    print(f"\nTensorRT engine created successfully: {engine_path}")
    print(f"\nTo use it, update your config:")
    print(f"  use_tensorrt: true")
    print(f"  trt_engine: {Path(engine_path).name}")

if __name__ == "__main__":
    main()
