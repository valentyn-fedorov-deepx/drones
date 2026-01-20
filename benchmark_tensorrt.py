"""Benchmark TensorRT engine vs PyTorch model for FPS comparison."""
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def benchmark_model(model_path: str, num_iterations: int = 100, warmup: int = 10):
    """Benchmark inference speed of a model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {Path(model_path).name}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...")
    model = YOLO(model_path)
    
    # Create dummy input (640x640 RGB image)
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        model(dummy_img, verbose=False, imgsz=640)
    
    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    times = []
    
    for i in range(num_iterations):
        start = time.perf_counter()
        results = model(dummy_img, verbose=False, imgsz=640)
        end = time.perf_counter()
        
        inference_time = (end - start) * 1000  # Convert to ms
        times.append(inference_time)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_iterations}")
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / mean_time
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Mean inference time: {mean_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"  Min/Max: {min_time:.2f} ms / {max_time:.2f} ms")
    print(f"  Average FPS: {fps:.1f}")
    print(f"{'='*60}\n")
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps
    }

def main():
    model_dir = Path("models/drones_hf/yolov11x/weight")
    
    pt_model = model_dir / "best.pt"
    engine_model = model_dir / "best.engine"
    
    print("\n" + "="*60)
    print("TensorRT vs PyTorch Performance Benchmark")
    print("="*60)
    
    results = {}
    
    # Benchmark PyTorch model
    if pt_model.exists():
        results['pytorch'] = benchmark_model(str(pt_model))
    else:
        print(f"PyTorch model not found: {pt_model}")
    
    # Benchmark TensorRT engine
    if engine_model.exists():
        results['tensorrt'] = benchmark_model(str(engine_model))
    else:
        print(f"TensorRT engine not found: {engine_model}")
    
    # Compare results
    if 'pytorch' in results and 'tensorrt' in results:
        pt_fps = results['pytorch']['fps']
        trt_fps = results['tensorrt']['fps']
        speedup = trt_fps / pt_fps
        
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"PyTorch:   {pt_fps:.1f} FPS ({results['pytorch']['mean_time']:.2f} ms)")
        print(f"TensorRT:  {trt_fps:.1f} FPS ({results['tensorrt']['mean_time']:.2f} ms)")
        print(f"Speedup:   {speedup:.2f}x faster with TensorRT")
        print("="*60)

if __name__ == "__main__":
    main()
