# Benchmark all attention/transformer models on NEU-DET
import sys
import os
import time
import numpy as np
from multiprocessing import freeze_support

# Numpy 2.0 compatibility
if hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

# Use local ultralytics from repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NEU-DET-with-yolov8'))
from ultralytics import YOLO

# Models to benchmark
MODELS = {
    'ShuffleAttention': 'NEU-DET-with-yolov8/ultralytics/cfg/models/v8/yolov8_ShuffleAttention.yaml',
    'Swin-1': 'NEU-DET-with-yolov8/ultralytics/cfg/models/v8/yolov8_one_swinTrans.yaml',
    'Swin-3': 'NEU-DET-with-yolov8/ultralytics/cfg/models/v8/yolov8_three_swinTrans.yaml',
}

DATA_PATH = 'NEU-DET-with-yolov8/data/data.yaml'
EPOCHS = 100
BATCH_SIZE = 8

def main():
    results = {}

    for name, yaml_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"TRAINING: {name}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            model = YOLO(yaml_path)
            train_results = model.train(
                data=DATA_PATH,
                epochs=EPOCHS,
                imgsz=640,
                batch=BATCH_SIZE,
                device=0,
                workers=0,  # Disable multiprocessing for Windows
                project='NEU-DET-with-yolov8/runs',
                name=f'{name.lower()}_train',
                deterministic=True,
                amp=False
            )
            
            elapsed = time.time() - start_time
            
            # Get final metrics from results
            results_dict = getattr(train_results, 'results_dict', {})
            results[name] = {
                'mAP50': results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': results_dict.get('metrics/precision(B)', 0),
                'recall': results_dict.get('metrics/recall(B)', 0),
                'time_hours': elapsed / 3600
            }
            
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            results[name] = {'error': str(e)}

    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Model':<20} {'mAP50':>10} {'mAP50-95':>10} {'Precision':>10} {'Recall':>10} {'Time(h)':>10}")
    print("-" * 80)

    # Add baseline comparison
    print(f"{'Plain YOLOv8n':<20} {'73.7%':>10} {'40.2%':>10} {'64.0%':>10} {'71.8%':>10} {'0.55':>10}")
    print(f"{'CoordAtt':<20} {'73.6%':>10} {'39.8%':>10} {'66.2%':>10} {'68.5%':>10} {'1.15':>10}")

    for name, r in results.items():
        if 'error' in r:
            print(f"{name:<20} ERROR: {r['error'][:50]}")
        else:
            print(f"{name:<20} {r['mAP50']*100:>9.1f}% {r['mAP50-95']*100:>9.1f}% {r['precision']*100:>9.1f}% {r['recall']*100:>9.1f}% {r['time_hours']:>9.2f}")

    print(f"\n{'='*80}")

if __name__ == '__main__':
    freeze_support()
    main()
