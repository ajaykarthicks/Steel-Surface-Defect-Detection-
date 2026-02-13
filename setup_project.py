"""
NEU-DET Project Setup Script
Run this on a new computer to set up the entire project
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed!")
        return False
    print(f"[OK] {description} complete")
    return True

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     NEU-DET Steel Defect Detection - Project Setup       ║
    ║              YOLOv8 + ShuffleAttention                   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Step 1: Create virtual environment if not exists
    if not os.path.exists(".venv"):
        if not run_command("python -m venv .venv", "Creating virtual environment"):
            return
    else:
        print("[OK] Virtual environment already exists")
    
    # Activate script based on OS
    if sys.platform == "win32":
        pip_path = ".venv\\Scripts\\pip.exe"
        python_path = ".venv\\Scripts\\python.exe"
    else:
        pip_path = ".venv/bin/pip"
        python_path = ".venv/bin/python"
    
    # Step 2: Upgrade pip
    run_command(f"{pip_path} install --upgrade pip", "Upgrading pip")
    
    # Step 3: Install PyTorch with CUDA
    print("\n[INFO] Installing PyTorch with CUDA support...")
    run_command(
        f"{pip_path} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
        "Installing PyTorch (CUDA 12.4)"
    )
    
    # Step 4: Install other dependencies
    dependencies = [
        "pillow",
        "numpy", 
        "opencv-python",
        "matplotlib",
        "pandas",
        "pyyaml",
        "tqdm",
        "scipy",
        "seaborn",
        "psutil",
        "py-cpuinfo",
        "thop",
    ]
    
    run_command(
        f"{pip_path} install {' '.join(dependencies)}",
        "Installing dependencies"
    )
    
    # Step 5: Verify installation
    print("\n" + "="*60)
    print("  Verifying Installation")
    print("="*60)
    
    verify_script = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import sys
sys.path.insert(0, "NEU-DET-with-yolov8")
from ultralytics import YOLO
print("Ultralytics (ShuffleAttention) loaded successfully!")
'''
    
    result = subprocess.run([python_path, "-c", verify_script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Step 6: Check dataset
    dataset_path = "NEU-DET-with-yolov8/data/NEU-DET"
    if os.path.exists(dataset_path):
        train_images = len(os.listdir(os.path.join(dataset_path, "train", "images")))
        test_images = len(os.listdir(os.path.join(dataset_path, "test", "images")))
        print(f"\n[OK] Dataset found: {train_images} train + {test_images} test images")
    else:
        print(f"\n[WARNING] Dataset not found at {dataset_path}")
        print("Please download the NEU-DET dataset and extract to:")
        print(f"  {os.path.abspath(dataset_path)}")
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                    Setup Complete!                       ║
    ╠══════════════════════════════════════════════════════════╣
    ║  To run detection GUI:                                   ║
    ║    .venv\\Scripts\\python.exe defect_detector_gui.py      ║
    ║                                                          ║
    ║  To train a new model:                                   ║
    ║    .venv\\Scripts\\python.exe train_gui.py                ║
    ╚══════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
