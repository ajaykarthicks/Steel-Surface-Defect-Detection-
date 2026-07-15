# Steel Surface Defect Detection (YOLOv8 + ShuffleAttention)

Real-time detection of surface defects on steel sheets/coils, built to automate the kind of visual quality inspection that's normally done by hand on a production line.

Trained on the **NEU-DET** dataset, achieving **74.06% mAP@50** across 6 defect classes, with a custom **ShuffleAttention** module added to the YOLOv8 backbone to improve detection of small, low-contrast micro-defects.

---

## 📌 Problem Statement

Steel manufacturers need to inspect coils and sheets for surface defects (scratches, pitted surface, rolled-in scale, patches, crazing, inclusions) before shipping. Manual visual inspection on a fast-moving production line is slow, inconsistent between inspectors, and small/subtle defects are easy to miss — leading to defective product reaching customers, returns, rework, and reputational damage.

This project automates that inspection step with a real-time object detection pipeline, wrapped in simple GUI tools so it doesn't need to be run from raw scripts every time.

---

## 🚀 Features

- Real-time defect detection on images/video via a desktop GUI (`defect_detector_gui.py`)
- A separate GUI for training/retraining the model (`train_gui.py`)
- One-command project setup (`setup_project.py`)
- A benchmarking script to evaluate model performance across the dataset (`benchmark_all.py`)
- A Windows menu launcher (`run_menu.bat`) that ties setup, training, detection, and benchmarking together
- Custom **ShuffleAttention** blocks integrated into the YOLOv8 backbone for better micro-defect recall

---

## 🧠 Tech Stack

| Component | Tool / Library | Why |
|---|---|---|
| Detector | YOLOv8 (Ultralytics) | Fast, single-stage detector suitable for real-time inference |
| Attention | ShuffleAttention (custom `nn.Module`) | Lightweight channel + spatial attention that improves focus on small/low-contrast defects with minimal extra compute |
| Framework | PyTorch | Model training and custom layer implementation |
| Image I/O | OpenCV | Frame capture, pre-processing, and drawing final detections |
| GUI | Python (Tkinter/PyQt-style GUI scripts) | `train_gui.py` and `defect_detector_gui.py` give a point-and-click interface over the pipeline |
| Language | Python | Training, inference, and benchmarking pipeline |

---

## 🗂️ Dataset

**NEU-DET** — a standard steel surface defect dataset with 6 defect classes:

- Scratch
- Pitted Surface
- Rolled-in Scale
- Patches
- Crazing
- Inclusion

Annotations are in YOLO format (class + bounding box per image).

---

## 🏗️ Architecture

```
Input Image
    │
    ▼
Pre-processing (resize / letterbox / normalize)
    │
    ▼
YOLOv8 Backbone (CSPDarknet-style)
    │
    ▼
ShuffleAttention blocks  ── channel attention + spatial attention → channel shuffle
    │
    ▼
PANet Neck (multi-scale feature fusion)
    │
    ▼
Detection Head (bbox, objectness, class scores)
    │
    ▼
Non-Max Suppression (NMS)
    │
    ▼
Final Detections → shown in defect_detector_gui.py
```

---

## 📁 Project Structure

```
Steel-Surface-Defect-Detection/
├── setup_project.py        # One-time environment/project setup
├── train_gui.py            # GUI for training the YOLOv8 + ShuffleAttention model
├── defect_detector_gui.py  # GUI for running defect detection on images/video
├── benchmark_all.py        # Benchmarks model performance (mAP, speed, etc.)
├── run_menu.bat            # Windows launcher — menu to access setup/train/detect/benchmark
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Sudarsan003-max/Steel-Surface-Defect-Detection-.git
cd Steel-Surface-Defect-Detection-

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Then run the one-time setup:

```bash
python setup_project.py
```

---

## ▶️ Usage

### Option 1 — Windows menu (easiest)

Just double-click or run:

```bash
run_menu.bat
```

This opens a menu to jump straight into setup, training, detection, or benchmarking without remembering individual commands.

### Option 2 — Run scripts directly

**Train the model:**
```bash
python train_gui.py
```
Opens a GUI to configure and launch training (dataset path, epochs, image size, etc.).

**Run detection:**
```bash
python defect_detector_gui.py
```
Opens a GUI to load an image/video and view detected defects with bounding boxes and confidence scores.

**Benchmark the model:**
```bash
python benchmark_all.py
```
Runs evaluation across the dataset and reports metrics like mAP@50 and inference speed.

---

## 📊 Results

| Metric | Score |
|---|---|
| mAP@50 (overall) | **74.06%** |
| Dataset | NEU-DET (6 classes) |

ShuffleAttention specifically improved recall on **micro-defects** — the small, low-contrast defects that are both the hardest to catch and the most business-critical, since they're the ones inspectors miss most often.

---

## 🧩 Challenges

- **Small, imbalanced dataset** — some defect classes had far fewer samples than others; addressed with augmentation (flips, rotation, brightness jitter).
- **Missed micro-defects at default settings** — required tuning input resolution and confidence thresholds to recover recall without a spike in false positives.
- **Compute vs. accuracy trade-off** — adding ShuffleAttention increased training time slightly, so model complexity had to stay in balance with the real-time inference requirement.

---

## 🔮 Future Work

- Expand training data to cover more defect variety and lighting conditions
- Explore INT8 quantization / TensorRT export for faster edge deployment
- Integrate directly with a production-line camera feed for live monitoring
- Add an alerting layer for defects above a severity threshold

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.

---

## 🙋 Author

**Sudarsana Narayanan U R**
B.Tech CSE (AI & ML), SRM Institute of Science and Technology
[LinkedIn](https://linkedin.com/in/sudarsananarayanan-u-r/)
