"""
NEU-DET Training GUI with Real-time Progress Visualization
Train YOLOv8 + ShuffleAttention model with live stats
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import os
import sys
import time
import json
from datetime import datetime

# Add local ultralytics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NEU-DET-with-yolov8'))

# Fix numpy 2.0 compatibility (trapz renamed to trapezoid)
import numpy as np
if hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

# Matplotlib for plotting
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Fix PyTorch 2.6 weights_only issue
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = patched_load


class TrainingCallback:
    """Custom callback to capture training metrics"""
    
    def __init__(self, queue, epochs, batch_size):
        self.queue = queue
        self.epochs = epochs
        self.batch_size = batch_size
        self.epoch = 0
        self.total_batches = 0
        self.current_batch = 0
        self.best_mAP50 = 0.0
        self.best_mAP50_95 = 0.0
        self.best_epoch = 0
        self.start_time = None
        
    def on_train_start(self, trainer):
        """Called when training starts"""
        # Calculate total batches per epoch
        if hasattr(trainer, 'train_loader') and trainer.train_loader:
            self.total_batches = len(trainer.train_loader)
        else:
            # Fallback: try to calculate from dataset size and batch size
            try:
                if hasattr(trainer, 'batch_size') and hasattr(trainer, 'trainset'):
                    self.total_batches = len(trainer.trainset) // trainer.batch_size + 1
                else:
                    self.total_batches = 100  # Default fallback
            except:
                self.total_batches = 100
        
        self.current_batch = 0
        self.start_time = time.time()
        self.queue.put({
            'type': 'train_start',
            'total_batches': self.total_batches,
            'save_dir': str(getattr(trainer, 'save_dir', ''))
        })
    
    def on_train_batch_end(self, trainer):
        """Called at end of each training batch"""
        self.current_batch += 1
        
        # Reset batch counter at start of each epoch
        if hasattr(trainer, 'epoch') and trainer.epoch != self.epoch:
            self.epoch = trainer.epoch
            self.current_batch = 1
        
        self.queue.put({
            'type': 'batch',
            'batch': self.current_batch,
            'total_batches': self.total_batches,
            'epoch': trainer.epoch + 1 if hasattr(trainer, 'epoch') else 1,
            'epochs': trainer.epochs if hasattr(trainer, 'epochs') else 100
        })

        if self.start_time and self.total_batches:
            epoch_now = trainer.epoch + 1 if hasattr(trainer, 'epoch') else 1
            epochs_total = trainer.epochs if hasattr(trainer, 'epochs') else self.epochs
            total_steps = epochs_total * self.total_batches
            current_step = ((epoch_now - 1) * self.total_batches) + self.current_batch
            if current_step > 0:
                elapsed = time.time() - self.start_time
                remaining = max(total_steps - current_step, 0) * (elapsed / current_step)
                self.queue.put({'type': 'time', 'elapsed': elapsed, 'remaining': remaining})
        
    def on_train_epoch_end(self, trainer):
        """Called at end of each training epoch"""
        metrics = {
            'type': 'epoch',
            'epoch': trainer.epoch + 1,
            'epochs': trainer.epochs,
            'box_loss': float(trainer.loss_items[0]) if trainer.loss_items is not None else 0,
            'cls_loss': float(trainer.loss_items[1]) if trainer.loss_items is not None else 0,
            'dfl_loss': float(trainer.loss_items[2]) if trainer.loss_items is not None else 0,
        }
        self.queue.put(metrics)
    
    def on_val_end(self, validator):
        """Called at end of validation"""
        mAP50 = mAP50_95 = precision = recall = 0.0

        # Try to read metrics from validator.metrics first
        try:
            if hasattr(validator, 'metrics') and hasattr(validator.metrics, 'results_dict'):
                results_dict = validator.metrics.results_dict
                if isinstance(results_dict, dict):
                    mAP50 = float(results_dict.get('metrics/mAP50(B)', 0))
                    mAP50_95 = float(results_dict.get('metrics/mAP50-95(B)', 0))
                    precision = float(results_dict.get('metrics/precision(B)', 0))
                    recall = float(results_dict.get('metrics/recall(B)', 0))
        except Exception as e:
            self.queue.put({'type': 'log', 'message': f'Error reading validator metrics: {e}'})

        # Fallback: read metrics from the results.csv file
        if mAP50 == 0 and mAP50_95 == 0:
            try:
                import csv
                csv_path = os.path.join(validator.trainer.save_dir, 'results.csv')
                if os.path.exists(csv_path):
                    with open(csv_path, 'r', newline='') as f:
                        reader = list(csv.DictReader(f))
                        if reader:
                            last_row = reader[-1]
                            for key, value in last_row.items():
                                key_stripped = key.strip()
                                value_stripped = value.strip() if isinstance(value, str) else value
                                if key_stripped == 'metrics/mAP50(B)':
                                    mAP50 = float(value_stripped)
                                elif key_stripped == 'metrics/mAP50-95(B)':
                                    mAP50_95 = float(value_stripped)
                                elif key_stripped == 'metrics/precision(B)':
                                    precision = float(value_stripped)
                                elif key_stripped == 'metrics/recall(B)':
                                    recall = float(value_stripped)
                            self.queue.put({'type': 'log', 'message': f'CSV metrics: mAP50={mAP50}, mAP50_95={mAP50_95}, precision={precision}, recall={recall}'})
                        else:
                            self.queue.put({'type': 'log', 'message': 'CSV file exists but is empty'})
                else:
                    self.queue.put({'type': 'log', 'message': f'CSV file not found at {csv_path}'})
            except Exception as e:
                self.queue.put({'type': 'log', 'message': f'Error reading CSV in on_val_end: {e}'})
        
        # Ensure we have reasonable values
        if mAP50 == 0 and mAP50_95 == 0:
            self.queue.put({'type': 'log', 'message': 'Warning: Validation metrics appear to be 0'})
        
        metrics = {
            'type': 'validation',
            'mAP50': mAP50,
            'mAP50_95': mAP50_95,
            'precision': precision,
            'recall': recall,
        }
        self.queue.put(metrics)
    
    def on_train_end(self, trainer):
        """Called at end of training"""
        # Try to read final metrics from CSV as fallback
        try:
            import csv
            csv_path = os.path.join(trainer.save_dir, 'results.csv')
            if os.path.exists(csv_path):
                with open(csv_path, 'r', newline='') as f:
                    reader = list(csv.DictReader(f))
                if reader:
                    last_row = reader[-1]
                    final_mAP50 = final_mAP50_95 = 0.0
                    for key, value in last_row.items():
                        key_stripped = key.strip()
                        value_stripped = value.strip() if isinstance(value, str) else value
                        if key_stripped == 'metrics/mAP50(B)':
                            final_mAP50 = float(value_stripped)
                        elif key_stripped == 'metrics/mAP50-95(B)':
                            final_mAP50_95 = float(value_stripped)
                    if final_mAP50 > 0 or final_mAP50_95 > 0:
                        if final_mAP50 > self.best_mAP50:
                            self.best_mAP50 = final_mAP50
                            self.best_mAP50_95 = final_mAP50_95
                            self.best_epoch = self.epochs  # last epoch
                        self.queue.put({'type': 'log', 'message': f'Fallback metrics from CSV: mAP50={final_mAP50}, mAP50-95={final_mAP50_95}'})
        except Exception as e:
            self.queue.put({'type': 'log', 'message': f'Error reading CSV fallback: {e}'})
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.queue.put({'type': 'time', 'elapsed': elapsed, 'remaining': 0})
        self.queue.put({'type': 'complete', 'save_dir': str(trainer.save_dir)})


class TrainingGUI:
    """GUI for training with real-time visualization"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("NEU-DET Model Training - YOLOv8 + ShuffleAttention")
        self.root.geometry("1200x900")
        self.root.configure(bg='#1e1e1e')
        
        # Training state
        self.training = False
        self.training_thread = None
        self.metrics_queue = queue.Queue()
        
        # Metrics history
        self.epochs = []
        self.box_losses = []
        self.cls_losses = []
        self.dfl_losses = []
        self.mAP50_history = []
        self.mAP50_95_history = []
        self.total_batches = 0
        self.elapsed_time = 0.0
        self.remaining_time = 0.0
        self.current_save_dir = ''
        self.last_csv_check = 0.0
        self.last_val_epoch = 0
        
        self.setup_ui()
        self.set_defaults_for_model()
        self.check_queue()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#1e1e1e')
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            title_frame,
            text="🔧 NEU-DET Model Training",
            font=('Helvetica', 18, 'bold'),
            fg='#ffffff',
            bg='#1e1e1e'
        ).pack(side=tk.LEFT)
        
        # Config frame
        config_frame = tk.LabelFrame(
            self.root,
            text=" Training Configuration ",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b',
            relief=tk.GROOVE
        )
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        config_inner = tk.Frame(config_frame, bg='#2b2b2b')
        config_inner.pack(padx=10, pady=10)
        
        # Model selection
        tk.Label(config_inner, text="Model:", fg='white', bg='#2b2b2b', font=('Helvetica', 10)).grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.model_var = tk.StringVar(value='ShuffleAttention')
        models = ['ShuffleAttention', 'CoordAtt', 'Swin-1 Transformer', 'Swin-3 Transformer', 'Plain YOLOv8n']
        model_combo = ttk.Combobox(config_inner, textvariable=self.model_var, values=models, width=20, state='readonly')
        model_combo.grid(row=0, column=1, padx=5, pady=5)
        model_combo.bind('<<ComboboxSelected>>', self.set_defaults_for_model)
        
        # Epochs
        tk.Label(config_inner, text="Epochs:", fg='white', bg='#2b2b2b', font=('Helvetica', 10)).grid(row=0, column=2, sticky='e', padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=100)
        self.epochs_spin = tk.Spinbox(config_inner, from_=10, to=500, textvariable=self.epochs_var, width=8)
        self.epochs_spin.grid(row=0, column=3, padx=5, pady=5)
        
        # Batch size
        tk.Label(config_inner, text="Batch:", fg='white', bg='#2b2b2b', font=('Helvetica', 10)).grid(row=0, column=4, sticky='e', padx=5, pady=5)
        self.batch_var = tk.IntVar(value=8)
        self.batch_spin = tk.Spinbox(config_inner, from_=1, to=64, textvariable=self.batch_var, width=8)
        self.batch_spin.grid(row=0, column=5, padx=5, pady=5)
        
        # Image size
        tk.Label(config_inner, text="ImgSize:", fg='white', bg='#2b2b2b', font=('Helvetica', 10)).grid(row=0, column=6, sticky='e', padx=5, pady=5)
        self.imgsz_var = tk.IntVar(value=640)
        self.imgsz_combo = ttk.Combobox(config_inner, textvariable=self.imgsz_var, values=['320', '416', '512', '640', '800'], width=8)
        self.imgsz_combo.grid(row=0, column=7, padx=5, pady=5)
        
        # Buttons
        btn_frame = tk.Frame(config_inner, bg='#2b2b2b')
        btn_frame.grid(row=0, column=8, padx=20)
        
        self.train_btn = tk.Button(
            btn_frame,
            text="▶ Start Training",
            command=self.start_training,
            font=('Helvetica', 11, 'bold'),
            bg='#28a745',
            fg='white',
            activebackground='#218838',
            relief=tk.FLAT,
            padx=15,
            pady=5,
            cursor='hand2'
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            btn_frame,
            text="⬛ Stop",
            command=self.stop_training,
            font=('Helvetica', 11, 'bold'),
            bg='#dc3545',
            fg='white',
            activebackground='#c82333',
            relief=tk.FLAT,
            padx=15,
            pady=5,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress frame - Overall epochs
        progress_frame = tk.Frame(self.root, bg='#1e1e1e')
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            progress_frame,
            text="Overall:",
            font=('Helvetica', 9),
            fg='#888888',
            bg='#1e1e1e',
            width=8
        ).pack(side=tk.LEFT)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to train",
            font=('Helvetica', 10),
            fg='#888888',
            bg='#1e1e1e',
            width=30
        )
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        # Batch progress frame - Current epoch batches
        batch_frame = tk.Frame(self.root, bg='#1e1e1e')
        batch_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        tk.Label(
            batch_frame,
            text="Epoch:",
            font=('Helvetica', 9),
            fg='#888888',
            bg='#1e1e1e',
            width=8
        ).pack(side=tk.LEFT)
        
        self.batch_progress_var = tk.DoubleVar(value=0)
        self.batch_progress_bar = ttk.Progressbar(
            batch_frame,
            variable=self.batch_progress_var,
            maximum=100,
            length=400,
            mode='determinate',
            style='Batch.Horizontal.TProgressbar'
        )
        self.batch_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.batch_label = tk.Label(
            batch_frame,
            text="Batch: 0/0",
            font=('Helvetica', 10),
            fg='#888888',
            bg='#1e1e1e',
            width=30
        )
        self.batch_label.pack(side=tk.LEFT, padx=10)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Charts
        charts_frame = tk.LabelFrame(
            content_frame,
            text=" Training Metrics ",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b',
            relief=tk.GROOVE
        )
        charts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Loss subplot
        self.ax_loss = self.fig.add_subplot(2, 2, 1)
        self.ax_loss.set_facecolor('#1e1e1e')
        self.ax_loss.set_title('Training Loss', color='white', fontsize=10)
        self.ax_loss.set_xlabel('Epoch', color='gray', fontsize=8)
        self.ax_loss.tick_params(colors='gray')
        for spine in self.ax_loss.spines.values():
            spine.set_color('gray')
        
        # mAP subplot
        self.ax_map = self.fig.add_subplot(2, 2, 2)
        self.ax_map.set_facecolor('#1e1e1e')
        self.ax_map.set_title('Validation mAP', color='white', fontsize=10)
        self.ax_map.set_xlabel('Epoch', color='gray', fontsize=8)
        self.ax_map.tick_params(colors='gray')
        for spine in self.ax_map.spines.values():
            spine.set_color('gray')
        
        # Box loss detail
        self.ax_box = self.fig.add_subplot(2, 2, 3)
        self.ax_box.set_facecolor('#1e1e1e')
        self.ax_box.set_title('Box Loss', color='white', fontsize=10)
        self.ax_box.tick_params(colors='gray')
        for spine in self.ax_box.spines.values():
            spine.set_color('gray')
        
        # Class loss detail
        self.ax_cls = self.fig.add_subplot(2, 2, 4)
        self.ax_cls.set_facecolor('#1e1e1e')
        self.ax_cls.set_title('Class Loss', color='white', fontsize=10)
        self.ax_cls.tick_params(colors='gray')
        for spine in self.ax_cls.spines.values():
            spine.set_color('gray')
        
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right side - Stats and Log
        right_frame = tk.Frame(content_frame, bg='#1e1e1e')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0), expand=False)
        
        # Current stats frame
        stats_frame = tk.LabelFrame(
            right_frame,
            text=" Current Stats ",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b',
            relief=tk.GROOVE
        )
        stats_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.stats_labels = {}
        stats = [
            ('Epoch', '0/0'),
            ('Box Loss', '-'),
            ('Cls Loss', '-'),
            ('DFL Loss', '-'),
            ('mAP50', '-'),
            ('mAP50-95', '-'),
            ('Precision', '-'),
            ('Recall', '-'),
            ('Time Elapsed', '-'),
            ('Time Remaining', '-'),
        ]
        
        for i, (name, default) in enumerate(stats):
            row_frame = tk.Frame(stats_frame, bg='#2b2b2b')
            row_frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(
                row_frame,
                text=f"{name}:",
                font=('Helvetica', 10),
                fg='#888888',
                bg='#2b2b2b',
                width=12,
                anchor='w'
            ).pack(side=tk.LEFT)
            
            label = tk.Label(
                row_frame,
                text=default,
                font=('Helvetica', 10, 'bold'),
                fg='#4a9eff',
                bg='#2b2b2b',
                width=12,
                anchor='e'
            )
            label.pack(side=tk.RIGHT)
            self.stats_labels[name] = label
        
        # Best results frame
        best_frame = tk.LabelFrame(
            right_frame,
            text=" Best Results ",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b',
            relief=tk.GROOVE
        )
        best_frame.pack(fill=tk.X, pady=5)
        
        self.best_labels = {}
        for name in ['Best mAP50', 'Best mAP50-95', 'Best Epoch']:
            row_frame = tk.Frame(best_frame, bg='#2b2b2b')
            row_frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(
                row_frame,
                text=f"{name}:",
                font=('Helvetica', 10),
                fg='#888888',
                bg='#2b2b2b',
                width=14,
                anchor='w'
            ).pack(side=tk.LEFT)
            
            label = tk.Label(
                row_frame,
                text='-',
                font=('Helvetica', 10, 'bold'),
                fg='#28a745',
                bg='#2b2b2b',
                width=10,
                anchor='e'
            )
            label.pack(side=tk.RIGHT)
            self.best_labels[name] = label
        
        # All runs stats frame
        runs_frame = tk.LabelFrame(
            right_frame,
            text=" All Trained Runs ",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b',
            relief=tk.GROOVE
        )
        runs_frame.pack(fill=tk.BOTH, expand=False, pady=5)

        columns = ("Run Name", "Epochs", "Best mAP50", "Best mAP50-95", "Args")
        self.runs_table = ttk.Treeview(runs_frame, columns=columns, show="headings", height=6)
        for col in columns:
            self.runs_table.heading(col, text=col)
            self.runs_table.column(col, width=90 if col!="Args" else 120, anchor='center')
        self.runs_table.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.runs_table.bind("<Double-1>", self.on_run_double_click)

        self.refresh_runs_table()
        
        # Training log
        log_frame = tk.LabelFrame(
            right_frame,
            text=" Training Log ",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b',
            relief=tk.GROOVE
        )
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=('Consolas', 9),
            bg='#1e1e1e',
            fg='#cccccc',
            insertbackground='white',
            width=35,
            height=15
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Best tracking
        self.best_mAP50 = 0
        self.best_mAP50_95 = 0
        self.best_epoch = 0
        self.elapsed_time = 0.0
        self.remaining_time = 0.0
    
    def refresh_runs_table(self):
        """Scan runs folders and populate the table with stats from results.csv"""
        import csv
        runs_root = os.path.join(os.path.dirname(__file__), 'NEU-DET-with-yolov8', 'runs')
        if not os.path.exists(runs_root):
            return
        run_dirs = [d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
        self.runs_table.delete(*self.runs_table.get_children())
        for run in run_dirs:
            results_path = os.path.join(runs_root, run, 'results.csv')
            args_path = os.path.join(runs_root, run, 'args.yaml')
            best_map50 = best_map95 = epochs = '-'
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        reader = list(csv.DictReader(f))
                        if reader:
                            epochs = len(reader)
                            # Handle CSV with leading spaces in column names
                            map50_values = []
                            map95_values = []
                            for row in reader:
                                # Find the mAP columns by looking for the pattern
                                for key, value in row.items():
                                    key = key.strip()
                                    if key == 'metrics/mAP50(B)' and value.strip():
                                        try:
                                            map50_values.append(float(value.strip()))
                                        except ValueError:
                                            pass
                                    elif key == 'metrics/mAP50-95(B)' and value.strip():
                                        try:
                                            map95_values.append(float(value.strip()))
                                        except ValueError:
                                            pass
                            if map50_values:
                                best_map50 = max(map50_values)
                            if map95_values:
                                best_map95 = max(map95_values)
                except Exception as e:
                    print(f"Error reading {results_path}: {e}")
            args_str = ''
            if os.path.exists(args_path):
                try:
                    with open(args_path, 'r') as f:
                        for i, line in enumerate(f):
                            if i > 8: break
                            args_str += line.strip() + ' '
                except Exception:
                    pass
            self.runs_table.insert('', 'end', values=(run, epochs, f"{best_map50:.4f}" if isinstance(best_map50, (int, float)) else '-', f"{best_map95:.4f}" if isinstance(best_map95, (int, float)) else '-', args_str.strip()))

    def on_run_double_click(self, event):
        """Show details of the selected run in a popup"""
        item = self.runs_table.selection()
        if not item:
            return
        vals = self.runs_table.item(item[0], 'values')
        run_name = vals[0]
        runs_root = os.path.join(os.path.dirname(__file__), 'NEU-DET-with-yolov8', 'runs')
        results_path = os.path.join(runs_root, run_name, 'results.csv')
        args_path = os.path.join(runs_root, run_name, 'args.yaml')
        msg = f"Run: {run_name}\n\n"
        if os.path.exists(args_path):
            msg += f"Args (first lines):\n"
            with open(args_path, 'r') as f:
                for i, line in enumerate(f):
                    if i > 12: break
                    msg += line
            msg += '\n'
        if os.path.exists(results_path):
            msg += f"Results (last 5 epochs):\n"
            import csv
            with open(results_path, 'r') as f:
                reader = list(csv.DictReader(f))
                for row in reader[-5:]:
                    # Find the correct column names (may have leading spaces)
                    map50_val = map95_val = box_val = 'N/A'
                    for key, value in row.items():
                        key = key.strip()
                        if key == 'metrics/mAP50(B)':
                            map50_val = value.strip()
                        elif key == 'metrics/mAP50-95(B)':
                            map95_val = value.strip()
                        elif key == 'train/box_loss':
                            box_val = value.strip()
                    msg += f"Epoch {row.get('epoch', 'N/A').strip()}: mAP50={map50_val} mAP50-95={map95_val} Box={box_val}\n"
        messagebox.showinfo(f"Run Details: {run_name}", msg)
    
    def set_defaults_for_model(self, event=None):
        """Set best defaults for each model for RTX 4060"""
        model = self.model_var.get()
        # These values are based on typical RTX 4060 8GB VRAM, batch size can be increased if VRAM allows
        defaults = {
            'ShuffleAttention':   {'batch': 16, 'epochs': 100, 'imgsz': 640},
            'CoordAtt':           {'batch': 16, 'epochs': 100, 'imgsz': 640},
            'Swin-1 Transformer': {'batch': 8,  'epochs': 100, 'imgsz': 640},
            'Swin-3 Transformer': {'batch': 8,  'epochs': 100, 'imgsz': 640},
            'Plain YOLOv8n':     {'batch': 32, 'epochs': 100, 'imgsz': 640},
        }
        d = defaults.get(model, {'batch': 8, 'epochs': 100, 'imgsz': 640})
        self.batch_var.set(d['batch'])
        self.epochs_var.set(d['epochs'])
        self.imgsz_var.set(d['imgsz'])
        # Update widgets
        self.batch_spin.delete(0, 'end')
        self.batch_spin.insert(0, d['batch'])
        self.epochs_spin.delete(0, 'end')
        self.epochs_spin.insert(0, d['epochs'])
        self.imgsz_combo.set(d['imgsz'])
        self.log(f"Set defaults for {model}: batch={d['batch']}, epochs={d['epochs']}, imgsz={d['imgsz']}")
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def format_duration(self, seconds):
        """Format seconds as H:MM:SS"""
        if seconds is None or seconds < 0:
            return "-"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"

    def poll_results_csv(self):
        """Poll results.csv for latest validation metrics as a fallback."""
        if not self.training or not self.current_save_dir:
            return
        now = time.time()
        if now - self.last_csv_check < 1.0:
            return
        self.last_csv_check = now

        csv_path = os.path.join(self.current_save_dir, 'results.csv')
        if not os.path.exists(csv_path):
            return

        try:
            import csv
            with open(csv_path, 'r', newline='') as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return

            last_row = rows[-1]
            epoch_val = None
            mAP50 = mAP50_95 = precision = recall = 0.0
            for key, value in last_row.items():
                key_stripped = key.strip()
                value_stripped = value.strip() if isinstance(value, str) else value
                if key_stripped == 'epoch':
                    try:
                        epoch_val = int(float(value_stripped))
                    except (ValueError, TypeError):
                        epoch_val = None
                elif key_stripped == 'metrics/mAP50(B)':
                    mAP50 = float(value_stripped)
                elif key_stripped == 'metrics/mAP50-95(B)':
                    mAP50_95 = float(value_stripped)
                elif key_stripped == 'metrics/precision(B)':
                    precision = float(value_stripped)
                elif key_stripped == 'metrics/recall(B)':
                    recall = float(value_stripped)

            if epoch_val is None:
                epoch_val = len(rows)
            if epoch_val <= self.last_val_epoch:
                return

            if mAP50 > 0 or mAP50_95 > 0:
                self.last_val_epoch = epoch_val
                self.mAP50_history.append(mAP50)
                self.mAP50_95_history.append(mAP50_95)
                self.stats_labels['mAP50'].config(text=f"{mAP50:.4f}")
                self.stats_labels['mAP50-95'].config(text=f"{mAP50_95:.4f}")
                self.stats_labels['Precision'].config(text=f"{precision:.4f}")
                self.stats_labels['Recall'].config(text=f"{recall:.4f}")

                if mAP50 > self.best_mAP50:
                    self.best_mAP50 = mAP50
                    self.best_epoch = epoch_val or len(self.mAP50_history)
                    self.best_labels['Best mAP50'].config(text=f"{mAP50:.4f}")
                    self.best_labels['Best Epoch'].config(text=str(self.best_epoch))

                if mAP50_95 > self.best_mAP50_95:
                    self.best_mAP50_95 = mAP50_95
                    self.best_labels['Best mAP50-95'].config(text=f"{mAP50_95:.4f}")

                self.update_plots()
        except Exception:
            return
    
    def update_plots(self):
        """Update all plots with current data"""
        # Clear and redraw loss plot
        self.ax_loss.clear()
        self.ax_loss.set_facecolor('#1e1e1e')
        self.ax_loss.set_title('Training Loss', color='white', fontsize=10)
        if self.epochs:
            self.ax_loss.plot(self.epochs, self.box_losses, 'r-', label='Box', linewidth=1.5)
            self.ax_loss.plot(self.epochs, self.cls_losses, 'g-', label='Cls', linewidth=1.5)
            self.ax_loss.plot(self.epochs, self.dfl_losses, 'b-', label='DFL', linewidth=1.5)
            self.ax_loss.legend(loc='upper right', fontsize=8, facecolor='#2b2b2b', labelcolor='white')
        self.ax_loss.tick_params(colors='gray')
        for spine in self.ax_loss.spines.values():
            spine.set_color('gray')
        
        # mAP plot
        self.ax_map.clear()
        self.ax_map.set_facecolor('#1e1e1e')
        self.ax_map.set_title('Validation mAP', color='white', fontsize=10)
        if self.mAP50_history:
            epochs_val = list(range(1, len(self.mAP50_history) + 1))
            self.ax_map.plot(epochs_val, self.mAP50_history, 'c-', label='mAP50', linewidth=2)
            self.ax_map.plot(epochs_val, self.mAP50_95_history, 'm-', label='mAP50-95', linewidth=2)
            self.ax_map.legend(loc='lower right', fontsize=8, facecolor='#2b2b2b', labelcolor='white')
            self.ax_map.set_ylim(0, 1)
        self.ax_map.tick_params(colors='gray')
        for spine in self.ax_map.spines.values():
            spine.set_color('gray')
        
        # Box loss detail
        self.ax_box.clear()
        self.ax_box.set_facecolor('#1e1e1e')
        self.ax_box.set_title('Box Loss', color='white', fontsize=10)
        if self.epochs:
            self.ax_box.fill_between(self.epochs, self.box_losses, alpha=0.3, color='red')
            self.ax_box.plot(self.epochs, self.box_losses, 'r-', linewidth=1)
        self.ax_box.tick_params(colors='gray')
        for spine in self.ax_box.spines.values():
            spine.set_color('gray')
        
        # Class loss detail
        self.ax_cls.clear()
        self.ax_cls.set_facecolor('#1e1e1e')
        self.ax_cls.set_title('Class Loss', color='white', fontsize=10)
        if self.epochs:
            self.ax_cls.fill_between(self.epochs, self.cls_losses, alpha=0.3, color='green')
            self.ax_cls.plot(self.epochs, self.cls_losses, 'g-', linewidth=1)
        self.ax_cls.tick_params(colors='gray')
        for spine in self.ax_cls.spines.values():
            spine.set_color('gray')
        
        self.canvas.draw()
    
    def check_queue(self):
        """Check for messages from training thread"""
        try:
            while True:
                msg = self.metrics_queue.get_nowait()
                
                if msg['type'] == 'train_start':
                    self.total_batches = msg.get('total_batches', 0)
                    self.current_save_dir = msg.get('save_dir', '')
                    self.batch_label.config(text=f"Batch: 0/{self.total_batches}")
                
                elif msg['type'] == 'batch':
                    batch = msg['batch']
                    total_batches = msg['total_batches']
                    epoch = msg['epoch']
                    epochs = msg['epochs']
                    
                    # Update batch progress bar
                    if total_batches > 0:
                        batch_progress = (batch / total_batches) * 100
                        self.batch_progress_var.set(batch_progress)
                    
                    self.batch_label.config(text=f"Epoch {epoch}: Batch {batch}/{total_batches}")
                
                elif msg['type'] == 'epoch':
                    epoch = msg['epoch']
                    total = msg['epochs']
                    
                    self.epochs.append(epoch)
                    self.box_losses.append(msg['box_loss'])
                    self.cls_losses.append(msg['cls_loss'])
                    self.dfl_losses.append(msg['dfl_loss'])
                    
                    # Update stats
                    self.stats_labels['Epoch'].config(text=f"{epoch}/{total}")
                    self.stats_labels['Box Loss'].config(text=f"{msg['box_loss']:.4f}")
                    self.stats_labels['Cls Loss'].config(text=f"{msg['cls_loss']:.4f}")
                    self.stats_labels['DFL Loss'].config(text=f"{msg['dfl_loss']:.4f}")
                    
                    # Update overall progress
                    progress = (epoch / total) * 100
                    self.progress_var.set(progress)
                    self.progress_label.config(text=f"Epoch {epoch}/{total} ({progress:.1f}%)")
                    
                    # Reset batch progress for next epoch
                    self.batch_progress_var.set(0)
                    
                    self.log(f"Epoch {epoch}/{total} - box:{msg['box_loss']:.3f} cls:{msg['cls_loss']:.3f}")
                    self.update_plots()
                
                elif msg['type'] == 'validation':
                    mAP50 = msg['mAP50']
                    mAP50_95 = msg['mAP50_95']
                    
                    self.mAP50_history.append(mAP50)
                    self.mAP50_95_history.append(mAP50_95)
                    
                    self.stats_labels['mAP50'].config(text=f"{mAP50:.4f}")
                    self.stats_labels['mAP50-95'].config(text=f"{mAP50_95:.4f}")
                    self.stats_labels['Precision'].config(text=f"{msg['precision']:.4f}")
                    self.stats_labels['Recall'].config(text=f"{msg['recall']:.4f}")

                    self.last_val_epoch = len(self.mAP50_history)
                    
                    # Track best
                    if mAP50 > self.best_mAP50:
                        self.best_mAP50 = mAP50
                        self.best_epoch = len(self.mAP50_history)
                        self.best_labels['Best mAP50'].config(text=f"{mAP50:.4f}")
                        self.best_labels['Best Epoch'].config(text=str(self.best_epoch))
                    
                    if mAP50_95 > self.best_mAP50_95:
                        self.best_mAP50_95 = mAP50_95
                        self.best_labels['Best mAP50-95'].config(text=f"{mAP50_95:.4f}")
                    
                    self.log(f"Val: mAP50={mAP50:.3f} mAP50-95={mAP50_95:.3f}")
                    self.update_plots()

                elif msg['type'] == 'time':
                    self.elapsed_time = msg.get('elapsed', 0)
                    self.remaining_time = msg.get('remaining', 0)
                    self.stats_labels['Time Elapsed'].config(text=self.format_duration(self.elapsed_time))
                    self.stats_labels['Time Remaining'].config(text=self.format_duration(self.remaining_time))
                
                elif msg['type'] == 'complete':
                    self.training = False
                    self.train_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.progress_label.config(text="Training Complete!")
                    self.log(f"Training complete! Model saved to {msg['save_dir']}")
                    
                    # Show results dialog
                    self.show_results_dialog(msg['save_dir'])
                
                elif msg['type'] == 'error':
                    self.training = False
                    self.train_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.log(f"Error: {msg['message']}")
                    messagebox.showerror("Training Error", msg['message'])
                
                elif msg['type'] == 'log':
                    self.log(msg['message'])
                    
        except queue.Empty:
            pass

        self.poll_results_csv()
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def show_results_dialog(self, save_dir):
        """Show final results in a dialog"""
        results = f"""
Training Complete!

Model: {self.model_var.get()}
Epochs: {self.epochs_var.get()}

FINAL RESULTS:
═══════════════════════════════
Best mAP50:     {self.best_mAP50:.4f} ({self.best_mAP50*100:.1f}%)
Best mAP50-95:  {self.best_mAP50_95:.4f} ({self.best_mAP50_95*100:.1f}%)
Best Epoch:     {self.best_epoch}
═══════════════════════════════

Model saved to:
{save_dir}/weights/best.pt
"""
        messagebox.showinfo("Training Results", results)
    
    def get_model_yaml(self):
        """Get the model YAML path based on selection"""
        model_map = {
            'ShuffleAttention': 'NEU-DET-with-yolov8/ultralytics/cfg/models/v8/yolov8_ShuffleAttention.yaml',
            'CoordAtt': 'NEU-DET-with-yolov8/ultralytics/cfg/models/v8/yolov8_CoordAtt.yaml',
            'Swin-1 Transformer': 'NEU-DET-with-yolov8/ultralytics/cfg/models/v8/yolov8_one_swinTrans.yaml',
            'Swin-3 Transformer': 'NEU-DET-with-yolov8/ultralytics/cfg/models/v8/yolov8_three_swinTrans.yaml',
            'Plain YOLOv8n': 'yolov8n.yaml',
        }
        return model_map.get(self.model_var.get(), 'yolov8n.yaml')
    
    def training_worker(self):
        """Training worker thread"""
        try:
            from ultralytics import YOLO
            from ultralytics.utils import callbacks
            
            model_yaml = self.get_model_yaml()
            model_name = self.model_var.get().lower().replace(' ', '_').replace('-', '_')
            
            self.metrics_queue.put({'type': 'log', 'message': f'Loading model: {model_yaml}'})
            
            model = YOLO(model_yaml)
            
            # Add custom callbacks
            callback = TrainingCallback(self.metrics_queue, self.epochs_var.get(), self.batch_var.get())
            model.add_callback('on_train_start', callback.on_train_start)
            model.add_callback('on_train_batch_end', callback.on_train_batch_end)
            model.add_callback('on_train_epoch_end', callback.on_train_epoch_end)
            model.add_callback('on_val_end', callback.on_val_end)
            model.add_callback('on_train_end', callback.on_train_end)
            
            self.metrics_queue.put({'type': 'log', 'message': 'Starting training...'})
            
            # Start training
            model.train(
                data='NEU-DET-with-yolov8/data/data.yaml',
                epochs=self.epochs_var.get(),
                batch=self.batch_var.get(),
                imgsz=self.imgsz_var.get(),
                device=0 if torch.cuda.is_available() else 'cpu',
                workers=0,  # Windows compatibility
                project='NEU-DET-with-yolov8/runs',
                name=f'{model_name}_train',
                exist_ok=False,
                amp=False,
                verbose=False,
            )
            
        except Exception as e:
            self.metrics_queue.put({'type': 'error', 'message': str(e)})
    
    def start_training(self):
        """Start training"""
        if self.training:
            return
        
        # Reset metrics
        self.epochs = []
        self.box_losses = []
        self.cls_losses = []
        self.dfl_losses = []
        self.mAP50_history = []
        self.mAP50_95_history = []
        self.best_mAP50 = 0
        self.best_mAP50_95 = 0
        self.best_epoch = 0
        self.elapsed_time = 0.0
        self.remaining_time = 0.0
        self.current_save_dir = ''
        self.last_csv_check = 0.0
        self.last_val_epoch = 0
        
        # Reset UI
        for label in self.stats_labels.values():
            label.config(text='-')
        for label in self.best_labels.values():
            label.config(text='-')
        self.progress_var.set(0)
        self.batch_progress_var.set(0)
        self.batch_label.config(text="Batch: 0/0")
        self.log_text.delete(1.0, tk.END)
        
        self.training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.log(f"Starting {self.model_var.get()} training...")
        self.log(f"Epochs: {self.epochs_var.get()}, Batch: {self.batch_var.get()}, ImgSize: {self.imgsz_var.get()}")
        
        # Start training thread
        self.training_thread = threading.Thread(target=self.training_worker, daemon=True)
        self.training_thread.start()
    
    def stop_training(self):
        """Stop training (not fully implemented - would need to interrupt YOLO)"""
        if messagebox.askyesno("Stop Training", "Are you sure you want to stop training?\nThis will terminate the process."):
            self.log("Training stopped by user")
            self.training = False
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            # Note: Properly stopping YOLO training would require more complex handling


def main():
    root = tk.Tk()
    
    # Configure ttk style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TProgressbar', thickness=20, background='#4a9eff')
    style.configure('Batch.Horizontal.TProgressbar', thickness=15, background='#28a745')
    
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
