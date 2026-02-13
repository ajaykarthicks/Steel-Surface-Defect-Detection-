"""
NEU-DET Steel Surface Defect Detection GUI
Uses YOLOv8 with ShuffleAttention for optimal detection performance
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import sys

# Add the local ultralytics to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NEU-DET-with-yolov8'))

# Fix for PyTorch 2.6+ weights_only security change
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO
import numpy as np


class DefectDetectorGUI:
    """GUI Application for Steel Surface Defect Detection"""
    
    # NEU-DET class names
    CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    # Colors for each defect type (RGB)
    CLASS_COLORS = {
        'crazing': (255, 0, 0),           # Red
        'inclusion': (0, 255, 0),          # Green
        'patches': (0, 0, 255),            # Blue
        'pitted_surface': (255, 165, 0),   # Orange  
        'rolled-in_scale': (128, 0, 128),  # Purple
        'scratches': (255, 255, 0)         # Yellow
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("NEU-DET Steel Defect Detector (ShuffleAttention)")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2b2b2b')
        
        # Model paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.runs_root = os.path.join(script_dir, 'NEU-DET-with-yolov8', 'runs')
        self.available_runs = self.find_available_runs()
        self.run_var = tk.StringVar()
        self.model_path = None
        
        self.model = None
        self.current_image_path = None
        self.current_result = None
        self.original_image = None
        self.result_image = None
        
        self.setup_ui()
        self.refresh_run_options(select_default=True)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Steel Surface Defect Detection",
            font=('Helvetica', 20, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b'
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = tk.Label(
            main_frame,
            text="YOLOv8 + ShuffleAttention | 74.1% mAP50",
            font=('Helvetica', 10),
            fg='#888888',
            bg='#2b2b2b'
        )
        subtitle_label.pack(pady=(0, 10))
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#3c3c3c', relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        control_inner = tk.Frame(control_frame, bg='#3c3c3c')
        control_inner.pack(pady=10, padx=10)
        
        # Load Image button
        self.load_btn = tk.Button(
            control_inner,
            text="📂 Load Image",
            command=self.load_image,
            font=('Helvetica', 11, 'bold'),
            bg='#4a9eff',
            fg='white',
            activebackground='#3d8ce6',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Detect button
        self.detect_btn = tk.Button(
            control_inner,
            text="🔍 Detect Defects",
            command=self.detect_defects,
            font=('Helvetica', 11, 'bold'),
            bg='#28a745',
            fg='white',
            activebackground='#218838',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.save_btn = tk.Button(
            control_inner,
            text="💾 Save Result",
            command=self.save_result,
            font=('Helvetica', 11, 'bold'),
            bg='#6c757d',
            fg='white',
            activebackground='#5a6268',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Model run selection
        run_frame = tk.Frame(control_inner, bg='#3c3c3c')
        run_frame.pack(side=tk.LEFT, padx=15)

        tk.Label(
            run_frame,
            text="Model Run:",
            font=('Helvetica', 10),
            fg='#ffffff',
            bg='#3c3c3c'
        ).pack(side=tk.LEFT)

        self.run_combo = ttk.Combobox(
            run_frame,
            textvariable=self.run_var,
            values=self.available_runs,
            width=26,
            state='readonly'
        )
        self.run_combo.pack(side=tk.LEFT, padx=5)
        self.run_combo.bind('<<ComboboxSelected>>', self.on_run_select)

        self.refresh_runs_btn = tk.Button(
            run_frame,
            text="Refresh",
            command=self.refresh_run_options,
            font=('Helvetica', 9),
            bg='#6c757d',
            fg='white',
            activebackground='#5a6268',
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor='hand2'
        )
        self.refresh_runs_btn.pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold
        threshold_frame = tk.Frame(control_inner, bg='#3c3c3c')
        threshold_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(
            threshold_frame,
            text="Confidence:",
            font=('Helvetica', 10),
            fg='#ffffff',
            bg='#3c3c3c'
        ).pack(side=tk.LEFT)
        
        self.conf_var = tk.DoubleVar(value=0.25)
        self.conf_slider = ttk.Scale(
            threshold_frame,
            from_=0.1,
            to=0.9,
            variable=self.conf_var,
            orient=tk.HORIZONTAL,
            length=150,
            command=self.on_conf_change
        )
        self.conf_slider.pack(side=tk.LEFT, padx=5)
        
        self.conf_label = tk.Label(
            threshold_frame,
            text="25%",
            font=('Helvetica', 10),
            fg='#ffffff',
            bg='#3c3c3c',
            width=4
        )
        self.conf_label.pack(side=tk.LEFT)
        
        # Image display area
        image_container = tk.Frame(main_frame, bg='#2b2b2b')
        image_container.pack(fill=tk.BOTH, expand=True)
        
        # Input image frame
        input_frame = tk.LabelFrame(
            image_container,
            text=" Input Image ",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b',
            relief=tk.GROOVE,
            bd=2
        )
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.input_canvas = tk.Canvas(
            input_frame,
            bg='#1e1e1e',
            highlightthickness=0
        )
        self.input_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Output image frame
        output_frame = tk.LabelFrame(
            image_container,
            text=" Detection Result ",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b',
            relief=tk.GROOVE,
            bd=2
        )
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.output_canvas = tk.Canvas(
            output_frame,
            bg='#1e1e1e',
            highlightthickness=0
        )
        self.output_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results panel
        results_frame = tk.LabelFrame(
            main_frame,
            text=" Detection Summary ",
            font=('Helvetica', 11, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b',
            relief=tk.GROOVE,
            bd=2
        )
        results_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Legend and detection results
        legend_results_frame = tk.Frame(results_frame, bg='#2b2b2b')
        legend_results_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Legend
        legend_frame = tk.Frame(legend_results_frame, bg='#2b2b2b')
        legend_frame.pack(side=tk.LEFT)
        
        tk.Label(
            legend_frame,
            text="Defect Classes: ",
            font=('Helvetica', 10, 'bold'),
            fg='#ffffff',
            bg='#2b2b2b'
        ).pack(side=tk.LEFT)
        
        for cls_name, color in self.CLASS_COLORS.items():
            color_box = tk.Frame(legend_frame, bg=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}', width=15, height=15)
            color_box.pack(side=tk.LEFT, padx=(10, 2))
            color_box.pack_propagate(False)
            
            tk.Label(
                legend_frame,
                text=cls_name,
                font=('Helvetica', 9),
                fg='#cccccc',
                bg='#2b2b2b'
            ).pack(side=tk.LEFT)
        
        # Detection count
        self.result_label = tk.Label(
            legend_results_frame,
            text="No detections yet",
            font=('Helvetica', 11),
            fg='#888888',
            bg='#2b2b2b'
        )
        self.result_label.pack(side=tk.RIGHT, padx=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Loading model...")
        self.status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Helvetica', 9),
            fg='#888888',
            bg='#1e1e1e',
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Bind resize events
        self.input_canvas.bind('<Configure>', self.on_resize)
        self.output_canvas.bind('<Configure>', self.on_resize)
    
    def on_conf_change(self, value):
        """Handle confidence threshold change"""
        self.conf_label.config(text=f"{int(float(value) * 100)}%")
        # Re-run detection if we have results
        if self.current_result is not None and self.original_image is not None:
            self.update_detection_display()

    def find_available_runs(self):
        """Find run folders that contain weights/best.pt"""
        runs = []
        if not os.path.exists(self.runs_root):
            return runs
        for name in sorted(os.listdir(self.runs_root)):
            run_dir = os.path.join(self.runs_root, name)
            if not os.path.isdir(run_dir):
                continue
            best_path = os.path.join(run_dir, 'weights', 'best.pt')
            if os.path.exists(best_path):
                runs.append(name)
        return runs

    def refresh_run_options(self, select_default=False):
        """Refresh the run list and reload selected model if needed"""
        current = self.run_var.get()
        self.available_runs = self.find_available_runs()
        self.run_combo['values'] = self.available_runs

        if select_default:
            if 'shuffle_train' in self.available_runs:
                self.run_var.set('shuffle_train')
            elif self.available_runs:
                self.run_var.set(self.available_runs[0])
            else:
                self.run_var.set('')
        elif current in self.available_runs:
            self.run_var.set(current)
        elif self.available_runs:
            self.run_var.set(self.available_runs[0])
        else:
            self.run_var.set('')

        if self.run_var.get():
            self.on_run_select()
        else:
            self.model = None
            self.model_path = None
            self.status_var.set("No trained runs found")

    def on_run_select(self, event=None):
        """Handle selection of a run model"""
        run_name = self.run_var.get()
        if not run_name:
            return
        self.model_path = os.path.join(self.runs_root, run_name, 'weights', 'best.pt')
        self.load_model()
    
    def load_model(self):
        """Load the ShuffleAttention YOLOv8 model"""
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                messagebox.showerror(
                    "Model Not Found",
                    f"Model not found at:\n{self.model_path}\n\n"
                    "Please select a valid trained run."
                )
                self.status_var.set("Error: Model not found")
                return
            
            self.status_var.set(f"Loading model: {os.path.basename(os.path.dirname(self.model_path))}...")
            self.root.update()
            
            # Fix for PyTorch 2.6+ weights_only security change
            import torch
            torch.serialization.add_safe_globals([dict])
            
            self.model = YOLO(self.model_path)
            self.status_var.set(f"Model loaded: {os.path.basename(os.path.dirname(self.model_path))}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def load_image(self):
        """Load an image file"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes,
            initialdir=os.path.join(os.path.dirname(__file__), 'NEU-DET-with-yolov8', 'data', 'NEU-DET')
        )
        
        if filepath:
            try:
                self.current_image_path = filepath
                self.original_image = Image.open(filepath).convert('RGB')
                self.current_result = None
                self.result_image = None
                
                # Display input image
                self.display_image(self.original_image, self.input_canvas)
                
                # Clear output
                self.output_canvas.delete("all")
                
                # Enable detect button
                self.detect_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)
                
                self.result_label.config(text="Click 'Detect Defects' to analyze", fg='#888888')
                self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def detect_defects(self):
        """Run defect detection on the loaded image"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
        
        try:
            if not self.current_image_path:
                messagebox.showerror("Error", "No image selected")
                return
            
            self.status_var.set("Detecting defects...")
            self.root.update()
            
            # Run inference
            results = self.model.predict(
                self.current_image_path,
                conf=0.1,  # Low threshold, we'll filter in display
                verbose=False
            )
            
            self.current_result = results[0]
            self.update_detection_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed:\n{str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def update_detection_display(self):
        """Update the detection display with current confidence threshold"""
        if self.current_result is None or self.original_image is None:
            return
        
        conf_threshold = self.conf_var.get()
        
        # Create annotated image
        result_img = self.original_image.copy()
        draw = ImageDraw.Draw(result_img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Get detections
        boxes = self.current_result.boxes
        detection_counts = {cls: 0 for cls in self.CLASS_NAMES}
        total_detections = 0
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])
                
                # Apply confidence threshold
                if conf < conf_threshold:
                    continue
                
                cls_id = int(box.cls[0])
                cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"class_{cls_id}"
                
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get color for this class
                color = self.CLASS_COLORS.get(cls_name, (255, 255, 255))
                
                # Draw bounding box
                box_width = 3
                draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
                
                # Draw label background
                label = f"{cls_name} {conf:.2f}"
                bbox = draw.textbbox((x1, y1), label, font=small_font)
                label_height = bbox[3] - bbox[1] + 6
                label_width = bbox[2] - bbox[0] + 6
                
                # Semi-transparent background
                draw.rectangle(
                    [x1, y1 - label_height, x1 + label_width, y1],
                    fill=color
                )
                
                # Draw label text
                draw.text((x1 + 3, y1 - label_height + 2), label, fill='white', font=small_font)
                
                # Update counts
                detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                total_detections += 1
        
        self.result_image = result_img
        
        # Display result
        self.display_image(result_img, self.output_canvas)
        
        # Update result label
        if total_detections > 0:
            details = ", ".join([f"{name}: {count}" for name, count in detection_counts.items() if count > 0])
            self.result_label.config(
                text=f"Found {total_detections} defect(s) — {details}",
                fg='#28a745'
            )
        else:
            self.result_label.config(text="No defects detected at current confidence level", fg='#ffc107')
        
        self.save_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Detection complete: {total_detections} defect(s) found (conf ≥ {int(conf_threshold*100)}%)")
    
    def display_image(self, image, canvas):
        """Display an image on a canvas, scaled to fit"""
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Calculate scale to fit
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized)
        
        # Clear canvas and display
        canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        
        # Keep reference
        canvas.image = photo
    
    def on_resize(self, event):
        """Handle window resize"""
        if self.original_image is not None:
            self.display_image(self.original_image, self.input_canvas)
        if self.result_image is not None:
            self.display_image(self.result_image, self.output_canvas)
    
    def save_result(self):
        """Save the detection result image"""
        if self.result_image is None or not self.current_image_path:
            messagebox.showerror("Error", "No result to save!")
            return
        
        # Generate default filename
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        default_name = f"{base_name}_detected.jpg"
        
        filepath = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                self.result_image.save(filepath, quality=95)
                self.status_var.set(f"Saved: {filepath}")
                messagebox.showinfo("Success", f"Result saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save:\n{str(e)}")


def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set dark theme for ttk widgets
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TScale', background='#3c3c3c', troughcolor='#1e1e1e')
    
    app = DefectDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
