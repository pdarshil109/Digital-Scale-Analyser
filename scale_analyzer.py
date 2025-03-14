import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

class ImagePreviewWindow:
    def __init__(self, parent, image, callback=None):
        self.top = tk.Toplevel(parent)
        self.top.title("Image Preview & Calibration")
        self.top.geometry("1000x700")
        self.top.minsize(800, 600)
        
        # Store the original image and working copy
        self.original_image = image.copy()
        self.working_image = image.copy()
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.brightness = 0
        self.contrast = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.crop_mode = False
        self.crop_start = None
        self.crop_rect = None
        self.calibration_mode = None  # '0' or '10' or None
        
        # Create main frame
        main_frame = ttk.Frame(self.top)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Create top toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # Add Fit to Window button at the start
        ttk.Button(toolbar, text="Fit to Window", command=self.fit_to_window).pack(side=tk.LEFT, padx=2)
        
        # Add separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Add zoom controls
        ttk.Button(toolbar, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        
        # Add separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Add crop controls
        ttk.Button(toolbar, text="Start Crop", command=self.start_crop).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Apply Crop", command=self.apply_crop).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Cancel Crop", command=self.cancel_crop).pack(side=tk.LEFT, padx=2)
        
        # Add separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Add calibration buttons
        ttk.Button(toolbar, text="Set 0 Point", 
                  command=lambda: self.set_calibration_point('0')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Set 10 Point", 
                  command=lambda: self.set_calibration_point('10')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear Calibration", 
                  command=self.clear_calibration).pack(side=tk.LEFT, padx=2)
        
        # Add separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Add action buttons
        ttk.Button(toolbar, text="Apply & Save", 
                  command=self.apply_and_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Reset All", 
                  command=self.reset_all).pack(side=tk.LEFT, padx=2)
        
        # Create canvas with scrollbars
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(expand=True, fill=tk.BOTH)
        
        self.canvas = tk.Canvas(canvas_frame)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.config(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        # Grid layout for canvas and scrollbars
        self.canvas.grid(row=0, column=0, sticky='nsew')
        h_scroll.grid(row=1, column=0, sticky='ew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)
        
        # Create control panel
        control_frame = ttk.LabelFrame(main_frame, text="Image Adjustments", padding="5")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Rotation control
        ttk.Label(control_frame, text="Rotation:").grid(row=0, column=0, padx=5)
        self.rotation_var = tk.DoubleVar(value=0)
        rotation_scale = ttk.Scale(control_frame, from_=-180, to=180, 
                                 variable=self.rotation_var, orient=tk.HORIZONTAL,
                                 command=self.on_rotation_change)
        rotation_scale.grid(row=0, column=1, sticky='ew', padx=5)
        
        # Brightness control
        ttk.Label(control_frame, text="Brightness:").grid(row=1, column=0, padx=5)
        self.brightness_var = tk.DoubleVar(value=0)
        brightness_scale = ttk.Scale(control_frame, from_=-50, to=50, 
                                   variable=self.brightness_var, orient=tk.HORIZONTAL,
                                   command=self.on_brightness_change)
        brightness_scale.grid(row=1, column=1, sticky='ew', padx=5)
        
        # Contrast control
        ttk.Label(control_frame, text="Contrast:").grid(row=2, column=0, padx=5)
        self.contrast_var = tk.DoubleVar(value=1)
        contrast_scale = ttk.Scale(control_frame, from_=0.5, to=2.0, 
                                 variable=self.contrast_var, orient=tk.HORIZONTAL,
                                 command=self.on_contrast_change)
        contrast_scale.grid(row=2, column=1, sticky='ew', padx=5)
        
        # Status label for calibration
        self.status_label = ttk.Label(control_frame, text="Click on the image to set calibration points")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Initialize calibration points
        self.calibration_points = {'0': None, '10': None}
        
        # Bind mouse events
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        
        # Bind window resize event
        self.top.bind('<Configure>', self.on_window_resize)
        
        # Display initial image
        self.update_image()
        
    def on_window_resize(self, event):
        if event.widget == self.top:
            self.fit_to_window()
            
    def fit_to_window(self):
        # Calculate zoom factor to fit window
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_height, image_width = self.working_image.shape[:2]
        
        width_ratio = canvas_width / image_width
        height_ratio = canvas_height / image_height
        self.zoom_factor = min(width_ratio, height_ratio) * 0.95  # 95% of fit
        self.update_image()
        
    def on_rotation_change(self, *args):
        self.rotation_angle = self.rotation_var.get()
        self.update_image()
        
    def on_brightness_change(self, *args):
        self.brightness = self.brightness_var.get()
        self.update_image()
        
    def on_contrast_change(self, *args):
        self.contrast = self.contrast_var.get()
        self.update_image()
        
    def update_image(self):
        # Apply rotation
        center = (self.working_image.shape[1] // 2, self.working_image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
        rotated = cv2.warpAffine(self.original_image, rotation_matrix, 
                                (self.working_image.shape[1], self.working_image.shape[0]))
        
        # Apply brightness and contrast
        adjusted = cv2.convertScaleAbs(rotated, alpha=self.contrast, 
                                     beta=self.brightness)
        
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Apply zoom
        new_size = (
            int(image_pil.size[0] * self.zoom_factor),
            int(image_pil.size[1] * self.zoom_factor)
        )
        resized_image = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(resized_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.config(width=min(new_size[0], 1200), height=min(new_size[1], 800))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Redraw calibration points
        self.draw_calibration_points()
        
    def set_calibration_point(self, value):
        self.calibration_mode = value
        self.status_label.config(text=f"Click on the {value} point in the image")
        self.canvas.config(cursor="crosshair")
        
    def on_mouse_down(self, event):
        if self.calibration_mode is not None:
            # Convert canvas coordinates to image coordinates
            img_x = int(event.x / self.zoom_factor)
            img_y = int(event.y / self.zoom_factor)
            self.calibration_points[self.calibration_mode] = (img_x, img_y)
            self.draw_calibration_points()
            self.calibration_mode = None
            self.canvas.config(cursor="")
            self.status_label.config(text="Click on the image to set calibration points")
        elif self.crop_mode:
            self.crop_start = (event.x, event.y)
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
        else:
            self.canvas.scan_mark(event.x, event.y)
    
    def on_mouse_drag(self, event):
        if self.crop_mode and self.crop_start:
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
            self.crop_rect = self.canvas.create_rectangle(
                self.crop_start[0], self.crop_start[1], event.x, event.y,
                outline='red', width=2
            )
        else:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            
    def on_mouse_up(self, event):
        if self.crop_mode and self.crop_start:
            self.crop_end = (event.x, event.y)
    
    def start_crop(self):
        self.crop_mode = True
        self.canvas.config(cursor="crosshair")
        self.status_label.config(text="Click and drag to select crop area")
        
    def apply_crop(self):
        if hasattr(self, 'crop_start') and hasattr(self, 'crop_end'):
            # Convert canvas coordinates to image coordinates
            x1 = min(self.crop_start[0], self.crop_end[0]) / self.zoom_factor
            y1 = min(self.crop_start[1], self.crop_end[1]) / self.zoom_factor
            x2 = max(self.crop_start[0], self.crop_end[0]) / self.zoom_factor
            y2 = max(self.crop_start[1], self.crop_end[1]) / self.zoom_factor
            
            # Apply crop
            self.working_image = self.working_image[int(y1):int(y2), int(x1):int(x2)]
            self.original_image = self.original_image[int(y1):int(y2), int(x1):int(x2)]
            
            # Reset crop mode and update
            self.cancel_crop()
            self.fit_to_window()
            self.status_label.config(text="Click on the image to set calibration points")
            
    def cancel_crop(self):
        self.crop_mode = False
        self.canvas.config(cursor="")
        if hasattr(self, 'crop_rect') and self.crop_rect:
            self.canvas.delete(self.crop_rect)
        self.crop_start = None
        self.crop_end = None
        self.crop_rect = None
        self.status_label.config(text="Click on the image to set calibration points")
        
    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.update_image()
        
    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.update_image()
        
    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_image()
        
    def draw_calibration_points(self):
        # Remove old calibration points
        self.canvas.delete("calibration")
        
        # Draw new calibration points
        for value, point in self.calibration_points.items():
            if point is not None:
                x, y = point
                # Convert image coordinates to canvas coordinates
                canvas_x = x * self.zoom_factor
                canvas_y = y * self.zoom_factor
                
                # Draw point and label
                self.canvas.create_oval(
                    canvas_x-5, canvas_y-5, canvas_x+5, canvas_y+5,
                    fill='red', tags="calibration"
                )
                self.canvas.create_text(
                    canvas_x+10, canvas_y, text=value,
                    fill='red', anchor='w', tags="calibration"
                )

    def clear_calibration(self):
        self.calibration_points = {'0': None, '10': None}
        self.calibration_mode = None
        self.canvas.config(cursor="")
        self.status_label.config(text="Click on the image to set calibration points")
        self.update_image()
        
    def reset_all(self):
        self.rotation_var.set(0)
        self.brightness_var.set(0)
        self.contrast_var.set(1)
        self.zoom_factor = 1.0
        self.clear_calibration()
        self.working_image = self.original_image.copy()
        self.update_image()
        
    def apply_and_save(self):
        # Create processed image with current settings
        processed_image = self.get_processed_image()
        
        # Get current settings
        settings = {
            'rotation': self.rotation_angle,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'calibration_points': self.calibration_points
        }
        
        if self.callback:
            self.callback(processed_image, settings)
        
        # Minimize window instead of destroying it
        self.top.iconify()
        
    def get_processed_image(self):
        # Apply current rotation and adjustments
        center = (self.working_image.shape[1] // 2, self.working_image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
        rotated = cv2.warpAffine(self.original_image, rotation_matrix, 
                                (self.working_image.shape[1], self.working_image.shape[0]))
        return cv2.convertScaleAbs(rotated, alpha=self.contrast, beta=self.brightness)
        
    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        self.update_image()

class ScaleAnalyzer:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Digital Scale Analyzer")
        self.window.geometry("1000x800")
        
        # Store images and results
        self.images = []  # List to store image data
        self.data_file = "scale_data.json"
        self.load_saved_data()  # Load previously saved data
        self.setup_gui()
        
    def load_saved_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    saved_data = json.load(f)
                    
                for item in saved_data:
                    if os.path.exists(item['path']):  # Only load if image file exists
                        image = cv2.imread(item['path'])
                        if image is not None:
                            item['image'] = image
                            self.images.append(item)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load saved data: {str(e)}")
                
    def save_data(self):
        try:
            # Create a copy of image data without the actual images
            save_data = []
            for item in self.images:
                save_item = item.copy()
                save_item.pop('image', None)  # Remove the image data
                # Include settings in saved data
                if 'settings' in item:
                    save_item['settings'] = item['settings']
                save_data.append(save_item)
                
            with open(self.data_file, 'w') as f:
                json.dump(save_data, f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")

    def setup_gui(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.window, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create left panel for controls
        left_panel = ttk.Frame(self.main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Add buttons frame
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=0, column=0, columnspan=3, pady=5)
        
        # Add buttons
        self.select_btn = ttk.Button(button_frame, text="Add Image", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=2)
        
        self.camera_btn = ttk.Button(button_frame, text="Capture from Camera", command=self.capture_from_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=2)
        
        self.remove_btn = ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected)
        self.remove_btn.pack(side=tk.LEFT, padx=2)
        
        self.analyze_btn = ttk.Button(button_frame, text="Analyze Selected", command=self.analyze_selected)
        self.analyze_btn.pack(side=tk.LEFT, padx=2)
        
        # Create image list
        list_frame = ttk.Frame(left_panel)
        list_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview for images
        columns = ("Filename", "Time Added", "Scale Value")
        self.image_list = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        
        # Set column headings
        for col in columns:
            self.image_list.heading(col, text=col)
            self.image_list.column(col, width=150)
        
        self.image_list.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_list.yview)
        self.image_list.config(yscrollcommand=scrollbar.set)
        
        # Bind selection event
        self.image_list.bind('<<TreeviewSelect>>', self.on_select)
        
        # Bind double-click event for preview
        self.image_list.bind('<Double-1>', self.show_preview)
        
        # Create right panel for image display
        right_panel = ttk.Frame(self.main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Image display label
        self.image_label = ttk.Label(right_panel)
        self.image_label.grid(row=0, column=0, pady=10)
        
        # Result label
        self.result_label = ttk.Label(right_panel, text="Scale Value: --")
        self.result_label.grid(row=1, column=0, pady=10)
    
    def select_image(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        for file_path in file_paths:
            if file_path:
                # Load image
                image = cv2.imread(file_path)
                if image is not None:
                    # Store image data with fresh settings
                    filename = os.path.basename(file_path)
                    time_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Create fresh image data without any previous settings
                    image_data = {
                        "path": file_path,
                        "image": image,
                        "filename": filename,
                        "time_added": time_added,
                        "scale_value": "--",
                        "settings": {
                            'rotation': 0,
                            'brightness': 0,
                            'contrast': 1.0,
                            'calibration_points': {'0': None, '10': None}
                        }
                    }
                    
                    self.images.append(image_data)
                    
                    # Add to list
                    self.image_list.insert("", "end", values=(filename, time_added, "--"))
                    
                    # Display first image if it's the only one
                    if len(self.images) == 1:
                        self.display_image(image)
                        
                    # Save the fresh state
                    self.save_data()
    
    def remove_selected(self):
        selected_items = self.image_list.selection()
        if not selected_items:
            messagebox.showinfo("Info", "Please select an image to remove")
            return
            
        for item in selected_items:
            index = self.image_list.index(item)
            self.images.pop(index)
            self.image_list.delete(item)
            
        # Clear display if no images left
        if not self.images:
            self.clear_display()
    
    def clear_display(self):
        self.image_label.configure(image="")
        self.result_label.configure(text="Scale Value: --")
    
    def on_select(self, event):
        selected_items = self.image_list.selection()
        if selected_items:
            index = self.image_list.index(selected_items[0])
            if 0 <= index < len(self.images):
                image_data = self.images[index]
                self.display_image(image_data["image"])
                self.result_label.configure(text=f"Scale Value: {image_data['scale_value']}")
    
    def display_image(self, image):
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Resize to fit display
        display_size = (400, 300)
        image_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image_pil)
        self.image_label.configure(image=self.photo)
    
    def reset_settings(self, image_data):
        """Reset all settings for an image to default values"""
        image_data["settings"] = {
            'rotation': 0,
            'brightness': 0,
            'contrast': 1.0,
            'calibration_points': {'0': None, '10': None}
        }
        image_data["scale_value"] = "--"
        self.save_data()
        
    def analyze_selected(self):
        selected_items = self.image_list.selection()
        if not selected_items:
            messagebox.showinfo("Info", "Please select an image to analyze")
            return
            
        for item in selected_items:
            index = self.image_list.index(item)
            image_data = self.images[index]
            
            # Reset settings before analysis
            self.reset_settings(image_data)
            
            # Analyze image
            scale_value = self.analyze_image(image_data["image"])
            
            # Check if the image is invalid
            if scale_value == "NULL":
                messagebox.showerror("Invalid Image", "The selected image does not appear to be a scale reading.\n\nPlease ensure:\n- The image shows a clear scale display\n- The blue/white boundary is visible\n- The image is properly focused")
                continue
            
            # Update stored data
            image_data["scale_value"] = scale_value
            
            # Update list view
            self.image_list.set(item, "Scale Value", f"{scale_value:.1f}")
            
            # Update result label if this is the currently displayed image
            if self.image_list.selection() and self.image_list.selection()[0] == item:
                self.result_label.configure(text=f"Scale Value: {scale_value:.1f}")
                
            # Save the changes
            self.save_data()
    
    def analyze_image(self, image, scale_points=None):
        try:
            # First, validate that this looks like a scale image
            def is_valid_scale_image(img):
                # Convert to HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Define blue range for scale - even more permissive
                lower_blue = np.array([90, 30, 30])  # Even more permissive
                upper_blue = np.array([150, 255, 255])  # Wider range
                
                # Get blue mask
                blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                
                # Clean up mask
                kernel = np.ones((5,5), np.uint8)
                blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
                blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Sort contours by area
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Check if we have at least one significant contour
                if not contours or cv2.contourArea(contours[0]) < 100:
                    return False
                    
                # Count significant blue regions
                significant_regions = 0
                largest_area = cv2.contourArea(contours[0])
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > largest_area * 0.1:  # Consider regions at least 10% of the largest
                        significant_regions += 1
                
                # Accept if we have 1-3 significant blue regions
                return 1 <= significant_regions <= 3
            
            # Validate image before processing
            if not is_valid_scale_image(image):
                return "NULL"
                
            # Convert to HSV for blue detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Use more permissive blue range
            lower_blue = np.array([90, 30, 30])
            upper_blue = np.array([150, 255, 255])
            
            # Create blue mask
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Clean up mask
            kernel = np.ones((5,5), np.uint8)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Find the center-most significant region
            height, width = image.shape[:2]
            image_center_x = width // 2
            
            center_region = None
            min_distance = float('inf')
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Minimum area threshold
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        distance = abs(cx - image_center_x)
                        if distance < min_distance:
                            min_distance = distance
                            center_region = contour
            
            if center_region is None:
                return "NULL"
            
            # Get bounding box of center region
            x, y, w, h = cv2.boundingRect(center_region)
            
            # Add margins
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(width, x + w + margin_x)
            y2 = min(height, y + h + margin_y)
            
            # Extract the region
            cropped = image[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # Calculate intensity profile
            intensity_profile = np.mean(gray, axis=1)
            
            # Smooth the profile
            intensity_profile = cv2.GaussianBlur(intensity_profile.reshape(-1, 1), (5, 1), 0).flatten()
            
            # Find the transition point
            gradient = np.gradient(intensity_profile)
            gradient = cv2.GaussianBlur(gradient.reshape(-1, 1), (5, 1), 0).flatten()
            
            # Find the strongest transition point
            max_gradient_idx = np.argmax(np.abs(gradient))
            
            # Calculate relative position (inverted since 0 is at bottom)
            total_height = len(intensity_profile)
            relative_position = 1.0 - (max_gradient_idx / total_height)
            
            # Convert to Brix value (0-10 scale)
            scale_value = 10 * relative_position
            
            # Validate the result
            if not (0 <= scale_value <= 10):
                return "NULL"
            
            # Save debug images
            debug_image = cropped.copy()
            cv2.line(debug_image, (0, max_gradient_idx), (debug_image.shape[1], max_gradient_idx), (0, 255, 0), 2)
            cv2.imwrite('debug_analysis.jpg', debug_image)
            cv2.imwrite('debug_mask.jpg', blue_mask)
            
            # Create and save intensity profile visualization
            plt_height = 200
            plt_width = 400
            intensity_plot = np.zeros((plt_height, plt_width), dtype=np.uint8)
            for i in range(len(intensity_profile)-1):
                y1 = int((1-intensity_profile[i]/255) * (plt_height-1))
                y2 = int((1-intensity_profile[i+1]/255) * (plt_height-1))
                x1 = int(i * (plt_width-1) / len(intensity_profile))
                x2 = int((i+1) * (plt_width-1) / len(intensity_profile))
                cv2.line(intensity_plot, (x1, y1), (x2, y2), 255, 1)
            cv2.imwrite('debug_intensity.jpg', intensity_plot)
            
            return round(scale_value, 1)
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return "NULL"
    
    def show_preview(self, event):
        selected_items = self.image_list.selection()
        if selected_items:
            index = self.image_list.index(selected_items[0])
            if 0 <= index < len(self.images):
                image_data = self.images[index]
                preview_window = ImagePreviewWindow(self.window, image_data["image"], 
                                                 callback=self.on_preview_complete)
                
                # Reset to default settings first
                preview_window.rotation_var.set(0)
                preview_window.brightness_var.set(0)
                preview_window.contrast_var.set(1.0)
                preview_window.calibration_points = {'0': None, '10': None}
                
                # Only apply saved settings if they exist AND user hasn't selected new image
                if "settings" in image_data and image_data.get("is_new", False) is False:
                    settings = image_data["settings"]
                    preview_window.rotation_var.set(settings['rotation'])
                    preview_window.brightness_var.set(settings['brightness'])
                    preview_window.contrast_var.set(settings['contrast'])
                    preview_window.calibration_points = settings['calibration_points']
                
                preview_window.update_image()
    
    def on_preview_complete(self, processed_image, settings):
        # Update the current selection with new settings
        selected_items = self.image_list.selection()
        if selected_items:
            index = self.image_list.index(selected_items[0])
            image_data = self.images[index]
            
            # Update stored data
            image_data["image"] = processed_image
            image_data["settings"] = settings  # Save settings for future use
            
            # Update display
            self.display_image(processed_image)
            
            # Save changes
            self.save_data()
            
            # Bring main window to front
            self.window.lift()
            self.window.focus_force()
            
    def capture_from_camera(self):
        """Open camera window for capturing image"""
        try:
            # Simple camera initialization
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return

            # Set basic properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Create simple camera window
            self.camera_window = tk.Toplevel(self.window)
            self.camera_window.title("Camera")
            self.camera_window.geometry("700x600")
            
            # Create simple label for camera feed
            self.camera_label = ttk.Label(self.camera_window)
            self.camera_label.pack(pady=10)
            
            # Add buttons
            btn_frame = ttk.Frame(self.camera_window)
            btn_frame.pack(pady=5)
            
            ttk.Button(btn_frame, text="Capture", command=self.take_photo).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Close", command=self.close_camera).pack(side=tk.LEFT, padx=5)
            
            # Start camera feed
            self.camera_active = True
            self.update_camera()
            
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {str(e)}")
            self.close_camera()

    def update_camera(self):
        """Update camera feed"""
        if not hasattr(self, 'cap') or not self.cap.isOpened() or not self.camera_active:
            return

        try:
            ret, frame = self.cap.read()
            if ret:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL and resize
                image = Image.fromarray(frame_rgb)
                image = image.resize((640, 480))
                
                # Convert to PhotoImage
                self.camera_photo = ImageTk.PhotoImage(image)
                self.camera_label.configure(image=self.camera_photo)
            
            # Update every 100ms (10 FPS - much safer)
            if self.camera_active and hasattr(self, 'camera_window'):
                self.camera_window.after(100, self.update_camera)
                
        except Exception as e:
            print(f"Camera update error: {str(e)}")
            self.close_camera()

    def take_photo(self):
        """Capture photo from camera"""
        try:
            if hasattr(self, 'cap') and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Create captures directory
                    if not os.path.exists('captures'):
                        os.makedirs('captures')
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}.jpg"
                    filepath = os.path.join('captures', filename)
                    cv2.imwrite(filepath, frame)
                    
                    # Add to image list
                    image_data = {
                        "path": filepath,
                        "image": frame,
                        "filename": filename,
                        "time_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "scale_value": "--"
                    }
                    
                    self.images.append(image_data)
                    self.image_list.insert("", "end", values=(filename, image_data["time_added"], "--"))
                    
                    # Close camera
                    self.close_camera()
                    messagebox.showinfo("Success", "Image captured successfully!")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture: {str(e)}")
            self.close_camera()

    def close_camera(self):
        """Safely close camera"""
        try:
            self.camera_active = False
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
            if hasattr(self, 'camera_window') and self.camera_window is not None:
                self.camera_window.destroy()
                self.camera_window = None
        except Exception as e:
            print(f"Error closing camera: {str(e)}")
            
    def run(self):
        # Load initial images into the list
        for image_data in self.images:
            self.image_list.insert("", "end", values=(
                image_data["filename"],
                image_data["time_added"],
                image_data["scale_value"] if image_data["scale_value"] != "--" else "--"
            ))
            
        # Display first image if available
        if self.images:
            self.display_image(self.images[0]["image"])
            self.result_label.configure(text=f"Scale Value: {self.images[0]['scale_value']}")
            
        self.window.mainloop()
        
    def __del__(self):
        # Clean up camera if still open
        self.close_camera()
        # Save data when the application closes
        self.save_data()

if __name__ == "__main__":
    app = ScaleAnalyzer()
    app.run() 