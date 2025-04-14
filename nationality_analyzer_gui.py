import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame, Canvas, Text, Scrollbar
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATHS = {
    'nationality': 'saved_models/nationality_model.keras',
    'emotion': 'saved_models/emotion_model.keras',
    'age': 'saved_models/age_model.keras',
    'dress_color': 'saved_models/dress_color_model.keras',
}

IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3

NATIONALITY_CLASSES = ['African', 'Indian', 'Other', 'US']
EMOTION_CLASSES = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
AGE_CLASSES = ['0-12', '13-19', '20-29', '30-39', '40-49', '50-60', '61+']
DRESS_COLOR_CLASSES = ['Black', 'Blue', 'Green', 'Other', 'Red', 'White', 'Yellow']

models = {}
print("Loading trained models...")
all_models_loaded = True
for task, path in MODEL_PATHS.items():
    try:
        if os.path.exists(path):
            models[task] = tf.keras.models.load_model(path)
            print(f"Loaded {task} model successfully.")
        else:
            print(f"Warning: Model file not found for {task} at {path}. This task will be unavailable.")
            models[task] = None
            if task in ['nationality', 'emotion']:
                 all_models_loaded = False
    except Exception as e:
        print(f"Error loading {task} model from {path}: {e}")
        models[task] = None
        if task in ['nationality', 'emotion']:
            all_models_loaded = False

if not all_models_loaded:
     print("ERROR: Failed to load one or more critical models (nationality/emotion).")

def predict_task(model, image_batch, class_list):
    if model is None or image_batch is None:
        return "Model N/A", 0.0
    try:
        predictions = model.predict(image_batch, verbose=0)
        score = np.max(predictions[0])
        pred_index = np.argmax(predictions[0])
        confidence = score * 100
        if 0 <= pred_index < len(class_list):
            pred_class = class_list[pred_index]
            return pred_class, confidence
        else:
             return "Index Error", confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction Error", 0.0

class NationalityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nationality Analyzer")
        self.root.geometry("1000x700")
        self.image_path = None
        self.original_image = None
        self.display_image_tk = None

        self.control_frame = Frame(root, pady=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.load_button = Button(self.control_frame, text="Load Image", command=self.load_image, width=15)
        self.load_button.pack(side=tk.LEFT, padx=20, pady=5)

        self.analyze_button = Button(self.control_frame, text="Analyze Image", command=self.analyze_image, state=tk.DISABLED, width=15)
        self.analyze_button.pack(side=tk.LEFT, padx=20, pady=5)

        self.status_label = Label(self.control_frame, text="Load an image to begin.")
        self.status_label.pack(side=tk.LEFT, padx=20)

        self.main_frame = Frame(root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = Canvas(self.main_frame, bg="lightgrey", width=500)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.results_frame = Frame(self.main_frame, width=400)
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.results_label = Label(self.results_frame, text="Analysis Results", font=('Arial', 14, 'bold'))
        self.results_label.pack(pady=(0, 5))
        self.results_text = Text(self.results_frame, wrap=tk.WORD, height=20, width=50, state=tk.DISABLED, font=('Arial', 11))
        self.scrollbar = Scrollbar(self.results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.root.update()
        self.root.update_idletasks()
        self._display_image_on_canvas(None, "Load image for analysis")

    def _display_image_on_canvas(self, image_cv, message):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 2 or canvas_height < 2:
            canvas_width = 480
            canvas_height = 640

        if image_cv is None:
            self.canvas.create_text(canvas_width / 2, canvas_height / 2, text=message, anchor=tk.CENTER, width=canvas_width-20)
            return

        try:
            img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
            self.display_image_tk = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=self.display_image_tk)
            self.canvas.image = self.display_image_tk
        except Exception as e:
             print(f"Error displaying image: {e}")
             self.canvas.create_text(canvas_width / 2, canvas_height / 2, text="Error Displaying Image", anchor=tk.CENTER)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Person Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        self.image_path = path
        try:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                raise ValueError("Could not read image file.")
            self._display_image_on_canvas(self.original_image, "")
            self.analyze_button.config(state=tk.NORMAL if all_models_loaded else tk.DISABLED)
            self.status_label.config(text=f"Loaded: {os.path.basename(path)}")
            self._update_results("Results will appear here after analysis.")
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Failed to load image:\n{e}")
            self.image_path = None
            self.original_image = None
            self.analyze_button.config(state=tk.DISABLED)
            self.status_label.config(text="Error loading image.")
            self._display_image_on_canvas(None, "Error loading image")
            self._update_results("")

    def _update_results(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)

    def analyze_image(self):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        if not all_models_loaded:
             messagebox.showerror("Model Error", "Cannot analyze: Nationality or Emotion model failed to load.")
             return
        self.status_label.config(text="Analyzing...")
        self._update_results("Processing...\n")
        self.root.update_idletasks()
        try:
            img_resized = cv2.resize(self.original_image, (IMG_WIDTH, IMG_HEIGHT))
            if CHANNELS == 1:
                if len(img_resized.shape) == 3:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                img_resized = np.expand_dims(img_resized, axis=-1)
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            nationality, nat_conf = predict_task(models.get('nationality'), img_batch, NATIONALITY_CLASSES)
            emotion, emo_conf = predict_task(models.get('emotion'), img_batch, EMOTION_CLASSES)

            results_str = f"--- Analysis Results ---\n\n"
            results_str += f"Predicted Nationality: {nationality} ({nat_conf:.1f}%)\n"
            results_str += f"Predicted Emotion: {emotion} ({emo_conf:.1f}%)\n"

            age_str = "N/A"
            color_str = "N/A"

            if nationality == 'Indian':
                age, age_conf = predict_task(models.get('age'), img_batch, AGE_CLASSES)
                dress_color, color_conf = predict_task(models.get('dress_color'), img_batch, DRESS_COLOR_CLASSES)
                age_str = f"{age} ({age_conf:.1f}%)"
                color_str = f"{dress_color} ({color_conf:.1f}%)"
                results_str += f"Predicted Age Range: {age_str}\n"
                results_str += f"Predicted Dress Color: {color_str}\n"

            elif nationality == 'US':
                age, age_conf = predict_task(models.get('age'), img_batch, AGE_CLASSES)
                age_str = f"{age} ({age_conf:.1f}%)"
                results_str += f"Predicted Age Range: {age_str}\n"

            self._update_results(results_str)
            self.status_label.config(text="Analysis complete.")
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze image:\n{e}")
            self.status_label.config(text="Error during analysis.")
            self._update_results("")

if __name__ == "__main__":
    root = tk.Tk()
    app = NationalityApp(root)
    root.mainloop()
