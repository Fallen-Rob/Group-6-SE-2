import os
import json
import datetime
import threading
import customtkinter as ctk  # type: ignore
import numpy as np
import torch
import onnxruntime as ort  # type: ignore
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import AutoImageProcessor
from history import HistoryPage
import sys


# ✅ allows correct file paths for both Python & PyInstaller EXE
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):  # running from EXE
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# theme or body color
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class MainPage(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI-nspect")
        self.geometry("900x600")
        self.resizable(False, False)

# Background 
        self.bg_frame = ctk.CTkFrame(self, corner_radius=20)
        self.bg_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.9)
        self.bg_frame.configure(fg_color="#8EA8FF")  # light blue background

# frame for main and history
        self.main_frame = ctk.CTkFrame(self.bg_frame, fg_color="transparent")
        self.history_page = HistoryPage(self.bg_frame, self.show_main)

        for frame in (self.main_frame, self.history_page):
            frame.place(relwidth=1, relheight=1)

# header
        header = ctk.CTkFrame(self.main_frame, fg_color="transparent", height=60, corner_radius=20)
        header.pack(fill="x", pady=(10, 20), padx=10)

# button for page switching to history page
        view_history_btn = ctk.CTkButton(
            header,
            text="View History",
            fg_color="white",
            hover_color="#E5E5E5",
            text_color="black",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=100,
            command=self.show_history
        )
        view_history_btn.place(x=10, rely=0.5, anchor="w")
        
# button for selecting image
        select_image_btn = ctk.CTkButton(
            self.main_frame,
            text="Select Image",
            width=180,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=15,
            command=self.select_image
        )
        select_image_btn.place(relx=0.5, rely=0.5, anchor="center")

        self.show_main()

# this is the function for switching page
    def show_main(self):
        self.main_frame.tkraise()

    def show_history(self):
        self.history_page.load_history_data()
        self.history_page.tkraise()

# select image from directory
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;")]
        )
        if not file_path:
            return

# animation for loading
        overlay = ctk.CTkFrame(self.bg_frame, fg_color=("gray10", "gray10"), corner_radius=0)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        box = ctk.CTkFrame(overlay, fg_color="white", corner_radius=20)
        box.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.4, relheight=0.3)

        title = ctk.CTkLabel(
            box,
            text="🔍 Processing image...",
            text_color="black",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(pady=(30, 10))

        progress = ctk.CTkProgressBar(box, mode="indeterminate", width=200)
        progress.pack(pady=(0, 20))
        progress.start()

        self.update_idletasks()

# animation 
        def fade_in_overlay(alpha=0.0):
            if alpha < 0.8:
                overlay.configure(fg_color=(f"gray{int(10 + alpha * 100)}", f"gray{int(10 + alpha * 100)}"))
                self.after(30, lambda: fade_in_overlay(alpha + 0.05))

        fade_in_overlay()

#this is where the ai process the image 
        def process_image():
            try:
                file_name = os.path.basename(file_path)
                current_time = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")

# ✅ updated: safe path for EXE
                onnx_path = resource_path("ai_detector_v2_optimized.onnx")

                so = ort.SessionOptions()
                so.intra_op_num_threads = 4
                so.inter_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
                session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

# ✅ updated: safe offline HuggingFace model path for EXE
                processor_path = resource_path("offline_model/models--google--vit-base-patch16-224/snapshots/config")
                processor = AutoImageProcessor.from_pretrained(processor_path)
                
                img = Image.open(file_path).convert("RGB")
                inputs = processor(img, return_tensors="np")

                outputs = session.run(None, {"pixel_values": inputs["pixel_values"]})
                logits = torch.tensor(outputs[0])
                probs = torch.nn.functional.softmax(logits, dim=-1)
                labels = ["real", "fake"]
                label_id = torch.argmax(probs).item()

                confidence = float(probs[0][label_id] * 100)
                confidence_str = f"{confidence:.2f}%"
                authenticity = labels[label_id].capitalize()
                
#to save in history.json file
                history = []
                if os.path.exists("history.json"):
                    with open("history.json", "r") as f:
                        try:
                            history = json.load(f)
                        except json.JSONDecodeError:
                            history = []

                history.append({
                    "File Name": file_name,
                    "Date": current_time,
                    "Authenticity": authenticity,
                    "Confidence": confidence_str
                })

                with open("history.json", "w") as f:
                    json.dump(history, f, indent=4)

            except Exception as e:
                authenticity = "Error"
                confidence_str = str(e)
                print("❌ Error processing image:", e)

            finally:
# animation for process
                def fade_out_overlay(alpha=0.8):
                    if alpha > 0:
                        overlay.configure(fg_color=(f"gray{int(10 + alpha * 100)}", f"gray{int(10 + alpha * 100)}"))
                        self.after(30, lambda: fade_out_overlay(alpha - 0.05))
                    else:
                        overlay.destroy()
                        self.display_result(file_path, authenticity, confidence_str)

                self.after(0, lambda: fade_out_overlay(0.8))

        threading.Thread(target=process_image, daemon=True).start()

# display result
    def display_result(self, file_path, authenticity, confidence_str):
        result_frame = ctk.CTkFrame(self.bg_frame, corner_radius=20)
        result_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=0.5)
        result_frame.attributes = {"alpha": 0.0}

# display image and result in left and right
        left = ctk.CTkFrame(result_frame, fg_color="white", corner_radius=15)
        left.place(relx=0.02, rely=0.5, anchor="w", relwidth=0.30, relheight=0.9)

# ✅ updated — new 3-column result section
        right = ctk.CTkFrame(result_frame, fg_color="white", corner_radius=15)
        right.place(relx=0.35, rely=0.5, anchor="w", relwidth=0.63, relheight=0.9)
        right.grid_columnconfigure((0, 1, 2), weight=1)

# to see preview of image
        try:
            preview = Image.open(file_path)
            preview.thumbnail((230, 230))
            preview_tk = ImageTk.PhotoImage(preview)
            image_label = ctk.CTkLabel(left, image=preview_tk, text="")
            image_label.image = preview_tk
            image_label.pack(expand=True)
        except Exception as e:
            ctk.CTkLabel(left, text=f"Error loading image:\n{e}", text_color="red").pack(pady=10)

# result color whether ai or not
        color = "#00C853" if authenticity.lower() == "real" else "#FF5252"

# whether its ai or not — COLUMN 1
        result_label = ctk.CTkLabel(
            right,
            text=f"Authenticity:\n{authenticity}",
            text_color=color,
            font=ctk.CTkFont(size=18, weight="bold"),
            justify="center"
        )
        result_label.grid(row=0, column=0, pady=(40, 10), padx=10)

# confidence on how sure the ai checker on authenticity — COLUMN 2
        confidence_label = ctk.CTkLabel(
            right,
            text=f"Confidence:\n{confidence_str}",
            text_color="black",
            font=ctk.CTkFont(size=18),
            justify="center"
        )
        confidence_label.grid(row=0, column=1, pady=(40, 10), padx=10)

# ✅ NEW — COLUMN 3 for future Deepfake AI model
        deepfake_placeholder = ctk.CTkLabel(
            right,
            text="Deepfake Result:\nPending",
            text_color="#0077CC",
            font=ctk.CTkFont(size=18, weight="bold"),
            justify="center"
        )
        deepfake_placeholder.grid(row=0, column=2, pady=(40, 10), padx=10)

# closing button for results
        close_btn = ctk.CTkButton(
            right,
            text="Close",
            fg_color="#E53935",
            hover_color="#D32F2F",
            text_color="white",
            width=100,
            command=lambda: result_frame.destroy()
        )
        close_btn.grid(row=1, column=1, pady=(20, 10))

        result_frame.lift()

# animation for process
        def fade_in(alpha=0.0):
            if alpha < 1.0:
                try:
                    result_frame.tk.call("tk", "scaling")
                    result_frame.update_idletasks()
                except:
                    pass
                result_frame.place(relx=0.5, rely=0.5, anchor="center")
                
                result_frame.attributes["alpha"] = alpha
                result_frame.configure(fg_color="blue")
                self.after(20, lambda: fade_in(alpha + 0.05))
            else:
                result_frame.attributes["alpha"] = 1.0

        fade_in()


# makes the app run
if __name__ == "__main__":
    app = MainPage()
    app.mainloop()
