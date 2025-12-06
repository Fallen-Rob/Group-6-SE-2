import os
import json
import datetime
import threading
import customtkinter as ctk  # type: ignore
import numpy as np
import torch
import onnxruntime as ort  # type: ignore
import sys
import traceback
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import AutoImageProcessor
from history import HistoryPage
from model_run import run_gradcam
from model_run import run_onnx

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
        self.bg_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)
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
            text="VIEW HISTORY",
            fg_color="white",
            hover_color="#E5E5E5",
            text_color="black",
            font=ctk.CTkFont(family="Verdana", size=18, weight="bold"),
            width=180,
            height=40,
            corner_radius=20,
            border_color="black",
            border_width=3,
            command=self.show_history
        )
        view_history_btn.place(x=10, rely=0.5, anchor="w")

        # button for selecting image
        select_image_btn = ctk.CTkButton(
            self.main_frame,
            text="SELECT IMAGE",
            fg_color="white",
            hover_color="#E5E5E5",
            text_color="black",
            width=180,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=18,
            border_color="black",
            border_width=3,
            command=self.select_image
        )
        select_image_btn.place(relx=0.5, rely=0.7, anchor="center")

        self.show_main()

     #  pang navigate
    def show_main(self):
        self.main_frame.tkraise()

    def show_history(self):
        self.history_page.load_history_data()
        self.history_page.tkraise()

    #  para pag nag exe di mag error
    def get_resource_path(self, relative_path):
        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    #  pang select ng image
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;")]
        )
        if not file_path:
            return

        #pang loading
        overlay = ctk.CTkFrame(self.bg_frame, corner_radius=0)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.configure(fg_color="#8EA8FF")

        box = ctk.CTkFrame(overlay, fg_color="white", corner_radius=20)
        box.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.4, relheight=0.3)

        ctk.CTkLabel(box,
                     text="🔍 Processing image…",
                     text_color="black",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(30, 10))

        progress = ctk.CTkProgressBar(box, mode="indeterminate", width=200)
        progress.pack(pady=(0, 20))
        progress.start()
        self.update_idletasks()

        # eto yung nag poprocess habang nasa loading 
        def process_image():
            try:
                file_name = os.path.basename(file_path)
                current_time = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")

                so = ort.SessionOptions()
                so.intra_op_num_threads = 4
                so.inter_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

                model_path = self.get_resource_path("./ai_model/model_ai-generated_opt.onnx")
                session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

                processor_path = self.get_resource_path("./ai_model")
                processor = AutoImageProcessor.from_pretrained(processor_path)

                img = Image.open(file_path).convert("RGB")
                inputs = processor(img, return_tensors="np")

                outputs = session.run(None, {"pixel_values": inputs["pixel_values"]})
                logits = torch.tensor(outputs[0])
                probs = torch.nn.functional.softmax(logits, dim=-1)
                labels = ["fake", "real"]
                label_id = torch.argmax(probs).item()

                confidence = float(probs[0][label_id] * 100)
                confidence_str = f"{confidence:.2f}%"
                authenticity = labels[label_id].capitalize()

                # save history
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
                with open("error_log.txt", "w", encoding="utf-8") as log:
                    log.write(traceback.format_exc())

            finally:
                overlay.destroy()
                self.display_result(file_path, authenticity, confidence_str)

        threading.Thread(target=process_image, daemon=True).start()

    
    #  result page
    def display_result(self, file_path, authenticity, confidence_str):
        result_frame = ctk.CTkFrame(self.bg_frame, corner_radius=20, fg_color="white")
        result_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.95, relheight=0.55)

        result_frame.grid_rowconfigure(0, weight=1)
        result_frame.grid_columnconfigure((0, 1), weight=1)

        # col1: preview nung image
        col1 = ctk.CTkFrame(result_frame, fg_color="white", corner_radius=15)
        col1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        ctk.CTkLabel(col1, text="Image", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 5))

        preview = Image.open(file_path)
        preview.thumbnail((250, 250))
        preview_tk = ImageTk.PhotoImage(preview)
        img_label = ctk.CTkLabel(col1, image=preview_tk, text="")
        img_label.image = preview_tk
        img_label.pack(pady=10)

        # col2: AI result  yung confidence etc
        col2 = ctk.CTkFrame(result_frame, fg_color="white", corner_radius=15)
        col2.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        ctk.CTkLabel(col2, text="AI Result", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 20))

        # result color whether ai or not
        color = "#00C853" if authenticity.lower() == "real" else "#FF5252"

        ctk.CTkLabel(
            col2,
            text=f"Authenticity: {authenticity}\nConfidence: {confidence_str}",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=color,
            justify="center"
        ).pack(pady=10)

        # explain button para mapunta sa explanation page
        explain_btn = ctk.CTkButton(
            result_frame,
            text="Explain",
            fg_color="green",
            hover_color="green",
            text_color="white",
            width=120,
            command=lambda: self.open_explain_overlay(file_path)
        )
        explain_btn.place(relx=0.7, rely=0.8, anchor="w")
        #close button sa result page
        close_btn = ctk.CTkButton(
            result_frame,
            text="Close",
            fg_color="#E53935",
            hover_color="#D32F2F",
            text_color="white",
            width=120,
            command=result_frame.destroy
        )
        close_btn.place(relx=0.7, rely=0.9, anchor="w")

        result_frame.lift()

    #  Explanation page 
    def open_explain_overlay(self, img_path):
        # para blue background
        overlay = ctk.CTkFrame(self.bg_frame, corner_radius=0)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.configure(fg_color="#8EA8FF")

        # loading screen pag nag po process yung heatmap para di mukang static para di awkward
        load_box = ctk.CTkFrame(overlay, fg_color="white", corner_radius=20)
        load_box.place(relx=0.5, rely=0.5, anchor="center",
                       relwidth=0.42, relheight=0.28)

        ctk.CTkLabel(load_box, text="🔍 Generating explanation…",
                     text_color="black",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(24, 8))

        load_pb = ctk.CTkProgressBar(load_box, mode="indeterminate", width=220)
        load_pb.pack(pady=(6, 12))
        load_pb.start()
        self.update_idletasks()

        # 3) background worker
        def worker():
            try:
                res = run_gradcam(img_path)
                if isinstance(res, tuple) and len(res) >= 2:
                    heatmap_path, explanation = res[0], res[1]
                else:
                    heatmap_path = os.path.join(os.getcwd(), "gradcam_overlay.png")
                    explanation = "Grad-CAM finished (no string returned)."
            except Exception as e:
                heatmap_path, explanation = None, f"Error generating explanation:\n{e}"
                
            # explanation page columns
            def show_panel():
                load_pb.stop()
                load_box.destroy()

                # para same size dun sa result page
                panel = ctk.CTkFrame(overlay, corner_radius=20, fg_color="white")
                panel.place(relx=0.5, rely=0.5, anchor="center",
                            relwidth=0.95, relheight=0.55)

                # yung columns sa explanation page
                panel.grid_columnconfigure(0, weight=25)   # image 
                panel.grid_columnconfigure(1, weight=75)   # text
                panel.grid_rowconfigure(1, weight=1)

                # titles  don sa explanation pede nyo paltan
                ctk.CTkLabel(panel, text="Image Heatmap",
                             font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, pady=(12, 0))
                ctk.CTkLabel(panel, text="Explanation",
                             font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=1, pady=(12, 0))

                # heatmap pic yung ginawa ng gradcam na pic 
                left = ctk.CTkFrame(panel, fg_color="white", corner_radius=15)
                left.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
                if heatmap_path and os.path.exists(heatmap_path):
                    try:
                        img = Image.open(heatmap_path)
                        img.thumbnail((280, 280))          
                        tkimg = ImageTk.PhotoImage(img)
                        lbl = ctk.CTkLabel(left, image=tkimg, text="")
                        lbl.image = tkimg
                        lbl.pack(expand=True)
                    except Exception as e:
                        ctk.CTkLabel(left, text=f"Error loading heatmap:\n{e}",
                                     text_color="red").pack(expand=True)
                else:
                    ctk.CTkLabel(left, text="Heatmap not found",
                                 text_color="red").pack(expand=True)

                # explanation yung text nung gradcam
                right = ctk.CTkFrame(panel, fg_color="white", corner_radius=15)
                right.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
                txt = ctk.CTkTextbox(right)
                txt.pack(fill="both", expand=True, padx=10, pady=10)
                txt.insert("0.0", explanation)
                txt.configure(state="disabled")

                # btn ng explanation page
                btn_frame = ctk.CTkFrame(panel, fg_color="transparent")
                btn_frame.place(relx=0.7, rely=0.8, anchor="center")

                back_btn = ctk.CTkButton(btn_frame, text="Back", width=120,
                                         fg_color="#3B82F6", hover_color="#2563EB",
                                         command=overlay.destroy)
                back_btn.pack(pady=(0, 8))          # Back on top

                close_btn = ctk.CTkButton(btn_frame, text="Close", width=120,
                                          fg_color="#E53935", hover_color="#D32F2F",
                                          command=lambda: (overlay.destroy(),
                                                           self.show_main()))
                close_btn.pack()                    # Close below

            self.after(0, show_panel)

        threading.Thread(target=worker, daemon=True).start()


#pang run
if __name__ == "__main__":
    app = MainPage()
    app.mainloop()
    