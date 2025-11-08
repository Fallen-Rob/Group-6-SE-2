import customtkinter as ctk
from tkinter import filedialog

# Set the theme (light, dark, or system)
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

# Create main window
app = ctk.CTk()
app.title("Ai-nspect")
app.geometry("1000x1000")

# Gradient background (simulated using a frame color)
bg_frame = ctk.CTkFrame(app, fg_color=("lightblue", "#274ad4"), corner_radius=20)
bg_frame.pack(fill="both", expand=True, padx=10, pady=10)

# --- Dropdown Menu ---
options = ["Model ver. 1.0", "Model ver. 1.0", "Model ver. 1.0", "Add New Model +"]
selected_option = ctk.StringVar(value=options[0])

dropdown = ctk.CTkOptionMenu(
    bg_frame,
    variable=selected_option,
    values=options,
    width=200,
    corner_radius=10
)
dropdown.pack(pady=(30, 10))

# --- View History Button ---
history_btn = ctk.CTkButton(
    bg_frame,
    text="View History",
    width=100,
    height=50,
    corner_radius=10
)
history_btn.place(x=20, y=20)

# --- Select Image Button ---
def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
    )
    if file_path:
        print("Selected:", file_path)

select_btn = ctk.CTkButton(
    bg_frame,
    text="Select Image",
    font=ctk.CTkFont(size=18, weight="bold"),
    width=180,
    height=50,
    corner_radius=20,
    command=select_image
)
select_btn.pack(pady=200)

app.mainloop()
