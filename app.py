import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

Processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")


def select_image():
    global img_pil, img_tk

    file_path = filedialog.askopenfilename(title="Select your Image", filetypes=[
                                           ("image Files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        return
    img_pil = Image.open(file_path)

    img_pil = img_pil.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_pil)
    label_img.config(image=img_tk)
    label_img.image = img_tk
    button_genai.config(state=tk.NORMAL)


def generate_caption():
    try:
        inputs = Processor(img_pil, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)
        caption = Processor.decode(out[0], skip_special_tokens=True)
        text_caption.delete(1.0, tk.END)
        text_caption.insert(tk.END, caption)

    except Exception as ex:
        messagebox.showerror("Error", str(ex))


root = tk.Tk()
root.title("AI Caption")
root.geometry("600x800")


label_title = tk.Label(root, text="AI Caption")
label_title.pack(pady=5)

button_select = tk.Button(root, text="Select Image",
                          command=select_image, width=20)
button_select.pack(pady=5)

label_img = tk.Label(root)
label_img.pack(pady=10)

button_genai = tk.Button(root, text="Generate Caption",
                         command=generate_caption, width=20, height=2, state=tk.DISABLED)
button_genai.pack(pady=5)
text_caption = tk.Text(root, height=4, padx=10, pady=10,)
text_caption.pack()


root.mainloop()