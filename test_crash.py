import os
import torch
import customtkinter as ctk
import threading
import time

app = ctk.CTk()
app.geometry("200x200")

def load():
    print("Starting load thread")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    spec = "./finetuned_model/"
    processor = WhisperProcessor.from_pretrained(spec)
    print("Loaded processor")
    model = WhisperForConditionalGeneration.from_pretrained(spec)
    print("Loaded model weights")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}")
    model.to(device)
    print("Moved to device")
    app.after(0, lambda: print("After callback executed!"))

threading.Thread(target=load, daemon=True).start()

app.mainloop()
