import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim

# استدعاء ملفات الـ Pipeline بتاعتك
from pipeline.data import run_data_pipeline
from pipeline.preprocessing import run_preprocessing_pipeline
from pipeline.model import run_model_pipeline
from member_1 import TOKENIZER

CHECKPOINT_PATH = "las_checkpoint.pth"

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LAS Model Trainer - SpecAugment ASR")
        self.root.geometry("600x500")
        
        # متغيرات التحكم في مسار التدريب
        self.is_training = False
        self.is_paused = False
        self.training_thread = None
        
        self.setup_ui()

    def setup_ui(self):
        # --- الأزرار ---
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.btn_start = tk.Button(btn_frame, text="▶ Start Training", bg="green", fg="white", font=("Arial", 12, "bold"), command=self.start_training)
        self.btn_start.grid(row=0, column=0, padx=5)
        
        self.btn_pause = tk.Button(btn_frame, text="⏸ Pause", bg="orange", font=("Arial", 12, "bold"), command=self.pause_training, state=tk.DISABLED)
        self.btn_pause.grid(row=0, column=1, padx=5)
        
        self.btn_resume = tk.Button(btn_frame, text="🔄 Resume from Checkpoint", bg="blue", fg="white", font=("Arial", 12, "bold"), command=self.resume_training)
        self.btn_resume.grid(row=0, column=2, padx=5)
        
        self.btn_stop = tk.Button(btn_frame, text="⏹ Stop & Save", bg="red", fg="white", font=("Arial", 12, "bold"), command=self.stop_training, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=3, padx=5)

        # --- شاشة عرض الـ Logs ---
        self.log_area = scrolledtext.ScrolledText(self.root, width=70, height=20, font=("Consolas", 10))
        self.log_area.pack(pady=10)
        self.log("Ready to train...")

    def log(self, message):
        """دالة لكتابة الأحداث في شاشة الواجهة"""
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def start_training(self):
        if self.is_training: return
        self.is_training = True
        self.is_paused = False
        self.update_buttons_state()
        
        self.log("🚀 Starting NEW training pipeline...")
        # تشغيل التدريب في مسار منفصل عشان الواجهة متهنجش
        self.training_thread = threading.Thread(target=self.training_loop_worker, args=(False,))
        self.training_thread.start()

    def resume_training(self):
        if not os.path.exists(CHECKPOINT_PATH):
            messagebox.showerror("Error", "No checkpoint found to resume from!")
            return
            
        if self.is_training:
            # لو هو شغال بس معموله Pause
            self.is_paused = False
            self.log("▶ Resumed current training.")
        else:
            # لو التطبيق كان مقفول واتفتح من جديد
            self.is_training = True
            self.is_paused = False
            self.log("🔄 Loading checkpoint to resume training...")
            self.training_thread = threading.Thread(target=self.training_loop_worker, args=(True,))
            self.training_thread.start()
            
        self.update_buttons_state()

    def pause_training(self):
        if self.is_training and not self.is_paused:
            self.is_paused = True
            self.log("⏸ Training Paused. Press Resume to continue.")
            self.update_buttons_state()

    def stop_training(self):
        if self.is_training:
            self.is_training = False
            self.is_paused = False
            self.log("⏹ Stopping training safely and saving checkpoint...")
            self.update_buttons_state()

    def update_buttons_state(self):
        if self.is_training:
            if self.is_paused:
                self.btn_start.config(state=tk.DISABLED)
                self.btn_pause.config(state=tk.DISABLED)
                self.btn_resume.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.NORMAL)
            else:
                self.btn_start.config(state=tk.DISABLED)
                self.btn_pause.config(state=tk.NORMAL)
                self.btn_resume.config(state=tk.DISABLED)
                self.btn_stop.config(state=tk.NORMAL)
        else:
            self.btn_start.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_resume.config(state=tk.NORMAL if os.path.exists(CHECKPOINT_PATH) else tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)
    def training_loop_worker(self, load_checkpoint=False):
            """ده العقل المدبر اللي بيشتغل في الخلفية"""
            try:
                # 1. تجهيز الداتا
                self.log("Loading Data & Preprocessing...")
                raw_data = run_data_pipeline()
                # هنا بنستخدم كل الداتا مش عينة صغيرة
                processed_data = run_preprocessing_pipeline(raw_data)
                
                nc = processed_data.get("num_classes")
                seq_len = processed_data.get("target_sequence_length")
                train_loader = processed_data["train_loader"]
                
                # 2. بناء الموديل
                self.log("Building Model...")
                model = run_model_pipeline(num_classes=nc, target_sequence_length=seq_len)
                device = next(model.parameters()).device
                
                # Label Smoothing لمنع الحفظ الأعمى
                criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1) 
                optimizer = optim.AdamW(model.parameters(), lr=1e-3)
                
                start_epoch = 1
                max_epochs = 150
                
                # 3. استرجاع الـ Checkpoint لو مطلوب
                if load_checkpoint and os.path.exists(CHECKPOINT_PATH):
                    checkpoint = torch.load(CHECKPOINT_PATH)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    self.log(f"✅ Checkpoint loaded! Resuming from Epoch {start_epoch}")
                
                model.train()
                
                # تهيئة الـ Early Stopping (هيستنى 10 لفات بدون تحسن عشان يوقف)
                early_stopping = EarlyStopping(patience=10, min_delta=0.01)
                
                # 4. لوب التدريب الفعلي
                for epoch in range(start_epoch, max_epochs + 1):
                    if not self.is_training:
                        break # اليوزر داس Stop
                    
                    # فحص الـ Pause
                    while self.is_paused:
                        time.sleep(1)
                        if not self.is_training: break
                    
                    total_loss = 0
                    for xb, yb in train_loader:
                        # فحص الـ Pause جوه اللوب الداخلي برضه
                        while self.is_paused: time.sleep(1)
                        if not self.is_training: break

                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        
                        logits = model(xb, target_seq=yb)
                        loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
                        
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    if not self.is_training: break # تأكيد الخروج
                    
                    avg_loss = total_loss / len(train_loader)
                    self.log(f"Epoch {epoch}/{max_epochs} | Loss = {avg_loss:.4f}")
                    
                    # إرسال الخسارة (Loss) للـ Early Stopping عشان يقرر
                    early_stopping(avg_loss)
                    
                    # حفظ الموديل فقط لو كان ده أفضل أداء وصلناله (بدون Overfitting)
                    if early_stopping.counter == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_loss,
                        }, CHECKPOINT_PATH)
                        self.log("💾 New Best Model Saved!")
                    else:
                        self.log(f"⚠️ No improvement. Early Stopping Counter: {early_stopping.counter}/{early_stopping.patience}")
                    
                    # ضرب فرامل لو الموديل بطل يتعلم
                    if early_stopping.early_stop:
                        self.log(f"🛑 Early Stopping Triggered at Epoch {epoch}! Training Halted.")
                        break # كسر اللوب وإيقاف التدريب نهائياً
                    
                self.log("🏁 Training session ended.")
                
            except Exception as e:
                self.log(f"❌ Error during training: {str(e)}")
            finally:
                self.is_training = False
                self.is_paused = False
                # نحدث الواجهة من الـ Main Thread باستخدام after
                self.root.after(0, self.update_buttons_state)
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        patience: عدد اللفات اللي هيستناها بدون تحسن قبل ما يوقف
        min_delta: أقل مقدار نعتبره "تحسن" حقيقي
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            # لو مفيش تحسن، زود العداد
            self.counter += 1
            print(f"⚠️ Early Stopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # لو في تحسن، صفر العداد واحفظ الرقم الجديد
            self.best_loss = current_loss
            self.counter = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()