import sys
import subprocess
import threading
import time
import os
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk


def _ensure_dependencies():
    """Install from requirements.txt if core ML stack is missing (best-effort bootstrap)."""
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import librosa  # noqa: F401
    except ImportError:
        req = Path(__file__).resolve().parent / "requirements.txt"
        print(f"[setup] Installing dependencies from {req} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


_ensure_dependencies()

import torch
from pipeline.data import run_data_pipeline
from pipeline.evaluation import run_evaluation_pipeline
from pipeline.model import run_model_pipeline
from pipeline.preprocessing import run_preprocessing_pipeline
from pipeline.training import run_training_pipeline


def _num_classes_from_loaders(data):
    if isinstance(data, dict) and data.get("num_classes") is not None:
        return int(data["num_classes"])
    return None


def _target_sequence_length_from_loaders(data):
    if isinstance(data, dict) and data.get("target_sequence_length") is not None:
        return int(data["target_sequence_length"])
    return None


def _apply_ctk_theme():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")


class TextRedirector:
    """توجيه print إلى مربع النص — يُجدول على الخيط الرئيسي (آمن مع Tk)."""

    def __init__(self, textbox, root_widget):
        self.textbox = textbox
        self.root = root_widget

    def write(self, text):
        def append():
            self.textbox.insert(ctk.END, text)
            self.textbox.see(ctk.END)

        try:
            self.root.after(0, append)
        except Exception:
            self.textbox.insert(ctk.END, text)
            self.textbox.see(ctk.END)

    def flush(self):
        pass


class SpecAugmentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        _apply_ctk_theme()

        self.title("SpecAugment — خط المعالجة والتعرف على الصوت (LAS Architecture)")
        self.geometry("1060x740")
        self._infer_model = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self._build_pipeline_tab()
        self._build_recognition_tab()

    def _build_pipeline_tab(self):
        pipe = self.tabview.add("خط المعالجة / Pipeline")
        pipe.grid_columnconfigure(1, weight=1)
        pipe.grid_rowconfigure(0, weight=1)

        self.sidebar_frame = ctk.CTkFrame(pipe, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="إعدادات التجربة",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 4), sticky="w")
        self.hint_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="python main.py\nثم شغّل الـ Pipeline أو جهّز الموديل من تبويب التعرف",
            font=ctk.CTkFont(size=12),
            text_color="gray70",
            justify="left",
        )
        self.hint_label.grid(row=1, column=0, padx=20, pady=(0, 12), sticky="w")

        self.policy_label = ctk.CTkLabel(self.sidebar_frame, text="سياسة SpecAugment:")
        self.policy_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")

        self.policy_optionemenu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=[
                "None",
                "LB (LibriSpeech Basic)",
                "LD (LibriSpeech Double)",
                "SM (Switchboard Mild)",
                "SS (Switchboard Strong)",
            ],
            command=self.change_policy,
        )
        self.policy_optionemenu.grid(row=3, column=0, padx=20, pady=(10, 10))

        self.w_label = ctk.CTkLabel(self.sidebar_frame, text="Time mask (W): 80")
        self.w_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")
        self.w_slider = ctk.CTkSlider(
            self.sidebar_frame,
            from_=0,
            to=100,
            number_of_steps=100,
            command=lambda v: self.w_label.configure(text=f"Time mask (W): {int(v)}"),
        )
        self.w_slider.grid(row=5, column=0, padx=20, pady=(5, 10))
        self.w_slider.set(80)

        self.f_label = ctk.CTkLabel(self.sidebar_frame, text="Freq mask (F): 27")
        self.f_label.grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
        self.f_slider = ctk.CTkSlider(
            self.sidebar_frame,
            from_=0,
            to=50,
            number_of_steps=50,
            command=lambda v: self.f_label.configure(text=f"Freq mask (F): {int(v)}"),
        )
        self.f_slider.grid(row=7, column=0, padx=20, pady=(5, 10))
        self.f_slider.set(27)

        self.run_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="تشغيل الـ Pipeline",
            command=self.start_pipeline_thread,
            fg_color="#28a745",
            hover_color="#218838",
        )
        self.run_btn.grid(row=9, column=0, padx=20, pady=20)

        self.main_frame = ctk.CTkFrame(pipe)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.console_label = ctk.CTkLabel(
            self.main_frame,
            text="سجل التشغيل (Console)",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.console_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        self.console_box = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.console_box.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        sys.stdout = TextRedirector(self.console_box, self)

    def _build_recognition_tab(self):
        rec = self.tabview.add("التعرف على الصوت")
        rec.grid_columnconfigure(0, weight=1)
        rec.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            rec,
            text="اختر ملف WAV (16 kHz يُفضّل — سيتم تحويل المونو وإعادة الاعتيان تلقائياً)",
            font=ctk.CTkFont(size=15),
        ).grid(row=0, column=0, padx=20, pady=(16, 8), sticky="w")

        btn_row = ctk.CTkFrame(rec, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="w", padx=16, pady=8)

        # الزر الجديد لتحميل الأوزان مباشرة بدون تدريب
        self.load_ckpt_btn = ctk.CTkButton(
            btn_row,
            text="📥 تحميل الأوزان (Checkpoint)",
            command=self.load_checkpoint,
            width=200,
            fg_color="#b8860b", # لون ذهبي مميز
            hover_color="#8b6508"
        )
        self.load_ckpt_btn.pack(side="left", padx=(0, 10))

        self.prepare_infer_btn = ctk.CTkButton(
            btn_row,
            text="⚙️ تدريب موديل جديد (من الصفر)",
            command=self.start_prepare_inference_thread,
            width=200,
            fg_color="#555555"
        )
        self.prepare_infer_btn.pack(side="left", padx=(0, 10))

        self.pick_wav_btn = ctk.CTkButton(
            btn_row,
            text="🎵 اختر ملف صوتي (.wav)",
            command=self.pick_wav_and_infer,
            width=200,
            fg_color="#1f538d",
        )
        self.pick_wav_btn.pack(side="left", padx=4)

        self.infer_result = ctk.CTkTextbox(
            rec,
            font=ctk.CTkFont(family="Segoe UI", size=16),
            wrap="word",
        )
        self.infer_result.grid(row=2, column=0, sticky="nsew", padx=20, pady=(8, 16))

        self.infer_status = ctk.CTkLabel(
            rec,
            text="الحالة: لم يُجهّز الموديل بعد — اضغط «تحميل الأوزان» للبدء فوراً.",
            text_color="gray70",
        )
        self.infer_status.grid(row=3, column=0, padx=20, pady=(0, 12), sticky="w")

    def change_policy(self, choice):
        if choice.startswith("LB") or choice.startswith("LD"):
            self.w_slider.set(80)
            self.f_slider.set(27)
        elif choice.startswith("SM"):
            self.w_slider.set(40)
            self.f_slider.set(15)
        elif choice.startswith("SS"):
            self.w_slider.set(40)
            self.f_slider.set(27)
        elif choice == "None":
            self.w_slider.set(0)
            self.f_slider.set(0)

        self.w_label.configure(text=f"Time mask (W): {int(self.w_slider.get())}")
        self.f_label.configure(text=f"Freq mask (F): {int(self.f_slider.get())}")

    def load_checkpoint(self):
        """دالة سريعة لتحميل الأوزان المحفوظة للتعرف الفوري"""
        ckpt_path = "las_checkpoint.pth"
        if not os.path.exists(ckpt_path):
            self.infer_status.configure(text="❌ الحالة: لم يتم العثور على ملف las_checkpoint.pth (قم بالتدريب أولاً).")
            return
            
        try:
            self.infer_status.configure(text="⏳ الحالة: جاري تحميل الأوزان... برجاء الانتظار.")
            self.update() # تحديث الواجهة
            
            from member_1 import TOKENIZER
            nc = getattr(TOKENIZER, "vocab_size", 31)
            
            # بناء هيكل الموديل (target_sequence_length=400 كافية جداً لـ Inference)
            model = run_model_pipeline(num_classes=nc, target_sequence_length=400)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            # استخراج الأوزان سواء كانت جوه قاموس أو مباشرة
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch_info = checkpoint.get('epoch', '?')
                loss_info = checkpoint.get('loss', '?')
                if isinstance(loss_info, float):
                    loss_info = f"{loss_info:.4f}"
                status_text = f"✅ الحالة: تم التحميل بنجاح! (اللفة: {epoch_info} | الخسارة: {loss_info}) — اختر ملف للترجمة."
            else:
                model.load_state_dict(checkpoint)
                status_text = "✅ الحالة: تم تحميل الأوزان بنجاح! — اختر ملف للترجمة."
                
            model.to(device)
            model.eval() # وضع الاختبار لمنع التدريب العشوائي
            
            self._infer_model = model
            self.infer_status.configure(text=status_text)
            
        except Exception as e:
            self.infer_status.configure(text=f"❌ خطأ في تحميل الأوزان: {str(e)}")

    def start_pipeline_thread(self):
        self.run_btn.configure(state="disabled", text="جاري التشغيل...")
        self.console_box.delete("0.0", ctk.END)

        policy = self.policy_optionemenu.get()
        w_val = int(self.w_slider.get())
        f_val = int(self.f_slider.get())
        self._gui_augment = {"policy": policy.split()[0] if policy else "None", "w": w_val, "f": f_val}

        print(f"[Config] Policy={self._gui_augment['policy']}, W={w_val}, F={f_val}\n")

        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        model = None
        try:
            print("Starting Full Pipeline...\n")
            time.sleep(0.2)

            print("Running Data Pipeline...")
            data = run_data_pipeline()
            print(f"Data splits loaded.\n")
            time.sleep(0.2)

            print("Running Preprocessing Pipeline...")
            aug = getattr(self, "_gui_augment", None)
            data = run_preprocessing_pipeline(data, gui_augment=aug)
            print("Preprocessing done.\n")
            time.sleep(0.2)

            print("Initializing model...")
            nc = _num_classes_from_loaders(data)
            seq_len = _target_sequence_length_from_loaders(data)
            model = run_model_pipeline(num_classes=nc, target_sequence_length=seq_len)
            
            model.train_epochs = 150
            
            print(f"Model: {type(model).__name__}\n")
            time.sleep(0.2)

            print(f"Training for {model.train_epochs} epochs...")
            model = run_training_pipeline(model, data)
            print("Training done.\n")
            time.sleep(0.2)

            print("Evaluation...")
            results = run_evaluation_pipeline(model)
            print(f"Results: {results}\n")

            self._infer_model = model
            self.after(
                0,
                lambda: self.infer_status.configure(
                    text="الحالة: الموديل جاهز — يمكنك فتح تبويب «التعرف على الصوت» واختيار WAV."
                ),
            )
            print("Pipeline Finished Successfully!")
        except Exception as e:
            print(f"\nError: {str(e)}")
        finally:
            self.after(0, lambda: self.run_btn.configure(state="normal", text="تشغيل الـ Pipeline"))

    def start_prepare_inference_thread(self):
        self.prepare_infer_btn.configure(state="disabled", text="جاري التجهيز...")
        threading.Thread(target=self._prepare_inference_worker, daemon=True).start()

    def _prepare_inference_worker(self):
        try:
            data = run_preprocessing_pipeline(run_data_pipeline())
            nc = _num_classes_from_loaders(data)
            seq_len = _target_sequence_length_from_loaders(data)
            model = run_model_pipeline(num_classes=nc, target_sequence_length=seq_len)
            
            model.train_epochs = 150
            
            model = run_training_pipeline(model, data)
            self._infer_model = model
            self.after(
                0,
                lambda: self.infer_status.configure(text="الحالة: الموديل جاهز — اختر ملف WAV."),
            )
        except Exception as e:
            self.after(0, lambda err=str(e): self.infer_status.configure(text=f"خطأ في التجهيز: {err}"))
        finally:
            self.after(0, lambda: self.prepare_infer_btn.configure(state="normal", text="⚙️ تدريب موديل جديد (من الصفر)"))

    def pick_wav_and_infer(self):
        if self._infer_model is None:
            self.infer_result.delete("0.0", ctk.END)
            self.infer_result.insert(
                "0.0",
                "الموديل غير جاهز.\n\nاضغط «📥 تحميل الأوزان» أولاً لتهيئة الموديل بنجاح.",
            )
            return

        path = filedialog.askopenfilename(
            title="اختر ملف صوتي",
            filetypes=[("WAV", "*.wav"), ("Audio", "*.wav;*.flac;*.ogg"), ("All", "*.*")],
        )
        if not path:
            return

        try:
            from member_1 import predict_from_wav_model_las

            out = predict_from_wav_model_las(self._infer_model, path)
            self.infer_result.delete("0.0", ctk.END)
            self.infer_result.insert("0.0", out["display_text"])
            self.infer_status.configure(text=f"تم التعرف على: {Path(path).name}")
        except Exception as e:
            self.infer_result.delete("0.0", ctk.END)
            self.infer_result.insert("0.0", f"خطأ أثناء التعرف:\n{e}")
            self.infer_status.configure(text="فشل التعرف — راجع الملف أو الموديل.")


def run_pipeline_cli():
    print("=== SpecAugment ASR pipeline (CLI) ===\n")

    print("[OK] data: loading...")
    data = run_data_pipeline()
    if isinstance(data, dict):
        print(
            f"[OK] data loaded: train={len(data.get('train', []))}, "
            f"valid={len(data.get('valid', []))}, test={len(data.get('test', []))}"
        )
    else:
        print(f"[OK] data loaded: type={type(data).__name__}")

    print("\n[OK] preprocessing: log-mel, SpecAugment (train), pad, DataLoaders...")
    data = run_preprocessing_pipeline(data)
    if isinstance(data, dict) and data.get("train_loader"):
        print(
            f"[OK] preprocessing done: train_batches={len(data['train_loader'])}, "
            f"num_classes={data.get('num_classes')}"
        )
    else:
        print("[OK] preprocessing done.")

    nc = _num_classes_from_loaders(data)
    seq_len = _target_sequence_length_from_loaders(data)
    print(f"\n[OK] model: building (num_classes={nc}, target_sequence_length={seq_len})...")
    model = run_model_pipeline(num_classes=nc, target_sequence_length=seq_len)
    
    model.train_epochs = 150
    
    print(f"[OK] model initialized: {type(model).__name__}")

    print(f"\n[OK] training: starting ({model.train_epochs} epochs + ReduceLROnPlateau)...")
    model = run_training_pipeline(model, data)
    print("[OK] training finished.")

    print("\n[OK] evaluation: metrics...")
    results = run_evaluation_pipeline(model)
    print(f"[OK] evaluation result: {results}")
    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_pipeline_cli()
        try:
            print("\u2705 PROJECT RUN SUCCESSFULLY")
        except UnicodeEncodeError:
            print("PROJECT RUN SUCCESSFULLY")
    else:
        app = SpecAugmentApp()
        app.mainloop()