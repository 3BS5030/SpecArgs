import customtkinter as ctk
import threading
import sys
import time

# استيراد وحدات الـ Pipeline الخاصة بك
from pipeline.data import run_data_pipeline
from pipeline.preprocessing import run_preprocessing_pipeline
from pipeline.model import run_model_pipeline
from pipeline.training import run_training_pipeline
from pipeline.evaluation import run_evaluation_pipeline

# إعداد المظهر العام للواجهة
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class TextRedirector:
    """كلاس مساعد لتوجيه مخرجات الطباعة (print) إلى مربع النص داخل الواجهة"""
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, text):
        self.textbox.insert(ctk.END, text)
        self.textbox.see(ctk.END)  # التمرير التلقائي للأسفل

    def flush(self):
        pass

class SpecAugmentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SpecAugment ASR Pipeline 🚀")
        self.geometry("900x650")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= Sidebar (إعدادات المودل) =================
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Model Settings", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # سياسات الـ Augmentation بناءً على الورقة البحثية
        self.policy_label = ctk.CTkLabel(self.sidebar_frame, text="Augmentation Policy:")
        self.policy_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.policy_optionemenu = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            values=["None", "LB (LibriSpeech Basic)", "LD (LibriSpeech Double)", "SM (Switchboard Mild)", "SS (Switchboard Strong)"],
            command=self.change_policy
        )
        self.policy_optionemenu.grid(row=2, column=0, padx=20, pady=(10, 10))

        # Time Warping (W)
        self.w_label = ctk.CTkLabel(self.sidebar_frame, text="Time Warp (W): 80")
        self.w_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.w_slider = ctk.CTkSlider(self.sidebar_frame, from_=0, to=100, number_of_steps=100, command=lambda v: self.w_label.configure(text=f"Time Warp (W): {int(v)}"))
        self.w_slider.grid(row=4, column=0, padx=20, pady=(5, 10))
        self.w_slider.set(80)

        # Frequency Masking (F)
        self.f_label = ctk.CTkLabel(self.sidebar_frame, text="Freq Mask (F): 27")
        self.f_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.f_slider = ctk.CTkSlider(self.sidebar_frame, from_=0, to=50, number_of_steps=50, command=lambda v: self.f_label.configure(text=f"Freq Mask (F): {int(v)}"))
        self.f_slider.grid(row=6, column=0, padx=20, pady=(5, 10))
        self.f_slider.set(27)

        # زر التشغيل
        self.run_btn = ctk.CTkButton(self.sidebar_frame, text="▶ Run Full Pipeline", command=self.start_pipeline_thread, fg_color="#28a745", hover_color="#218838")
        self.run_btn.grid(row=8, column=0, padx=20, pady=20)

        # ================= Main Frame (شاشة العرض) =================
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.console_label = ctk.CTkLabel(self.main_frame, text="Pipeline Output Console", font=ctk.CTkFont(size=16, weight="bold"))
        self.console_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        # مربع النص لعرض الـ logs
        self.console_box = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.console_box.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # إعادة توجيه الـ print إلى شاشة الواجهة
        sys.stdout = TextRedirector(self.console_box)

    def change_policy(self, choice):
        """تحديث المعاملات تلقائياً بناءً على السياسة المختارة من الورقة البحثية"""
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
            
        self.w_label.configure(text=f"Time Warp (W): {int(self.w_slider.get())}")
        self.f_label.configure(text=f"Freq Mask (F): {int(self.f_slider.get())}")

    def start_pipeline_thread(self):
        """تشغيل الـ Pipeline في Thread منفصل لمنع تجميد الواجهة"""
        self.run_btn.configure(state="disabled", text="Running...")
        self.console_box.delete("0.0", ctk.END) # مسح الشاشة
        
        # استخراج المعاملات لإرسالها للـ Pipeline إذا أردت تعديل دوالك لتقبلها
        policy = self.policy_optionemenu.get()
        w_val = int(self.w_slider.get())
        f_val = int(self.f_slider.get())
        
        print(f"⚙️ Config: Policy={policy.split()[0]}, W={w_val}, F={f_val}\n")
        
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        try:
            print("🚀 Starting Full Pipeline...\n")
            time.sleep(0.5)

            # ========================= DATA =========================
            print("⏳ Running Data Pipeline...")
            data = run_data_pipeline()
            print(f"📦 Data Output: {data}\n")
            time.sleep(0.5)

            # ========================= PREPROCESSING =========================
            print("⏳ Running Preprocessing Pipeline...")
            # يمكنك هنا تمرير معاملات الـ SpecAugment إذا قمت بتعديل دالة run_preprocessing_pipeline لتقبلها
            data = run_preprocessing_pipeline(data)
            print(f"🧹 Preprocessed Data: {data}\n")
            time.sleep(0.5)

            # ========================= MODEL =========================
            print("⏳ Initializing LAS Model...")
            model = run_model_pipeline()
            print(f"🧠 Model: {model}\n")
            time.sleep(0.5)

            # ========================= TRAINING =========================
            print("⏳ Running Training Pipeline (This may take a while)...")
            model = run_training_pipeline(model, data)
            print(f"🏋️ Trained Model: {model}\n")
            time.sleep(0.5)

            # ========================= EVALUATION =========================
            print("⏳ Running Evaluation Pipeline...")
            results = run_evaluation_pipeline(model)
            print(f"📊 Results: {results}\n")

            print("✅ Pipeline Finished Successfully!")
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}")
        finally:
            # إعادة تفعيل الزر بعد الانتهاء
            self.run_btn.configure(state="normal", text="▶ Run Full Pipeline")

if __name__ == "__main__":
    app = SpecAugmentApp()
    app.mainloop()