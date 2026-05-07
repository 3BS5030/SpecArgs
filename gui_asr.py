import customtkinter as ctk
import sounddevice as sd
import numpy as np
import threading
from transformers import pipeline

# إعدادات المظهر العام للواجهة
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ASRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Speech to Text - Live ASR")
        self.geometry("600x450")
        
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.samplerate = 16000
        
        # تحميل الموديل في الخلفية
        self.status_label = ctk.CTkLabel(self, text="⏳ Loading Model (Whisper)...", font=("Arial", 14), text_color="yellow")
        self.status_label.pack(pady=10)
        self.update() # لتحديث الشاشة قبل تحميل الموديل
        
        # تحميل الموديل (تأخذ بعض الوقت في البداية)
        self.pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        self.status_label.configure(text="✅ Model Loaded! Ready to record.", text_color="white")
        
        # عنوان البرنامج
        self.title_label = ctk.CTkLabel(self, text="🎙️ Live Speech Recognition", font=("Arial", 26, "bold"))
        self.title_label.pack(pady=10)
        
        # زر التشغيل والإيقاف
        self.record_btn = ctk.CTkButton(
            self, 
            text="Start Recording", 
            command=self.toggle_record, 
            fg_color="green", 
            hover_color="darkgreen", 
            font=("Arial", 18, "bold"),
            height=50,
            width=200
        )
        self.record_btn.pack(pady=15)
        
        # مربع عرض النص الناتج
        self.textbox = ctk.CTkTextbox(self, width=500, height=200, font=("Arial", 18))
        self.textbox.pack(pady=10)
        self.textbox.insert("0.0", "Transcription will appear here...\n\n")
        
    def toggle_record(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", "🗣️ ")
        
        self.record_btn.configure(text="🛑 Stop Recording", fg_color="red", hover_color="darkred")
        self.status_label.configure(text="🔴 Recording & Transcribing Live...", text_color="red")
        
        # الدالة التي تجمع أجزاء الصوت أثناء التحدث
        def callback(indata, frames, time, status):
            self.audio_data.append(indata.copy())
            
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=callback, dtype='float32')
        self.stream.start()
        
        # تشغيل حلقة الترجمة الفورية في الخلفية
        import time
        self.transcription_thread = threading.Thread(target=self.live_transcribe_loop)
        self.transcription_thread.start()
        
    def live_transcribe_loop(self):
        while self.is_recording:
            time.sleep(1.5) # نحدث النص كل 1.5 ثانية
            
            if not self.is_recording:
                break
                
            if len(self.audio_data) > 0:
                # نأخذ نسخة من الصوت الحالي لتجنب مشاكل التداخل
                current_audio = list(self.audio_data)
                audio_np = np.concatenate(current_audio, axis=0).squeeze()
                
                try:
                    result = self.pipeline({"sampling_rate": self.samplerate, "raw": audio_np})
                    text = result["text"].strip()
                    self.after(0, self.update_live_textbox, text)
                except Exception:
                    pass

    def stop_recording(self):
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        
        self.record_btn.configure(text="Start Recording", fg_color="green", hover_color="darkgreen", state="normal")
        self.status_label.configure(text="✅ Ready for next recording.", text_color="white")
        
        # معالجة أخيرة نهائية للصوت كاملاً
        if self.audio_data:
            audio_np = np.concatenate(self.audio_data, axis=0).squeeze()
            result = self.pipeline({"sampling_rate": self.samplerate, "raw": audio_np})
            text = result["text"].strip()
            self.after(0, self.update_live_textbox, text + "\n")
            
    def update_live_textbox(self, text):
        if text:
            # نقوم بمسح المربع وكتابة النص كاملاً من جديد ليعطي تأثير الكتابة والتصحيح الحي
            self.textbox.delete("0.0", "end")
            self.textbox.insert("end", "🗣️ " + text)
            self.textbox.see("end")

if __name__ == "__main__":
    app = ASRApp()
    app.mainloop()
