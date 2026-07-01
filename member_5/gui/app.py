import argparse
import os
import sys
import threading

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
except ImportError:
    ctk = None

from member_5.gui.components import RECORD_SECONDS, SAMPLE_RATE


class MissingDependencyError(RuntimeError):
    pass


class SpeechApp:
    def __init__(self, model_spec):
        if ctk is None:
            raise MissingDependencyError(
                "customtkinter / tkinter is not installed.\n"
                "  Install it with:  sudo apt install python3-tk\n"
                "  Then:             pip3 install customtkinter"
            )
        if torch is None:
            raise MissingDependencyError(
                "PyTorch is not installed.\n"
                "  Install it with:  pip3 install torch"
            )

        self.root = ctk.CTk()
        self.model_spec = model_spec
        self.root.title("Speech Recognition")
        self.root.geometry("550x420")
        self.model = None
        self.processor = None
        self.use_transformers = False
        self.is_recording = False
        self.last_transcription = ""

        self.label = ctk.CTkLabel(self.root, text="Speech Recognition", font=("Arial", 22))
        self.label.pack(pady=20)

        self.status = ctk.CTkLabel(self.root, text="Loading model...", font=("Arial", 13))
        self.status.pack(pady=5)

        self.model_label = ctk.CTkLabel(self.root, text="", font=("Arial", 11))
        self.model_label.pack()

        btn_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        btn_frame.pack(pady=10)

        self.record_btn = ctk.CTkButton(btn_frame, text="🎤  Record", command=self.start_recording, width=160, height=40, font=("Arial", 14))
        self.record_btn.pack(side="left", padx=5)

        self.speak_btn = ctk.CTkButton(btn_frame, text="🔊  Speak", command=self.speak_result, width=160, height=40, font=("Arial", 14), fg_color="#e67e22", hover_color="#d35400", state="disabled")
        self.speak_btn.pack(side="left", padx=5)

        self.progress = ctk.CTkProgressBar(self.root, width=300)
        self.progress.pack(pady=5)
        self.progress.set(0)

        self.result_box = ctk.CTkTextbox(self.root, height=120, width=450, font=("Arial", 13), wrap="word")
        self.result_box.pack(pady=10)

        self.info = ctk.CTkLabel(self.root, text="", font=("Arial", 11))
        self.info.pack()

        self.root.after(100, self.load_model)

    def load_model(self):
        self.status.configure(text="Loading model...")
        self.root.update()

        def _load():
            try:
                spec = self.model_spec
                if os.path.isdir(spec):
                    self.use_transformers = True
                    from transformers import WhisperForConditionalGeneration, WhisperProcessor
                    self.processor = WhisperProcessor.from_pretrained(spec)
                    self.model = WhisperForConditionalGeneration.from_pretrained(spec)
                    self.model.eval()
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.model.to(device)
                    name = os.path.basename(os.path.normpath(spec))
                    self.root.after(0, lambda: self.model_label.configure(text=f"Model: {name} ({device.upper()})"))
                else:
                    self.use_transformers = False
                    import whisper
                    self.model = whisper.load_model(spec)
                    self.root.after(0, lambda: self.model_label.configure(text=f"Model: {spec}"))

                self.root.after(0, lambda: self.status.configure(text="Ready \u2014 press Record and speak"))

            except Exception as e:
                self.root.after(0, lambda: self.status.configure(text=f"Error: {e}"))

        threading.Thread(target=_load, daemon=True).start()

    def start_recording(self):
        if self.is_recording or self.model is None:
            return
        self.is_recording = True
        self.record_btn.configure(state="disabled", text="🔴 Recording...")
        self.result_box.delete("0.0", "end")
        self.status.configure(text="Listening...")
        self.progress.set(0)
        threading.Thread(target=self._record_and_transcribe, daemon=True).start()

    def _record_and_transcribe(self):
        try:
            import sounddevice as sd
        except ImportError:
            self.root.after(0, lambda: self.status.configure(text="Install sounddevice & portaudio for recording"))
            self.root.after(0, lambda: self.result_box.insert("0.0", "Missing: pip install sounddevice\nSystem: sudo apt install libportaudio2"))
            self.is_recording = False
            self.root.after(0, lambda: self.record_btn.configure(state="normal", text="🎤  Record"))
            return
        except OSError:
            self.root.after(0, lambda: self.status.configure(text="PortAudio library not found"))
            self.root.after(0, lambda: self.result_box.insert("0.0", "Missing: sudo apt install libportaudio2"))
            self.is_recording = False
            self.root.after(0, lambda: self.record_btn.configure(state="normal", text="🎤  Record"))
            return

        try:
            audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            for i in range(RECORD_SECONDS * 10):
                if not self.is_recording:
                    break
                self.progress.set((i + 1) / (RECORD_SECONDS * 10))
                sd.sleep(100)
            sd.wait()

            audio_flat = audio.flatten()
            self.root.after(0, lambda: self.status.configure(text="Transcribing..."))
            self.root.after(0, lambda: self.progress.set(0))

            if self.use_transformers:
                inputs = self.processor(audio_flat, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                input_features = inputs.input_features
                if torch.cuda.is_available():
                    input_features = input_features.cuda()
                with torch.no_grad():
                    predicted_ids = self.model.generate(input_features)
                text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            else:
                result = self.model.transcribe(audio_flat, language="en")
                text = result["text"].strip()

            self.last_transcription = text
            self.root.after(0, lambda t=text: self._show_and_speak(t))

        except Exception as e:
            self.root.after(0, lambda: self.status.configure(text=f"Error: {e}"))
            self.root.after(0, lambda: self.result_box.insert("0.0", f"Error: {e}"))

        self.is_recording = False
        self.root.after(0, lambda: self.record_btn.configure(state="normal", text="🎤  Record"))
        self.root.after(0, lambda: self.speak_btn.configure(state="normal" if self.last_transcription else "disabled"))
        self.root.after(0, lambda: self.progress.set(1))

    def _show_and_speak(self, text):
        display = text if text else "(no speech detected)"
        self.result_box.insert("0.0", display)
        self.status.configure(text="Done \u2014 Press Record to try again")
        if text:
            self.speak_btn.configure(state="normal")
            threading.Thread(target=self._do_speak, args=(text,), daemon=True).start()

    def speak_result(self):
        text = self.last_transcription
        if not text:
            return
        self.speak_btn.configure(state="disabled", text="🔊  Speaking...")
        threading.Thread(target=self._do_speak, args=(text,), daemon=True).start()

    def _do_speak(self, text):
        try:
            from member_4.tts import speak_text, TextToSpeechError
        except ImportError:
            self.root.after(0, lambda: self.status.configure(text="TTS module not available"))
            self.root.after(0, lambda: self.speak_btn.configure(state="normal", text="🔊  Speak"))
            return

        try:
            speak_text(text)
        except TextToSpeechError as e:
            self.root.after(0, lambda: self.status.configure(text=f"TTS Error: {e}"))
        except Exception as e:
            self.root.after(0, lambda: self.status.configure(text=f"TTS Error: {e}"))
        finally:
            self.root.after(0, lambda: self.speak_btn.configure(state="normal", text="🔊  Speak"))


def launch_app():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny.en", help="model name (tiny, base, small) or path to finetuned model folder")
    args = parser.parse_args()

    try:
        app = SpeechApp(args.model)
        app.root.mainloop()
    except MissingDependencyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
