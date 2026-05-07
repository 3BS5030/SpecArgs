import torch
import torchaudio
from transformers import pipeline
import sounddevice as sd
import numpy as np

def record_audio(duration=5, fs=16000):
    print(f"\n🎤 يرجى التحدث بالإنجليزية الآن... (التسجيل لمدة {duration} ثوانٍ)")
    # Recording audio from the microphone
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("✅ تم التسجيل! جاري معالجة الصوت...")
    return recording.squeeze()

def main():
    print("Loading the Speech-to-Text Model (Whisper)...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny"
    )

    # 1. Record audio from the microphone
    sr = 16000
    audio_data = record_audio(duration=5, fs=sr)

    # 2. Pass the recorded audio directly to the ready-made model
    print("Transcribing...")
    result = asr_pipeline({"sampling_rate": sr, "raw": audio_data})

    print("\n" + "="*40)
    print("🔊 النص المسموع (English):")
    print(result["text"].strip())
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
