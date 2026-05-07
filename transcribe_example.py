import torch
import torchaudio
from transformers import pipeline

print("Loading the ready-made model (Whisper) for Speech-to-Text...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny"
)

audio_path = "./party-crowd.wav"
print(f"Loading audio file: {audio_path}...")

# نستخدم torchaudio لقراءة الملف لأن مكتبة transformers تحتاج برنامج ffmpeg إذا مررنا لها مسار الملف مباشرة
sig, sr = torchaudio.load(audio_path)

# موديل Whisper يحتاج أن يكون تردد الصوت 16000 هرتز، لذا نقوم بتحويله إن لم يكن كذلك
if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    sig = resampler(sig)
    sr = 16000

# تحويل البيانات إلى مصفوفة ذات بعد واحد (1D Numpy Array) كما يطلب الموديل
sig_np = sig.squeeze().numpy()

print("Transcribing...")
# نمرر مصفوفة الصوت مباشرة للموديل بدلاً من مسار الملف
result = asr_pipeline({"sampling_rate": sr, "raw": sig_np})

print("\n=== Transcription Result ===")
print(result["text"])
print("============================")
