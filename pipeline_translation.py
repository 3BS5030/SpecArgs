import torch
import torchaudio
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

print("1. Loading Audio-to-Text Model (Whisper)...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny"
)

print("2. Loading Translation Model (English to Arabic)...")
model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


audio_path = "./party-crowd.wav"
print(f"\nProcessing audio file: {audio_path}")

# قراءة الصوت
sig, sr = torchaudio.load(audio_path)
if sr != 16000:
    sig = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(sig)
    sr = 16000
sig_np = sig.squeeze().numpy()

# 1. تحويل الصوت إلى كلام إنجليزي
print("\n--> Step 1: Transcribing Audio to English Text...")
transcription_result = asr_pipeline({"sampling_rate": sr, "raw": sig_np})
english_text = transcription_result["text"].strip()

print(f"English Text: {english_text}")

# 2. ترجمة الكلام الإنجليزي إلى اللغة العربية
print("\n--> Step 2: Translating English to Arabic...")
if english_text:
    inputs = tokenizer(english_text, return_tensors="pt")
    translated_tokens = model.generate(**inputs)
    arabic_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
else:
    arabic_text = "[لم يتم التعرف على أي كلام لترجمته]"

print("\n===============================")
print("النص الإنجليزي:", english_text)
print("الترجمة العربية:", arabic_text)
print("===============================\n")
