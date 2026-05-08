import os
import soundfile as sf
from datasets import Dataset


def load_dataset_from_folder(folder, sample_rate=16000):
    import librosa
    rows = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(('.wav', '.mp3', '.flac')):
            txt = os.path.splitext(f)[0] + '.txt'
            txt_path = os.path.join(folder, txt)
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as fh:
                    text = fh.read().strip()
                sig, sr = sf.read(os.path.join(folder, f), dtype='float32')
                if sr != sample_rate:
                    sig = librosa.resample(sig, orig_sr=sr, target_sr=sample_rate)
                rows.append({"audio": {"array": sig, "sampling_rate": sample_rate}, "text": text})
    print(f"  Loaded {len(rows)} samples from {folder}")
    return Dataset.from_list(rows)
