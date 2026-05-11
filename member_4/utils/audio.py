import os
import numpy as np
import soundfile as sf
from datasets import Dataset


def preprocess_audio(sig, sr, target_sr=16000, trim_silence=True, normalize=True):
    import librosa
    if sig is None or (isinstance(sig, np.ndarray) and sig.size == 0):
        return np.zeros(0, dtype=np.float32), target_sr
    sig = np.asarray(sig)
    if np.issubdtype(sig.dtype, np.integer):
        info = np.iinfo(sig.dtype)
        sig = sig.astype(np.float32) / max(abs(info.min), abs(info.max))
    else:
        sig = sig.astype(np.float32)
    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)
    if sr != target_sr:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)
    if trim_silence and len(sig) > 0:
        try:
            sig, _ = librosa.effects.trim(sig, top_db=20)
        except Exception:
            pass
    if normalize and len(sig) > 0:
        peak = np.max(np.abs(sig))
        if peak > 0:
            sig = sig / peak * 0.95
    return sig.astype(np.float32), target_sr


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
                sig, sr = preprocess_audio(sig, sr, target_sr=sample_rate,
                                           trim_silence=True, normalize=True)
                rows.append({"audio": {"array": sig, "sampling_rate": sr}, "text": text})
    print(f"  Loaded {len(rows)} samples from {folder}")
    return Dataset.from_list(rows)
