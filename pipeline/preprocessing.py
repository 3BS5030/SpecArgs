import sys
import torch
import torchaudio
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# شيلنا استدعاء normalize_data و augment_data لأننا مش هنستخدمهم
from member_2 import tokenize_data
from member_3 import create_features
from member_4 import pad_sequences
from member_5 import batch_data


def safe_call(func, *args):
    try:
        return func(*args)
    except NotImplementedError as e:
        print(f"[WARNING] {e}")
        return args[0]

def process_audio_for_wav2vec(file_path):
    # 1. قراءة الملف الصوتي الخام
    waveform, sample_rate = torchaudio.load(file_path)
    
    # 2. تحويل الصوت إلى Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 3. توحيد التردد إلى 16000 Hz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        
    # 4. إزالة بُعد القنوات
    waveform = waveform.squeeze(0)
    
    # ==== التعديل الثاني السحري: قص الملفات الطويلة ====
    # الحد الأقصى 10 ثواني (16000 هرتز * 10 ثواني = 160,000 رقم)
    max_length = 16000 * 10
    if waveform.shape[0] > max_length:
        waveform = waveform[:max_length]
    # ===================================================
    
    # 5. تسوية الصوت (Normalization)
    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
    
    return waveform

def process_split_for_wav2vec(data_split):
    """دالة لتحويل الـ Tensors الجاهزة إلى 1D Waveforms مجهزة لـ Wav2Vec2"""
    processed = []
    for item in data_split:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            waveform, label = item[0], item[1]
            try:
                if torch.is_tensor(waveform):
                    # 1. إزالة بُعد القنوات لتصبح موجة 1D
                    if waveform.dim() > 1:
                        waveform = waveform.squeeze(0)
                    
                    # 2. تسوية الصوت (Normalization) لرفع كفاءة الموديل
                    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
                    
                processed.append((waveform, label))
            except Exception as e:
                print(f"❌ Error processing tensor: {e}")
        else:
            processed.append(item)
    return processed
def run_preprocessing_pipeline(data, gui_augment=None):
    """
    HYBRID PREPROCESSING PIPELINE (Wav2Vec2 + LAS)
    Flow: 1D Waveform -> tokenize -> features -> pad -> DataLoaders.
    """
    print("[preprocess] Converting Tensors to 1D Waveforms for Wav2Vec2...")
    
    # معالجة الداتا سواء كانت متقسمة (Dict) أو قائمة واحدة (List)
    if isinstance(data, dict):
        for split in ["train", "valid", "test"]:
            if split in data and isinstance(data[split], list):
                data[split] = process_split_for_wav2vec(data[split])
    elif isinstance(data, list):
        data = process_split_for_wav2vec(data)

    print("[preprocess] SpecAugment forcefully skipped (Not compatible with 1D Wav2Vec2 waveforms).")

    # استكمال باقي الـ Pipeline
    data = safe_call(tokenize_data, data)
    data = safe_call(create_features, data)
    data = safe_call(pad_sequences, data)
    data = safe_call(batch_data, data)

    return data

if __name__ == "__main__":
    from pipeline.data import run_data_pipeline

    print("[preprocessing] loading data via pipeline.data ...")
    raw = run_data_pipeline()
    out = run_preprocessing_pipeline(raw)
    if isinstance(out, dict):
        tl = out.get("train_loader")
        print(
            "[preprocessing] train_batches=",
            len(tl) if tl is not None else 0,
            "num_classes=",
            out.get("num_classes"),
            "keys=",
            list(out.keys()),
        )
    else:
        print("[preprocessing] output:", type(out).__name__)
