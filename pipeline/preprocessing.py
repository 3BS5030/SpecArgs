from member_1 import normalize_data
from member_2 import tokenize_data
from member_3 import create_features
from member_4 import pad_sequences
from member_5 import batch_data


def safe_call(func, *args):
    """
    🔒 وظيفة مساعدة:
    - بتشغّل الفانكشن
    - لو مش متنفذة → warning
    - وترجع الداتا زي ما هي

    💡 الهدف:
    منع توقف البرنامج أثناء التطوير
    """
    try:
        return func(*args)
    except NotImplementedError as e:
        print(f"[WARNING] {e}")
        return args[0]


def run_preprocessing_pipeline(data):
    """
    =========================
    🧹 PREPROCESSING PIPELINE
    =========================

    🎯 الهدف:
    تحويل البيانات من raw format إلى format مناسب للموديل

    💡 من الورقة:
    المدخل الأساسي للموديل هو:
    👉 Log-Mel Spectrogram (مش raw audio)

    """

    # =========================
    # 1. NORMALIZATION
    # =========================
    """
    👤 Member 1

    🎯 المطلوب:
    - تحويل الصوت إلى Log-Mel Spectrogram
    - عمل normalization بحيث mean = 0

    💡 من الورقة:
    "spectrograms are normalized to have zero mean"

    ⚠️ مهم:
    ده step أساسي قبل أي augmentation
    """
    data = safe_call(normalize_data, data)

    # =========================
    # 2. TOKENIZATION
    # =========================
    """
    👤 Member 2

    🎯 المطلوب:
    - تحويل النص (transcript) إلى tokens

    💡 من الورقة:
    استخدام Word Piece Model (WPM)

    مثال:
    "playing" → ["play", "##ing"]
    """
    data = safe_call(tokenize_data, data)

    # =========================
    # 3. FEATURE EXTRACTION
    # =========================
    """
    👤 Member 3

    🎯 المطلوب:
    - استخراج features إضافية من الصوت

    💡 من الورقة:
    استخدام:
    - filter banks
    - delta / delta-delta

    الهدف:
    تحسين تمثيل الصوت
    """
    data = safe_call(create_features, data)

    # =========================
    # 4. PADDING
    # =========================
    """
    👤 Member 4

    🎯 المطلوب:
    - توحيد طول sequences

    💡 مهم:
    - RNN و batch processing يحتاجوا نفس الطول
    - padding غالبًا يكون بـ 0

    مثال:
    [1,2,3] → [1,2,3,0,0]
    """
    data = safe_call(pad_sequences, data)

    # =========================
    # 5. BATCHING
    # =========================
    """
    👤 Member 5

    🎯 المطلوب:
    - تقسيم البيانات إلى batches

    💡 مهم:
    التدريب بيكون batch-wise مش sample-by-sample

    مثال:
    batch_size = 32
    """
    data = safe_call(batch_data, data)

    return data