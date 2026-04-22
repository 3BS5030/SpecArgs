from member_1 import build_embedding_layer
from member_2 import build_encoder
from member_3 import build_attention
from member_4 import build_decoder
from member_5 import compile_model


def safe_call(func, model):
    """
    🔒 وظيفة مساعدة:
    - بتشغّل الفانكشن
    - لو مش متنفذة → warning
    - وترجع الموديل زي ما هو عشان pipeline يكمل

    💡 الهدف:
    منع crash أثناء التجميع
    """
    try:
        return func(model)
    except NotImplementedError as e:
        print(f"[WARNING] {e}")
        return model


def run_model_pipeline():
    """
    =========================
    🧠 MODEL PIPELINE
    =========================

    🎯 الهدف:
    بناء موديل Speech Recognition

    💡 من الورقة:
    النموذج المستخدم هو:
    👉 LAS = Listen, Attend, Spell

    ويتكون من:
    1. CNN (embedding / feature extractor)
    2. Encoder (BiLSTM)
    3. Attention
    4. Decoder (RNN)
    """

    model = []

    # =========================
    # 1. EMBEDDING / CNN
    # =========================
    """
    👤 Member 1

    🎯 المطلوب:
    - استقبال Log-Mel Spectrogram
    - تطبيق CNN layers

    💡 من الورقة:
    "input passes through 2-layer CNN"

    ⚠️ مهم:
    ده بيحول الـ spectrogram ل features مفيدة
    """
    model = safe_call(build_embedding_layer, model)

    # =========================
    # 2. ENCODER
    # =========================
    """
    👤 Member 2

    🎯 المطلوب:
    - بناء Encoder (BiLSTM أو Transformer)

    💡 من الورقة:
    "encoder consists of stacked BiLSTMs"

    الهدف:
    فهم التسلسل الزمني للصوت
    """
    model = safe_call(build_encoder, model)

    # =========================
    # 3. ATTENTION
    # =========================
    """
    👤 Member 3

    🎯 المطلوب:
    - إضافة Attention mechanism

    💡 من الورقة:
    LAS = Listen + Attend + Spell

    الهدف:
    الموديل يركز على أجزاء مهمة من الصوت
    """
    model = safe_call(build_attention, model)

    # =========================
    # 4. DECODER
    # =========================
    """
    👤 Member 4

    🎯 المطلوب:
    - بناء Decoder (RNN)

    💡 من الورقة:
    decoder بيحوّل features → tokens (نص)

    مثال:
    حرف → كلمة → جملة
    """
    model = safe_call(build_decoder, model)

    # =========================
    # 5. COMPILE MODEL
    # =========================
    """
    👤 Member 5

    🎯 المطلوب:
    - تجميع كل layers
    - تحديد:
        - loss function (مثلاً CrossEntropy)
        - optimizer
        - metrics

    💡 مهم:
    ده بيخلي الموديل جاهز للتدريب
    """
    model = safe_call(compile_model, model)

    return model