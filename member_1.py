# =========================
# MEMBER 1
# =========================

# -------- DATA --------
def load_raw_data():
    """
    🎯 المطلوب:
    - تحميل بيانات صوتية (audio dataset)
    - ممكن تكون ملفات wav أو dataset جاهز
    - يفضل تحويل الصوت إلى sampling rate ثابت (مثلاً 16kHz)

    💡 من الورقة:
    احنا في النهاية هنشتغل على spectrogram مش raw audio
    """
    raise NotImplementedError("Member1: load_raw_data not implemented")


# -------- PREPROCESS --------
def normalize_data(data):
    """
    🎯 المطلوب:
    - تحويل الصوت إلى Log-Mel Spectrogram
    - عمل normalization بحيث يكون mean = 0

    💡 من الورقة:
    "spectrograms are normalized to have zero mean"
    """
    raise NotImplementedError("Member1: normalize_data not implemented")


# -------- MODEL --------
def build_embedding_layer(model):
    """
    🎯 المطلوب:
    - إضافة أول layer يستقبل spectrogram
    - ممكن يكون CNN layer (زي الورقة)

    💡 من الورقة:
    "input passes through 2-layer CNN"
    """
    raise NotImplementedError("Member1: build_embedding_layer not implemented")


# -------- TRAIN --------
def initialize_optimizer(model):
    """
    🎯 المطلوب:
    - تعريف optimizer (Adam مثلاً)
    - تحديد learning rate schedule (warmup + decay)

    💡 من الورقة:
    learning rate schedule مهم جدًا للأداء
    """
    raise NotImplementedError("Member1: initialize_optimizer not implemented")


# -------- EVALUATE --------
def compute_accuracy(model):
    """
    🎯 المطلوب:
    - حساب accuracy أو metric بسيط

    💡 ملاحظة:
    في speech الأفضل WER بس هنا accuracy كبداية
    """
    raise NotImplementedError("Member1: compute_accuracy not implemented")