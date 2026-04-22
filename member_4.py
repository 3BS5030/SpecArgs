# -------- DATA --------
def validate_data(data):
    """
    🎯 المطلوب:
    - التأكد إن البيانات سليمة
    - مفيش missing أو corrupt audio

    💡 مهم:
    clean input = better training
    """
    raise NotImplementedError()


# -------- PREPROCESS --------
def pad_sequences(data):
    """
    🎯 المطلوب:
    - توحيد طول sequences

    💡 مهم:
    RNN محتاج input نفس الطول
    """
    raise NotImplementedError()


# -------- MODEL --------
def build_decoder(model):
    """
    🎯 المطلوب:
    - بناء decoder (RNN)

    💡 من الورقة:
    decoder بيولد tokens
    """
    raise NotImplementedError()


# -------- TRAIN --------
def apply_regularization(model):
    """
    🎯 المطلوب:
    - تطبيق techniques زي:
        - dropout
        - label smoothing

    💡 من الورقة:
    label smoothing = مهم جدًا
    """
    raise NotImplementedError()


# -------- EVALUATE --------
def compute_recall(model):
    """
    🎯 المطلوب:
    - حساب recall
    """
    raise NotImplementedError()