# -------- DATA --------
def shuffle_data(data):
    """
    🎯 المطلوب:
    - عمل shuffle للبيانات

    💡 مهم:
    يقلل overfitting
    """
    raise NotImplementedError()


# -------- PREPROCESS --------
def create_features(data):
    """
    🎯 المطلوب:
    - استخراج features من الصوت

    💡 من الورقة:
    log-mel filter banks (مش raw audio)
    """
    raise NotImplementedError()


# -------- MODEL --------
def build_attention(model):
    """
    🎯 المطلوب:
    - إضافة attention mechanism

    💡 من الورقة:
    LAS = Listen + Attend + Spell
    """
    raise NotImplementedError()


# -------- TRAIN --------
def backpropagation_step(model, data):
    """
    🎯 المطلوب:
    - حساب gradients
    - backward pass

    💡 مهم:
    تحسين weights
    """
    raise NotImplementedError()


# -------- EVALUATE --------
def compute_precision(model):
    """
    🎯 المطلوب:
    - حساب precision
    """
    raise NotImplementedError()