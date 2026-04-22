# -------- DATA --------
def split_data(data):
    """
    🎯 المطلوب:
    - تقسيم البيانات إلى train / test

    💡 مهم:
    لازم evaluation يكون على test مش train
    """
    raise NotImplementedError()


# -------- PREPROCESS --------
def tokenize_data(data):
    """
    🎯 المطلوب:
    - تحويل النص إلى tokens

    💡 من الورقة:
    استخدام Word Piece Model (WPM)
    """
    raise NotImplementedError()


# -------- MODEL --------
def build_encoder(model):
    """
    🎯 المطلوب:
    - بناء encoder (BiLSTM أو Transformer)

    💡 من الورقة:
    "encoder consists of stacked BiLSTM"
    """
    raise NotImplementedError()


# -------- TRAIN --------
def training_loop(model, data):
    """
    🎯 المطلوب:
    - loop على epochs
    - forward + loss

    💡 مهم:
    التعامل مع augmented data
    """
    raise NotImplementedError()


# -------- EVALUATE --------
def compute_loss(model):
    """
    🎯 المطلوب:
    - حساب loss على test set
    """
    raise NotImplementedError()