# -------- DATA --------
def augment_data(data):
    """
    🎯 المطلوب:
    - تطبيق SpecAugment

    💡 من الورقة:
    يتكون من:
    1. Time Warping
    2. Frequency Masking
    3. Time Masking

    🔥 دي أهم فانكشن في المشروع
    """
    raise NotImplementedError()


# -------- PREPROCESS --------
def batch_data(data):
    """
    🎯 المطلوب:
    - تقسيم البيانات إلى batches

    💡 مهم:
    التدريب بيكون batch-wise
    """
    raise NotImplementedError()


# -------- MODEL --------
def compile_model(model):
    """
    🎯 المطلوب:
    - تجميع كل layers
    - تحديد loss function

    💡 مثال:
    CrossEntropy
    """
    raise NotImplementedError()


# -------- TRAIN --------
def update_weights(model):
    """
    🎯 المطلوب:
    - تحديث weights باستخدام optimizer

    💡 مهم:
    step النهائية في التدريب
    """
    raise NotImplementedError()


# -------- EVALUATE --------
def generate_report(model):
    """
    🎯 المطلوب:
    - طباعة تقرير نهائي

    💡 ممكن يشمل:
    - Accuracy
    - Loss
    - WER (الأهم في الورقة)
    """
    raise NotImplementedError()