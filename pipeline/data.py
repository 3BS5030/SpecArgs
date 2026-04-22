from member_1 import load_raw_data
from member_2 import split_data
from member_3 import shuffle_data
from member_4 import validate_data
from member_5 import augment_data


def safe_call(func, *args):
    """
    🔒 وظيفة مساعدة:
    - بتشغل أي function
    - لو مش متنفذة (NotImplemented) → تطبع Warning
    - وترجع الداتا زي ما هي عشان السيستم يكمل

    💡 الهدف:
    يخلي المشروع يشتغل حتى لو حد مخلصش شغله
    """
    try:
        return func(*args)
    except NotImplementedError as e:
        print(f"[WARNING] {e}")
        return args[0] if args else []


def run_data_pipeline():
    """
    =========================
    📦 DATA PIPELINE
    =========================

    🎯 الهدف:
    تجهيز الداتا من البداية لحد ما تبقى جاهزة تدخل preprocessing

    الترتيب مهم جدًا 👇
    """

    # =========================
    # 1. LOAD RAW DATA
    # =========================
    data = safe_call(load_raw_data)
    """
    👤 Member 1

    🎯 المطلوب:
    - تحميل ملفات الصوت (wav أو dataset)
    - توحيد sampling rate (مثلاً 16kHz)

    💡 من الورقة:
    الداتا في البداية raw audio
    """

    # =========================
    # 2. VALIDATE DATA
    # =========================
    data = safe_call(validate_data, data)
    """
    👤 Member 4

    🎯 المطلوب:
    - التأكد إن مفيش ملفات بايظة
    - إزالة missing / corrupt samples

    💡 مهم:
    جودة الداتا = جودة الموديل
    """

    # =========================
    # 3. SHUFFLE DATA
    # =========================
    data = safe_call(shuffle_data, data)
    """
    👤 Member 3

    🎯 المطلوب:
    - خلط البيانات عشوائيًا

    💡 من ML:
    يقلل overfitting ويخلي التدريب stable
    """

    # =========================
    # 4. SPLIT DATA
    # =========================
    data = safe_call(split_data, data)
    """
    👤 Member 2

    🎯 المطلوب:
    - تقسيم البيانات إلى:
        - train
        - test (أو validation)

    💡 مهم جدًا:
    evaluation لازم يكون على test فقط
    """

    # =========================
    # 5. AUGMENT DATA (SpecAugment)
    # =========================
    data = safe_call(augment_data, data)
    """
    👤 Member 5 (🔥 أهم دور)

    🎯 المطلوب:
    تطبيق SpecAugment على TRAIN DATA فقط:

    1. Time Warping
       - تحريك الإشارة يمين/شمال في الزمن

    2. Frequency Masking
       - إخفاء جزء من الترددات (channels)

    3. Time Masking
       - إخفاء جزء من الزمن

    💡 من الورقة:
    "masking blocks of time steps and frequency channels"

    ⚠️ مهم:
    - متطبقش augmentation على test data
    """

    return data