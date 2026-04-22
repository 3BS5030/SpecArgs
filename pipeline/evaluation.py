from member_1 import compute_accuracy
from member_2 import compute_loss
from member_3 import compute_precision
from member_4 import compute_recall
from member_5 import generate_report


def safe_call(func, model):
    """
    🔒 وظيفة مساعدة:
    - بتنفّذ الفانكشن
    - لو مش متنفذة → تطبع warning
    - وترجع dict فاضي عشان pipeline يكمل

    💡 الهدف:
    منع crash لو عضو لسه مخلصش الجزء بتاعه
    """
    try:
        return func(model)
    except NotImplementedError as e:
        print(f"[WARNING] {e}")
        return {}


def run_evaluation_pipeline(model):
    """
    =========================
    📊 EVALUATION PIPELINE
    =========================

    🎯 الهدف:
    تقييم أداء الموديل بعد التدريب

    💡 من الورقة:
    أهم metric في Speech Recognition هو:
    👉 WER (Word Error Rate)

    لكن احنا هنا بنقسم التقييم على أعضاء الفريق:
    """

    results = {}

    # =========================
    # 1. ACCURACY
    # =========================
    """
    👤 Member 1

    🎯 المطلوب:
    - حساب accuracy (توقعات صحيحة / إجمالي)

    💡 ملاحظة:
    مش الأفضل في ASR لكن useful كبداية
    """
    res = safe_call(compute_accuracy, model)
    if isinstance(res, dict):
        results.update(res)

    # =========================
    # 2. LOSS
    # =========================
    """
    👤 Member 2

    🎯 المطلوب:
    - حساب loss على test set

    💡 مهم:
    يعكس أداء الموديل أثناء التدريب
    """
    res = safe_call(compute_loss, model)
    if isinstance(res, dict):
        results.update(res)

    # =========================
    # 3. PRECISION
    # =========================
    """
    👤 Member 3

    🎯 المطلوب:
    - حساب precision

    💡 مفيد لو عندنا classification أو tokens
    """
    res = safe_call(compute_precision, model)
    if isinstance(res, dict):
        results.update(res)

    # =========================
    # 4. RECALL
    # =========================
    """
    👤 Member 4

    🎯 المطلوب:
    - حساب recall

    💡 مهم:
    يقيس قدرة الموديل على اكتشاف كل الحالات الصحيحة
    """
    res = safe_call(compute_recall, model)
    if isinstance(res, dict):
        results.update(res)

    # =========================
    # 5. FINAL REPORT
    # =========================
    """
    👤 Member 5

    🎯 المطلوب:
    - تجميع النتائج كلها
    - عرضها بشكل منظم

    💡 من الورقة:
    أهم حاجة ممكن يضيفها:
    👉 WER (Word Error Rate)

    مثال:
    {
        "accuracy": 0.85,
        "loss": 0.3,
        "WER": 0.12
    }
    """
    res = safe_call(generate_report, model)
    if isinstance(res, dict):
        results.update(res)

    return results