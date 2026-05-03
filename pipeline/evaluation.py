import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from member_1 import compute_accuracy, compute_wer_optional
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

    res = safe_call(compute_wer_optional, model)
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


if __name__ == "__main__":
    from pipeline.data import run_data_pipeline
    from pipeline.model import run_model_pipeline
    from pipeline.preprocessing import run_preprocessing_pipeline
    from pipeline.training import run_training_pipeline

    print("[evaluation] full mini-run: data -> preprocess -> model -> train -> eval")
    data = run_preprocessing_pipeline(run_data_pipeline())
    nc = data.get("num_classes") if isinstance(data, dict) else None
    seq_len = data.get("target_sequence_length") if isinstance(data, dict) else None
    model = run_model_pipeline(num_classes=nc, target_sequence_length=seq_len)
    model.train_epochs = 30
    model = run_training_pipeline(model, data)
    results = run_evaluation_pipeline(model)
    print("[evaluation] results keys:", list(results.keys()))
    print(
        "[evaluation] summary:",
        {k: results[k] for k in ("accuracy", "wer", "test_loss") if k in results},
    )
