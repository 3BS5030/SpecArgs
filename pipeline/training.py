from member_1 import initialize_optimizer
from member_2 import training_loop
from member_3 import backpropagation_step
from member_4 import apply_regularization
from member_5 import update_weights


def safe_call(func, *args):
    """
    🔒 وظيفة مساعدة:
    - بتنفذ الفانكشن
    - لو مش متنفذة → warning
    - وترجع أول argument عشان pipeline يكمل

    💡 الهدف:
    منع crash أثناء التطوير الجماعي
    """
    try:
        return func(*args)
    except NotImplementedError as e:
        print(f"[WARNING] {e}")
        return args[0]


def run_training_pipeline(model, data):
    """
    =========================
    🏋️ TRAINING PIPELINE
    =========================

    🎯 الهدف:
    تدريب الموديل باستخدام البيانات (بعد preprocessing + augmentation)

    💡 من الورقة:
    الأداء بيعتمد جدًا على:
    - Learning rate schedule
    - Regularization (label smoothing + noise)
    """

    # =========================
    # 1. INITIALIZE OPTIMIZER
    # =========================
    """
    👤 Member 1

    🎯 المطلوب:
    - تعريف optimizer (Adam غالبًا)
    - تحديد learning rate schedule:
        1. Warm-up (increase)
        2. ثابت
        3. Decay (exponential)

    💡 من الورقة:
    "learning rate schedule is critical for performance"
    """
    model = safe_call(initialize_optimizer, model)

    # =========================
    # 2. TRAINING LOOP
    # =========================
    """
    👤 Member 2

    🎯 المطلوب:
    - loop على epochs
    - تمرير البيانات (forward pass)
    - حساب loss

    💡 مهم:
    - استخدم augmented data
    - training يكون batch-wise
    """
    model = safe_call(training_loop, model, data)

    # =========================
    # 3. BACKPROPAGATION
    # =========================
    """
    👤 Member 3

    🎯 المطلوب:
    - حساب gradients
    - backward pass

    💡 الهدف:
    معرفة اتجاه تحسين weights
    """
    model = safe_call(backpropagation_step, model, data)

    # =========================
    # 4. REGULARIZATION
    # =========================
    """
    👤 Member 4

    🎯 المطلوب:
    تطبيق techniques لتقليل overfitting:

    - Dropout
    - Label smoothing (🔥 مهم جدًا في الورقة)
    - ممكن Weight Noise

    💡 من الورقة:
    label smoothing ممكن يعمل instability لو مش مظبوط
    """
    model = safe_call(apply_regularization, model)

    # =========================
    # 5. UPDATE WEIGHTS
    # =========================
    """
    👤 Member 5

    🎯 المطلوب:
    - تحديث weights باستخدام optimizer

    مثال:
    optimizer.step()

    💡 دي آخر خطوة في training step
    """
    model = safe_call(update_weights, model)

    return model