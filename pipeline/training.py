import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch.nn as nn

from member_1 import initialize_optimizer
from member_2 import training_loop
from member_3 import backpropagation_step
from member_4 import apply_regularization
from member_5 import update_weights


def safe_call(func, *args):
    try:
        return func(*args)
    except NotImplementedError as e:
        print(f"[WARNING] {e}")
        return args[0]


def run_training_pipeline(model, data):
    """
    TRAINING PIPELINE
    Fix: `initialize_optimizer` attaches (optimizer, scheduler) to nn.Module — do not assign a tuple to `model`.
    """
    if isinstance(model, nn.Module) and isinstance(data, dict):
        if data.get("train_loader") is not None:
            model.train_loader = data["train_loader"]
        if data.get("test_loader") is not None:
            model.test_loader = data["test_loader"]

    model = safe_call(initialize_optimizer, model)

    model = safe_call(training_loop, model, data)

    model = safe_call(backpropagation_step, model, data)

    model = safe_call(apply_regularization, model)

    model = safe_call(update_weights, model)

    return model


if __name__ == "__main__":
    from pipeline.data import run_data_pipeline
    from pipeline.model import run_model_pipeline
    from pipeline.preprocessing import run_preprocessing_pipeline

    print("[training] data -> preprocess ...")
    data = run_preprocessing_pipeline(run_data_pipeline())
    nc = data.get("num_classes") if isinstance(data, dict) else None
    seq_len = data.get("target_sequence_length") if isinstance(data, dict) else None
    print("[training] building model, num_classes=", nc, "target_sequence_length=", seq_len)
    model = run_model_pipeline(num_classes=nc, target_sequence_length=seq_len)
    model.train_epochs = 30
    print("[training] run_training_pipeline ...")
    model = run_training_pipeline(model, data)
    hist = getattr(model, "train_loss_history", None)
    print("[training] loss per epoch:", hist)

    from pipeline.evaluation import run_evaluation_pipeline

    metrics = run_evaluation_pipeline(model)
    print("[training] test accuracy:", metrics.get("accuracy"), "wer:", metrics.get("wer"))
