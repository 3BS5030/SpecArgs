import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from member_1 import build_embedding_layer
from member_2 import build_encoder
from member_3 import build_attention
from member_4 import build_decoder
from member_5 import compile_model


def safe_call(func, model):
    try:
        return func(model)
    except NotImplementedError as e:
        print(f"[WARNING] {e}")
        return model


def run_model_pipeline(num_classes=None, target_sequence_length=None):
    """
    MODEL PIPELINE
    Fix: build nn.Module with `num_classes` from preprocessing (batch labels) so heads match data.
    """
    model = build_embedding_layer(
        None,
        num_classes=num_classes,
        target_sequence_length=target_sequence_length,
    )

    model = safe_call(build_encoder, model)

    model = safe_call(build_attention, model)

    model = safe_call(build_decoder, model)

    model = safe_call(compile_model, model)

    if hasattr(model, "fc") and hasattr(model.fc, "out_features"):
        oc = model.fc.out_features
        model.class_names = ["no", "yes"] if oc == 2 else [f"c{i}" for i in range(oc)]

    return model


if __name__ == "__main__":
    # YESNO default: eight binary tokens.
    m = run_model_pipeline(num_classes=2, target_sequence_length=8)
    oc = getattr(m.fc, "out_features", "?")
    print(f"[model] built {type(m).__name__}, out_features={oc}, device={next(m.parameters()).device}")
