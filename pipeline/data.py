# Fix: running `python pipeline/data.py` sets cwd imports to `pipeline/`, not project root.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from member_1 import load_raw_data
from member_2 import split_data
from member_3 import shuffle_data
from member_4 import validate_data


def safe_call(func, *args):
    """
    Run a function; on NotImplementedError warn and pass data through.
    """
    try:
        return func(*args)
    except NotImplementedError as e:
        print(f"[WARNING] {e}")
        return args[0] if args else []


def run_data_pipeline():
    """
    DATA PIPELINE
    Order: load → validate → shuffle → split.
    Fix: SpecAugment on log-mel is applied in preprocessing (after normalize_data), not on raw wav here.
    """
    data = safe_call(load_raw_data)

    data = safe_call(validate_data, data)

    data = safe_call(shuffle_data, data)

    data = safe_call(split_data, data)

    return data


if __name__ == "__main__":
    # Run from repo root: python pipeline/data.py
    d = run_data_pipeline()
    if isinstance(d, dict):
        print(
            "train:",
            len(d.get("train", [])),
            "valid:",
            len(d.get("valid", [])),
            "test:",
            len(d.get("test", [])),
        )
    else:
        print("samples:", len(d) if isinstance(d, list) else d)
