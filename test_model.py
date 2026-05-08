"""
Load the fine-tuned model and measure accuracy on a test set.

Usage:
  python test_model.py                          # uses ./finetuned_model/ and ./data/
  python test_model.py --model ./my_model/ --data ./my_test_data/
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from member_4.inference import run_test

if __name__ == "__main__":
    run_test()
