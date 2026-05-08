"""
Fine-tune Whisper on ./data/ and save for GUI use.

Usage:
  python train_on_data.py                 # uses ./data/, trains small.en
  python train_on_data.py --model base.en  # use a different model

Expected ./data/ structure:
  data/
    audio001.wav    audio001.txt
    audio002.wav    audio002.txt
    ...

Reports WER before and after fine-tuning.
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from member_4.training import run_training

if __name__ == "__main__":
    run_training()
