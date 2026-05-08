"""
Speech recognition GUI supporting both pretrained and fine-tuned models.

Usage:
  python speech_app.py                              # default: pretrained tiny.en
  python speech_app.py --model ./finetuned_model/    # load HuggingFace fine-tuned model
  python speech_app.py --model base.en               # load pretrained base.en
"""

from member_5.gui import launch_app

if __name__ == "__main__":
    launch_app()
