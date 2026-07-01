"""
Speech recognition GUI supporting both pretrained and fine-tuned models.

Usage:
  python speech_app.py                              # default: pretrained tiny.en
  python speech_app.py --model ./finetuned_model/    # load HuggingFace fine-tuned model
  python speech_app.py --model base.en               # load pretrained base.en
"""

import sys
import os


def main():
    # Add project root to path so member_* packages are found
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from member_5.gui import launch_app
        launch_app()
    except ImportError as e:
        missing = str(e).replace("No module named ", "").strip("'\" ")
        print(f"Missing dependency: {missing}", file=sys.stderr)
        print(file=sys.stderr)
        print("Install all required packages:", file=sys.stderr)
        print(f"  pip install -r {os.path.join(project_root, 'requirements.txt')}", file=sys.stderr)
        print(file=sys.stderr)
        print("If tkinter is missing (Linux):", file=sys.stderr)
        print("  sudo apt install python3-tk", file=sys.stderr)
        print(file=sys.stderr)
        print("If PortAudio is missing (Linux):", file=sys.stderr)
        print("  sudo apt install libportaudio2", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
