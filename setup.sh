#!/usr/bin/env bash
set -euo pipefail

echo "=== Speech App — Setup ==="
echo ""

# ---- System dependencies ----
echo "[1/3] Installing system dependencies..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-tk libportaudio2 2>&1 | grep -v "already\|up to date" || true
elif command -v brew &>/dev/null; then
    brew install portaudio
elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm tk portaudio
fi
echo "  done."

# ---- Python packages ----
echo "[2/3] Installing Python packages..."
python3 -c "import venv" 2>/dev/null || { echo "  python3-venv not found, installing..."; sudo apt-get install -y -qq python3-venv; }

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  done."

# ---- Verify ----
echo "[3/3] Verifying installation..."
python3 -c "
import sys, subprocess
ok = True
try:
    import torch; print('  torch       OK')
except: print('  torch       MISSING'); ok = False
try:
    import sounddevice; print('  sounddevice OK')
except: print('  sounddevice MISSING (may need libportaudio2)'); ok = False
try:
    import customtkinter; print('  customtkint OK')
except: print('  customtkint MISSING (may need python3-tk)'); ok = False
try:
    import transformers; print('  transformers OK')
except: print('  transformers MISSING'); ok = False
try:
    from member_5.gui import launch_app; print('  app module  OK')
except: print('  app module  FAIL'); ok = False
sys.exit(0 if ok else 1)
"
echo ""
if [ $? -eq 0 ]; then
    echo "=== All dependencies installed! ==="
    echo "Run:  python speech_app.py --model ./finetuned_model/"
else
    echo "=== Some checks failed — see messages above ==="
fi
