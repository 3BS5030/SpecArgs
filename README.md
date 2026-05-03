# SpecAugment ASR Mini Project

This project implements a small, runnable version of the SpecAugment paper:

- log-mel spectrogram inputs at 16 kHz
- online SpecAugment policies: `None`, `LB`, `LD`, `SM`, `SS`
- time warping approximation, frequency masking, and time masking
- listen-style CNN front end with a BiLSTM encoder
- sequence prediction on the YESNO dataset
- token accuracy, sequence accuracy, precision, recall, loss, and WER
- CustomTkinter GUI plus CLI entry point

The original paper trains large LAS models on LibriSpeech/Switchboard. This repo uses the tiny public YESNO dataset so the full pipeline can run on a laptop for a course/demo project.

## Run

```powershell
python main.py --cli
```

For the GUI:

```powershell
python main.py
```

The app can train a model, evaluate it, and run recognition on a selected WAV file. YESNO filenames encode eight binary words, so predictions are shown as an eight-token `no/yes` sequence and evaluated with WER when the filename contains a reference.

## Main Files

- `member_1.py`: data loading, log-mel normalization, CNN/BiLSTM model, optimizer, accuracy, WER, inference
- `member_2.py`: split/tokenization compatibility, training loop, test loss
- `member_3.py`: attention compatibility hook, backprop helper, precision
- `member_4.py`: validation, padding, decoder compatibility hook, label smoothing, recall
- `member_5.py`: SpecAugment policies, batching, compile/update helpers, WER/reporting
- `pipeline/`: stage wrappers for data, preprocessing, model, training, and evaluation
- `main.py`: CLI and GUI

## Notes

The YESNO dataset has only 60 utterances, so metrics can vary with the split and augmentation policy. `LB`/`LD` are paper policies intended for much larger corpora; `None` or `SM` may look better on this tiny demo, while `LB` keeps the paper-style augmentation active by default.
