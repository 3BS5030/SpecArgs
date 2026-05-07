"""
Load the fine-tuned model and measure accuracy on a test set.

Usage:
  python test_model.py                          # uses ./finetuned_model/ and ./data/
  python test_model.py --model ./my_model/ --data ./my_test_data/
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import argparse
import torch
import soundfile as sf
from datasets import Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate

wer_metric = evaluate.load("wer")


def load_audio_folder(folder, sample_rate=16000):
    import librosa
    rows = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(('.wav', '.mp3', '.flac')):
            txt = os.path.splitext(f)[0] + '.txt'
            txt_path = os.path.join(folder, txt)
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as fh:
                    text = fh.read().strip()
                sig, sr = sf.read(os.path.join(folder, f), dtype='float32')
                if sr != sample_rate:
                    sig = librosa.resample(sig, orig_sr=sr, target_sr=sample_rate)
                rows.append({"audio": {"array": sig, "sampling_rate": sample_rate}, "text": text})
    print(f"  Loaded {len(rows)} test samples from {folder}")
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./finetuned_model", help="path to fine-tuned model folder")
    parser.add_argument("--data", default="./data", help="folder with WAV + TXT test files")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    print(f"\nLoading model from: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()
    model.to(device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print(f"Loading test data from: {args.data}")
    dataset = load_audio_folder(args.data)
    if len(dataset) == 0:
        print("No test data found!")
        return

    print(f"\nTranscribing {len(dataset)} samples...")
    refs, hyps = [], []
    for i, sample in enumerate(dataset):
        audio = sample["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = model.generate(inputs.to(device), num_beams=3)
        hyp = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        ref = sample["text"].strip().lower()
        hyp = hyp.lower()
        refs.append(ref)
        hyps.append(hyp)
        status = "[OK]" if ref == hyp else "[FAIL]"
        print(f"  [{i+1}/{len(dataset)}] Ref: {ref}")
        print(f"                  Hyp: {hyp}  {status}")

    wer = wer_metric.compute(predictions=hyps, references=refs)
    print(f"\n{'='*50}")
    print(f"  Word Error Rate (WER):     {wer:.2%}")
    print(f"  Word Accuracy:            {(1-wer):.2%}")
    correct = sum(1 for r, h in zip(refs, hyps) if r == h)
    print(f"  Exact match:              {correct}/{len(refs)} ({correct/len(refs):.1%})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
