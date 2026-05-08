import argparse
import torch
from member_4.utils import load_dataset_from_folder, get_device
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from member_4.utils import load_dataset_from_folder, get_device
from member_4.utils.paths import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR


def evaluate_model(model, processor, dataset, device):
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


def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help="path to fine-tuned model folder")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="folder with WAV + TXT test files")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device.upper()}")

    print(f"\nLoading model from: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()
    model.to(device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print(f"Loading test data from: {args.data}")
    dataset = load_dataset_from_folder(args.data)
    if len(dataset) == 0:
        print("No test data found!")
        return

    print(f"\nTranscribing {len(dataset)} samples...")
    evaluate_model(model, processor, dataset, device)
