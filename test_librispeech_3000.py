"""
Evaluate a Whisper model on up to 3000 LibriSpeech samples.

Usage:
  python test_librispeech_3000.py
  python test_librispeech_3000.py --model ./finetuned_model --librispeech ./LibriSpeech --limit 3000
  python test_librispeech_3000.py --subset dev-clean --limit 100 --batch-size 4
"""

import argparse
import csv
import random
import re
from pathlib import Path

import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None


DEFAULT_MODEL_DIR = Path("finetuned_model")
DEFAULT_LIBRISPEECH_DIR = Path("LibriSpeech")
SAMPLE_RATE = 16000


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ref_item != hyp_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def load_librispeech_manifest(root, subset=None, limit=3000, seed=42):
    root = Path(root)
    search_root = root / subset if subset else root
    if not search_root.exists():
        raise FileNotFoundError(f"LibriSpeech path not found: {search_root}")

    transcript_paths = sorted(search_root.rglob("*.trans.txt"))
    samples = []

    for transcript_path in transcript_paths:
        with transcript_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                sample_id, _, text = line.partition(" ")
                audio_path = transcript_path.parent / f"{sample_id}.flac"
                if audio_path.exists():
                    samples.append(
                        {
                            "id": sample_id,
                            "audio_path": audio_path,
                            "reference": text,
                        }
                    )

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:limit], len(samples)


def read_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        if librosa is None:
            raise ImportError("librosa is required to resample audio to 16 kHz")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return audio


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def compute_metrics(references, hypotheses):
    ref_words_total = 0
    word_errors = 0
    ref_chars_total = 0
    char_errors = 0
    exact_matches = 0

    for ref, hyp in zip(references, hypotheses):
        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)

        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        ref_words_total += len(ref_words)
        word_errors += edit_distance(ref_words, hyp_words)

        ref_chars = list(ref_norm.replace(" ", ""))
        hyp_chars = list(hyp_norm.replace(" ", ""))
        ref_chars_total += len(ref_chars)
        char_errors += edit_distance(ref_chars, hyp_chars)

        exact_matches += int(ref_norm == hyp_norm)

    wer = word_errors / max(ref_words_total, 1)
    cer = char_errors / max(ref_chars_total, 1)
    return {
        "wer": wer,
        "word_accuracy": max(0.0, 1.0 - wer),
        "cer": cer,
        "char_accuracy": max(0.0, 1.0 - cer),
        "exact_match": exact_matches / max(len(references), 1),
        "exact_count": exact_matches,
    }


def evaluate(args):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    samples, available_count = load_librispeech_manifest(
        root=args.librispeech,
        subset=args.subset,
        limit=args.limit,
        seed=args.seed,
    )
    if available_count < args.limit:
        print(
            f"Warning: requested {args.limit} samples, but only {available_count} "
            f"LibriSpeech samples were found. Evaluating all available samples."
        )
    if not samples:
        raise RuntimeError("No LibriSpeech samples found.")

    print(f"Device: {device.upper()}")
    print(f"Model: {args.model}")
    print(f"LibriSpeech: {args.librispeech}")
    print(f"Samples: {len(samples)}")

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    model.eval()
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    references = []
    hypotheses = []
    rows = []

    with torch.no_grad():
        for batch_index, batch in enumerate(batched(samples, args.batch_size), start=1):
            audios = [read_audio(sample["audio_path"]) for sample in batch]
            model_inputs = processor(
                audios,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            input_features = model_inputs.input_features.to(device)
            attention_mask = getattr(model_inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
            predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            for sample, prediction in zip(batch, predictions):
                references.append(sample["reference"])
                hypotheses.append(prediction)
                rows.append(
                    {
                        "id": sample["id"],
                        "audio_path": str(sample["audio_path"]),
                        "reference": sample["reference"],
                        "prediction": prediction,
                    }
                )

            done = min(batch_index * args.batch_size, len(samples))
            if done == len(samples) or batch_index % args.print_every == 0:
                current = compute_metrics(references, hypotheses)
                print(
                    f"[{done}/{len(samples)}] "
                    f"WER={current['wer']:.2%}, "
                    f"Word Acc={current['word_accuracy']:.2%}, "
                    f"Exact={current['exact_count']}/{len(references)}"
                )

    metrics = compute_metrics(references, hypotheses)

    print("\n" + "=" * 60)
    print("LibriSpeech Evaluation Results")
    print("=" * 60)
    print(f"Samples evaluated:       {len(samples)}")
    print(f"Word Error Rate (WER):   {metrics['wer']:.2%}")
    print(f"Word Accuracy:           {metrics['word_accuracy']:.2%}")
    print(f"Character Error Rate:    {metrics['cer']:.2%}")
    print(f"Character Accuracy:      {metrics['char_accuracy']:.2%}")
    print(
        f"Exact Sentence Match:    {metrics['exact_count']}/{len(samples)} "
        f"({metrics['exact_match']:.2%})"
    )
    print("=" * 60)

    if args.output_csv:
        output_path = Path(args.output_csv)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["id", "audio_path", "reference", "prediction"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Whisper model on LibriSpeech."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="model folder")
    parser.add_argument(
        "--librispeech",
        default=str(DEFAULT_LIBRISPEECH_DIR),
        help="LibriSpeech root folder",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="optional subset under LibriSpeech, for example dev-clean",
    )
    parser.add_argument("--limit", type=int, default=3000, help="number of samples")
    parser.add_argument("--seed", type=int, default=42, help="sample shuffle seed")
    parser.add_argument("--batch-size", type=int, default=4, help="inference batch size")
    parser.add_argument("--num-beams", type=int, default=3, help="generation beams")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--print-every", type=int, default=10, help="batches")
    parser.add_argument("--output-csv", default=None, help="optional predictions CSV")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
