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
import random
import argparse
import torch
import soundfile as sf
from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataclasses import dataclass
from typing import Any
import evaluate
from member_4.masking import freq_mask, time_mask

wer_metric = evaluate.load("wer")


def load_dataset_from_folder(folder, sample_rate=16000):
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
    print(f"  Loaded {len(rows)} samples from {folder}")
    return Dataset.from_list(rows)


def compute_wer(model, processor, dataset, device):
    model.eval()
    refs, hyps = [], []
    for sample in dataset:
        audio = sample["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = model.generate(inputs.to(device), num_beams=3)
        hyp = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        refs.append(sample["text"].lower())
        hyps.append(hyp.lower())
    return wer_metric.compute(predictions=hyps, references=refs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data", help="folder with WAV + TXT files")
    parser.add_argument("--model", default="small.en", help="tiny.en | base.en | small.en | medium.en")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output", default="./finetuned_model")
    args = parser.parse_args()

    model_id = f"openai/whisper-{args.model}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    print(f"Base model: {model_id}")

    print("\nLoading dataset...")
    dataset = load_dataset_from_folder(args.data)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = dataset["train"]
    test_data = dataset["test"]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    print("\nLoading processor and base model...")
    processor = WhisperProcessor.from_pretrained(model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []

    print("\n[Before] Measuring baseline WER...")
    base_wer = compute_wer(base_model, processor, test_data, device)
    print(f"  WER: {base_wer:.2%}  |  Word Accuracy: {(1-base_wer):.2%}")

    print("\nPreparing training data...")
    def prepare(batch):
        audio = batch["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features[0]
        batch["input_features"] = inputs
        batch["labels"] = processor(text=batch["text"], return_tensors="pt").input_ids[0]
        return batch

    train_data = train_data.map(prepare, remove_columns=train_data.column_names)
    test_data_proc = test_data.map(prepare, remove_columns=test_data.column_names)

    @dataclass
    class Collator:
        processor: Any
        specaug_prob: float = 0.0
        def __call__(self, features):
            inputs = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
            if random.random() < self.specaug_prob:
                x = batch["input_features"]
                for i in range(x.shape[0]):
                    xi = x[i].unsqueeze(0)
                    xi = freq_mask(xi, F=20, num_masks=2)
                    xi = time_mask(xi, T=40, num_masks=2)
                    x[i] = xi.squeeze(0)
            labels = [{"input_ids": f["labels"]} for f in features]
            lbl = self.processor.tokenizer.pad(labels, return_tensors="pt")
            batch["labels"] = lbl["input_ids"].masked_fill(lbl["attention_mask"].ne(1), -100)
            return batch

    def compute_metrics(pred):
        ids = pred.predictions
        lbls = pred.label_ids
        lbls[lbls == -100] = processor.tokenizer.pad_token_id
        return {"wer": wer_metric.compute(
            predictions=processor.batch_decode(ids, skip_special_tokens=True),
            references=processor.batch_decode(lbls, skip_special_tokens=True))}

    print(f"\nFine-tuning {args.model} for {args.epochs} epochs...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            args.output,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            predict_with_generate=True,
            generation_max_length=64,
            generation_num_beams=3,
            logging_steps=10,
            report_to=[],
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        ),
        train_dataset=train_data,
        eval_dataset=test_data_proc,
        data_collator=Collator(processor=processor, specaug_prob=0.8),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)

    print("\n=== FINAL RESULTS ===")
    print(f"  Before — WER: {base_wer:.2%}  |  Word Accuracy: {(1-base_wer):.2%}")
    finetuned_wer = compute_wer(model, processor, test_data, device)
    improvement = ((base_wer - finetuned_wer) / base_wer * 100) if base_wer > 0 else 0
    print(f"  After  — WER: {finetuned_wer:.2%}  |  Word Accuracy: {(1-finetuned_wer):.2%}")
    print(f"  Improvement: {improvement:.1f}% error reduction")
    print(f"\nModel saved to: {args.output}")
    print(f"Test it:   python test_model.py")
    print(f"GUI app:   python speech_app.py --model {args.output}")


if __name__ == "__main__":
    main()
