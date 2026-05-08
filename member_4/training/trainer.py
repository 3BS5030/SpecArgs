import argparse
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from member_4.utils import load_dataset_from_folder, get_device
from member_4.training.metrics import compute_wer, make_compute_metrics
from member_4.training.dataset import make_prepare_fn
from member_4.training.collator import Collator
from member_4.utils.paths import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="folder with WAV + TXT files")
    parser.add_argument("--model", default="small.en", help="tiny.en | base.en | small.en | medium.en")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    model_id = f"openai/whisper-{args.model}"
    device = get_device()
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
    prepare = make_prepare_fn(processor)
    train_data = train_data.map(prepare, remove_columns=train_data.column_names)
    test_data_proc = test_data.map(prepare, remove_columns=test_data.column_names)

    compute_metrics = make_compute_metrics(processor)

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
