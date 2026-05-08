import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor

wer_metric = evaluate.load("wer")


def compute_wer(model, processor, dataset, device):
    import torch
    model.to(device)
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


def make_compute_metrics(processor):
    def compute_metrics(pred):
        ids = pred.predictions
        lbls = pred.label_ids
        lbls[lbls == -100] = processor.tokenizer.pad_token_id
        return {"wer": wer_metric.compute(
            predictions=processor.batch_decode(ids, skip_special_tokens=True),
            references=processor.batch_decode(lbls, skip_special_tokens=True))}
    return compute_metrics
