def make_prepare_fn(processor):
    def prepare(batch):
        audio = batch["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features[0]
        batch["input_features"] = inputs
        batch["labels"] = processor(text=batch["text"], return_tensors="pt").input_ids[0]
        return batch
    return prepare
