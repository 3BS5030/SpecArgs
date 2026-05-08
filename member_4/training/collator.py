import random
from dataclasses import dataclass
from typing import Any
from member_4.masking import freq_mask, time_mask


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
