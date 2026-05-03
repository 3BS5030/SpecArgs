import random
import numpy as np
import torch
import torch.nn as nn


# -------- DATA --------
def validate_data(data):
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[k] = validate_data(v)
            else:
                out[k] = v
        return out

    clean_data = []
    for sample in data:
        if sample is None:
            continue
        if isinstance(sample, (tuple, list)) and len(sample) >= 1:
            s = sample[0]
            rest = sample[1:]
            if torch.is_tensor(s):
                if not torch.isfinite(s).all() or s.numel() == 0:
                    continue
                clean_data.append((s,) + tuple(rest))
            elif isinstance(s, np.ndarray):
                if np.isnan(s).any() or np.isinf(s).any() or s.size == 0:
                    continue
                clean_data.append((s,) + tuple(rest))
            else:
                clean_data.append(sample)
            continue

        if torch.is_tensor(sample):
            if not torch.isfinite(sample).all() or sample.numel() == 0:
                continue
            clean_data.append(sample)
            continue

        if isinstance(sample, np.ndarray):
            if np.isnan(sample).any() or np.isinf(sample).any() or sample.size == 0:
                continue
            clean_data.append(sample)

    return clean_data


def _pad_waveform_list(items, max_len=None):
    """Pad time axis for a list of (waveform, label) with waveform shape (time)."""
    if not items:
        return []

    tensors = []
    labels = []
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            x, y = item, 0
            
        if not torch.is_tensor(x):
            continue
            
        # التأكد إنها 1D
        if x.dim() > 1:
            x = x.squeeze()
            
        tensors.append(x)
        
        if torch.is_tensor(y):
            labels.append(y.to(dtype=torch.long) if y.dim() > 0 else int(y.item()))
        elif isinstance(y, (list, tuple)):
            labels.append(torch.tensor([int(v) for v in y], dtype=torch.long))
        else:
            labels.append(int(y))

    if not tensors:
        return []

    if max_len is None:
        max_len = max(t.shape[-1] for t in tensors)

    out = []
    for t, y in zip(tensors, labels):
        gap = max_len - t.shape[-1]
        if gap > 0:
            # Padding for 1D tensor
            t = torch.nn.functional.pad(t, (0, gap))
        out.append((t, y))
    return out

# -------- PREPROCESS --------
def _pad_mel_list(items, max_len=None):
    """Pad time axis for a list of (log_mel, label) with log_mel shape (1, n_mels, time)."""
    if not items:
        return []

    tensors = []
    labels = []
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            x, y = item, 0
        if not torch.is_tensor(x):
            continue
        if x.dim() == 2:
            x = x.unsqueeze(0)
        tensors.append(x)
        if torch.is_tensor(y):
            labels.append(y.to(dtype=torch.long) if y.dim() > 0 else int(y.item()))
        elif isinstance(y, (list, tuple)):
            labels.append(torch.tensor([int(v) for v in y], dtype=torch.long))
        else:
            labels.append(int(y))

    if not tensors:
        return []

    if max_len is None:
        max_len = max(t.shape[-1] for t in tensors)

    out = []
    for t, y in zip(tensors, labels):
        gap = max_len - t.shape[-1]
        if gap > 0:
            t = torch.nn.functional.pad(t, (0, gap))
        out.append((t, y))
    return out




def pad_sequences(data, max_len=None):
    return data


# -------- SPEC AUGMENT --------
def spec_augment(spec, F=20, T=50, num_masks=2):
    is_torch = torch.is_tensor(spec)
    out = spec.clone() if is_torch else spec.copy()
    squeezed = False
    transposed = False

    if is_torch and out.dim() == 3 and out.shape[0] == 1:
        out = out.squeeze(0).transpose(0, 1).contiguous()
        squeezed = True
        transposed = True
    elif out.ndim == 2 and out.shape[0] < out.shape[1]:
        out = out.T if not is_torch else out.transpose(0, 1).contiguous()
        transposed = True

    num_time_steps = out.shape[0]
    num_mel_channels = out.shape[1]

    for _ in range(num_masks):
        f = random.randint(0, min(F, max(num_mel_channels - 1, 0)))
        if f <= 0:
            continue
        f0 = random.randint(0, num_mel_channels - f)
        out[:, f0 : f0 + f] = 0

    for _ in range(num_masks):
        t = random.randint(0, min(T, max(num_time_steps - 1, 0)))
        if t <= 0:
            continue
        t0 = random.randint(0, num_time_steps - t)
        out[t0 : t0 + t, :] = 0

    if transposed:
        out = out.transpose(0, 1).contiguous() if is_torch else out.T
    if squeezed:
        out = out.unsqueeze(0)
    return out


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, encoder_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(
            input_size=embed_dim + encoder_dim, 
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, context):
        embedded = self.embedding(x) 
        rnn_input = torch.cat([embedded, context], dim=2) 
        out, hidden = self.rnn(rnn_input, hidden)
        logits = self.fc(out.squeeze(1))
        return logits, hidden

def build_decoder(model_or_vocab=None, embed_dim=256, hidden_dim=512):
    if isinstance(model_or_vocab, nn.Module):
        return model_or_vocab
    vocab_size = model_or_vocab if isinstance(model_or_vocab, int) else 1000
    return Decoder(vocab_size, embed_dim, hidden_dim)

# -------- REGULARIZATION --------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        if self.cls <= 1:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        if pred.dim() == 3:
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def apply_regularization(model, dropout=0.3):
    if not isinstance(model, nn.Module):
        return model
    for module in model.modules():
        if isinstance(module, nn.LSTM):
            module.dropout = dropout
    return model

# -------- EVALUATE --------
def compute_recall(model):
    """Macro recall over classes; sequence outputs are flattened to token predictions."""
    if not isinstance(model, nn.Module):
        return {"recall": 0.0}

    loader = getattr(model, "test_loader", None)
    if loader is None:
        return {"recall": 0.0}

    model.eval()
    dev = getattr(model, "train_device", next(model.parameters()).device)
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            logits = model(inputs, target_seq=labels) if hasattr(model, 'sos_id') else model(inputs)
            preds = logits.argmax(dim=-1) if logits.dim() == 3 else logits.argmax(dim=1)
            preds_all.append(preds.detach().cpu().view(-1))
            labels_all.append(labels.detach().cpu().view(-1))

    if not preds_all:
        return {"recall": 0.0}

    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    classes = torch.unique(labels)
    recalls = []
    per_class = {}
    for cls in classes.tolist():
        cls = int(cls)
        actual = labels == cls
        denom = actual.sum().item()
        value = ((preds == cls) & actual).sum().item() / denom if denom > 0 else 0.0
        recalls.append(value)
        per_class[str(cls)] = float(value)

    recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    return {"recall": recall, "per_class_recall": per_class}


def compute_recall_seq(y_true, y_pred):
    """Legacy sequence recall (local tests)."""
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    true_positive = ((y_true == y_pred) & (y_true != 0)).sum().item()
    actual_positive = (y_true != 0).sum().item()
    if actual_positive == 0:
        return 0.0
    return true_positive / actual_positive
