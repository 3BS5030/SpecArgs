import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- DATA --------
def shuffle_data(data):
    """
    Shuffle dataset. Fix: support dict splits (train/valid/test) from pipeline.data.
    """
    if isinstance(data, dict):
        for key in ("train", "valid", "test", "data", "samples"):
            if key in data and isinstance(data[key], list):
                random.shuffle(data[key])
        return data
    if isinstance(data, list):
        random.shuffle(data)
    return data


# -------- PREPROCESS --------
def create_features(data, sr=16000, n_mels=80):
    """
    Librosa deltas optional. Fix: if data is already log-mel tensors (torch), pass through
    so we do not call librosa on tensors / wrong types.
    """
    if isinstance(data, dict):
        sample_lists = [data.get("train"), data.get("valid"), data.get("test")]
        for lst in sample_lists:
            if lst and len(lst) > 0:
                first = lst[0][0] if isinstance(lst[0], (tuple, list)) else lst[0]
                if torch.is_tensor(first):
                    return data
        return data

    try:
        import librosa
    except ImportError:
        return data

    import numpy as np

    features = []

    def process_audio(audio):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel_spec)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        delta = librosa.feature.delta(log_mel)
        delta2 = librosa.feature.delta(log_mel, order=2)
        return np.stack([log_mel, delta, delta2], axis=0)

    for item in data:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            audio, label = item
            features.append((process_audio(audio), label))
        else:
            features.append(process_audio(item))

    return features


# -------- MODEL (Attention) --------
class Attention(nn.Module):
    def __init__(self, hidden_dim, query_dim=256):
        super().__init__()
        # دمج حالة المُفكك (query) مع مخرجات المُشفر (hidden_dim)
        self.attn = nn.Linear(hidden_dim + query_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, query, encoder_outputs):
        # query: (batch, query_dim) قادم من الـ Decoder
        # encoder_outputs: (batch, seq_len, hidden_dim)
        seq_len = encoder_outputs.size(1)
        
        # تكرار الـ query عشان يدمج مع كل لحظة زمنية في الصوت
        query_repeated = query.unsqueeze(1).repeat(1, seq_len, 1)
        
        # حساب طاقة الانتباه
        energy = torch.tanh(self.attn(torch.cat((query_repeated, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(-1)
        
        weights = F.softmax(attention, dim=1)
        context = torch.sum(encoder_outputs * weights.unsqueeze(-1), dim=1)
        
        return context, weights


def build_attention(model):
    """LAS attention is skipped for the small CNN classifier path."""
    if isinstance(model, nn.Module):
        return model

    hidden_dim = getattr(model, "hidden_dim", 128)
    model.attention = Attention(hidden_dim)
    return model


# -------- TRAIN --------
def backpropagation_step(model, data):
    """
    Fix: integrated PyTorch training already runs backward in training_loop —
    avoid a second backward on wrong `data` shape (dict of loaders).
    """
    if getattr(model, "_integrated_training", False):
        return model

    if not isinstance(model, nn.Module):
        return model

    model.train()
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise NotImplementedError("Member3: model must provide an optimizer attribute")

    optimizer.zero_grad()

    if isinstance(data, torch.utils.data.DataLoader):
        inputs, targets = next(iter(data))
    elif isinstance(data, dict) and data.get("train_loader") is not None:
        inputs, targets = next(iter(data["train_loader"]))
    elif isinstance(data, (tuple, list)) and len(data) == 2:
        inputs, targets = data
    else:
        raise ValueError("Member3: expected DataLoader or data dict with train_loader")

    device = getattr(model, "train_device", inputs.device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    if outputs.dim() == 3:
        loss = F.cross_entropy(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
    else:
        loss = F.cross_entropy(outputs, targets)
    loss.backward()
    return model


# -------- EVALUATE --------
def compute_precision(model):
    """Macro precision over classes; sequence outputs are flattened to token predictions."""
    if not isinstance(model, nn.Module):
        return {"precision": 0.0}

    model.eval()
    data_loader = getattr(model, "test_loader", None) or getattr(model, "validation_loader", None)
    if data_loader is None:
        return {"precision": 0.0}

    preds_all = []
    labels_all = []
    dev = getattr(model, "train_device", next(model.parameters()).device)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            outputs = model(inputs, target_seq=labels) if hasattr(model, 'sos_id') else model(inputs)
            preds = torch.argmax(outputs, dim=-1) if outputs.dim() == 3 else torch.argmax(outputs, dim=1)
            preds_all.append(preds.detach().cpu().view(-1))
            labels_all.append(labels.detach().cpu().view(-1))

    if not preds_all:
        return {"precision": 0.0}

    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    classes = torch.unique(labels)
    precisions = []
    per_class = {}
    for cls in classes.tolist():
        cls = int(cls)
        predicted = preds == cls
        denom = predicted.sum().item()
        value = ((labels == cls) & predicted).sum().item() / denom if denom > 0 else 0.0
        precisions.append(value)
        per_class[str(cls)] = float(value)

    precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
    return {"precision": precision, "per_class_precision": per_class}
