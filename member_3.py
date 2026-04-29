import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa


# -------- DATA --------
def shuffle_data(data):
    """
    Shuffle dataset to reduce overfitting.
    """
    random.shuffle(data)
    return data


# -------- PREPROCESS --------
def create_features(data, sr=16000, n_mels=80):
    """
    Extract filter-bank features plus delta and delta-delta.

    Accepts either:
    - iterable of raw audio arrays
    - iterable of (audio, label) pairs
    """
    features = []

    def process_audio(audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels
        )

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
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        """
        encoder_outputs: (batch, seq_len, hidden)
        """
        energy = torch.tanh(self.attn(encoder_outputs))
        attention = self.v(energy).squeeze(-1)

        weights = F.softmax(attention, dim=1)
        context = torch.sum(encoder_outputs * weights.unsqueeze(-1), dim=1)

        return context, weights


def build_attention(model):
    """
    Add attention module to the model (LAS style).
    """
    hidden_dim = getattr(model, "hidden_dim", 128)
    model.attention = Attention(hidden_dim)
    return model


# -------- TRAIN --------
def backpropagation_step(model, data):
    """
    Perform a single forward + backward pass.

    The pipeline currently passes `model` and `data` only.
    """
    model.train()

    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise NotImplementedError("Member3: model must provide an optimizer attribute")

    optimizer.zero_grad()

    if isinstance(data, torch.utils.data.DataLoader):
        inputs, targets = next(iter(data))
    elif isinstance(data, (tuple, list)) and len(data) == 2:
        inputs, targets = data
    else:
        raise ValueError("Member3: expected data to be a DataLoader or (inputs, targets) tuple")

    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()

    return model


# -------- EVALUATE --------
def compute_precision(model):
    """
    Compute precision using a data loader attached to the model.
    """
    model.eval()

    data_loader = getattr(model, "validation_loader", None)
    if data_loader is None:
        data_loader = getattr(model, "test_loader", None)
    if data_loader is None:
        data_loader = getattr(model, "data_loader", None)
    if data_loader is None:
        raise NotImplementedError("Member3: model must provide a data loader for evaluation")

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    precision = correct / total if total > 0 else 0.0
    return {"precision": precision}