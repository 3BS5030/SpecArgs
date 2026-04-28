import numpy as np
import torch
import torch.nn as nn
import random

# -------- DATA --------
def validate_data(data):
    """
    data: list of spectrograms (numpy arrays)
    """
    clean_data = []

    for sample in data:
        if sample is None:
            continue
        
        if not isinstance(sample, np.ndarray):
            continue
        
        if np.isnan(sample).any() or np.isinf(sample).any():
            continue
        
        if sample.size == 0:
            continue
        
        clean_data.append(sample)

    return clean_data


# -------- PREPROCESS --------
def pad_sequences(data, max_len=None):
    """
    data: list of (T, F) spectrograms
    """
    if max_len is None:
        max_len = max([x.shape[0] for x in data])

    padded = []
    
    for x in data:
        pad_size = max_len - x.shape[0]
        
        if pad_size > 0:
            pad = np.zeros((pad_size, x.shape[1]))
            x = np.vstack([x, pad])
        
        padded.append(x)

    return np.array(padded)


# -------- SPEC AUGMENT --------
def spec_augment(spec, F=20, T=50, num_masks=2):
    spec = spec.copy()
    num_mel_channels = spec.shape[1]
    num_time_steps = spec.shape[0]

    # Frequency masking
    for _ in range(num_masks):
        f = random.randint(0, F)
        f0 = random.randint(0, num_mel_channels - f)
        spec[:, f0:f0+f] = 0

    # Time masking
    for _ in range(num_masks):
        t = random.randint(0, T)
        t0 = random.randint(0, num_time_steps - t)
        spec[t0:t0+t, :] = 0

    return spec


# -------- MODEL --------
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.0
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


def build_decoder(vocab_size, embed_dim=256, hidden_dim=512):
    """
    بناء Decoder زي LAS paper:
    2-layer RNN يولد tokens
    """
    return Decoder(vocab_size, embed_dim, hidden_dim)


# -------- REGULARIZATION --------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def apply_regularization(model, dropout=0.3):
    """
    - Apply dropout على RNN
    - تجهيز label smoothing loss
    """
    for module in model.modules():
        if isinstance(module, nn.LSTM):
            module.dropout = dropout

    return model


# -------- EVALUATE --------
def compute_recall(y_true, y_pred):
    """
    y_true, y_pred: tensors (batch, seq)
    """
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    true_positive = ((y_true == y_pred) & (y_true != 0)).sum().item()
    actual_positive = (y_true != 0).sum().item()

    if actual_positive == 0:
        return 0.0

    return true_positive / actual_positive


# -------- EXAMPLE USAGE --------
if __name__ == "__main__":
    # Dummy data
    data = [np.random.rand(100, 80), np.random.rand(120, 80)]

    # Validate
    data = validate_data(data)

    # Pad
    data = pad_sequences(data)

    # Augment
    data_aug = [spec_augment(x) for x in data]

    # Model
    vocab_size = 1000
    model = build_decoder(vocab_size)

    # Apply regularization
    model = apply_regularization(model)

    # Dummy training step
    x = torch.randint(0, vocab_size, (2, 10))
    y_true = torch.randint(0, vocab_size, (2, 10))

    y_pred, _ = model(x)

    # Loss
    criterion = LabelSmoothingLoss(vocab_size)
    loss = criterion(y_pred.view(-1, vocab_size), y_true.view(-1))

    # Recall
    preds = y_pred.argmax(dim=-1)
    recall = compute_recall(y_true, preds)

    print("Loss:", loss.item())
    print("Recall:", recall)