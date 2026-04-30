import torch
import random


# -------- DATA --------
def augment_data(data, mode="train"):
    """
    Apply SpecAugment:
    1. Time Warping (simple placeholder)
    2. Frequency Masking
    3. Time Masking

    data shape expected: [time, freq]
    Applied only on training data.
    """
    if mode != "train":
        return data

    spec = data.clone()

    # 1) Time Warping - simplified version
    # Real time warping is more complex, so we keep it safe here.
    spec = torch.roll(spec, shifts=random.randint(-2, 2), dims=0)

    # 2) Frequency Masking
    num_freqs = spec.shape[1]
    max_freq_mask = min(15, num_freqs)

    f = random.randint(0, max_freq_mask)
    f0 = random.randint(0, max(1, num_freqs - f))

    spec[:, f0:f0 + f] = 0

    # 3) Time Masking
    num_times = spec.shape[0]
    max_time_mask = min(40, num_times)

    t = random.randint(0, max_time_mask)
    t0 = random.randint(0, max(1, num_times - t))

    spec[t0:t0 + t, :] = 0

    return spec


# -------- PREPROCESS --------
def batch_data(data, batch_size=32):
    """
    Split data into batches.
    """
    batches = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)

    return batches


# -------- MODEL --------
def compile_model(model, learning_rate=0.001):
    """
    Prepare model with loss function and optimizer.
    """
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_function, optimizer


# -------- TRAIN --------
def update_weights(optimizer):
    """
    Update model weights using optimizer.
    """
    optimizer.zero_grad()
    optimizer.step()


# -------- EVALUATE --------
def generate_report(accuracy=None, loss=None, wer=None):
    """
    Print final evaluation report.
    """
    print("========== Final Report ==========")

    if accuracy is not None:
        print(f"Accuracy: {accuracy:.4f}")

    if loss is not None:
        print(f"Loss: {loss:.4f}")

    if wer is not None:
        print(f"WER: {wer:.4f}")

    print("==================================")
