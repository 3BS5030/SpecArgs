import torch
import random


# -------- DATA --------
def augment_data(data, mode="train", max_freq_mask=15, max_time_mask=40):
    """
    Apply SpecAugment on spectrogram data.

    SpecAugment includes:
    1. Time Warping
    2. Frequency Masking
    3. Time Masking

    Expected input shape:
    - Single spectrogram: [time, freq]
    - Batch spectrograms: [batch, time, freq]

    Important:
    - Augmentation is applied only on training data.
    """

    if mode != "train":
        return data

    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)

    augmented = data.clone()

    # If single sample [time, freq], convert to batch shape
    single_sample = False
    if augmented.dim() == 2:
        augmented = augmented.unsqueeze(0)
        single_sample = True

    batch_size, time_steps, freq_bins = augmented.shape

    for i in range(batch_size):
        spec = augmented[i]

        # 1) Time Warping - simplified safe version
        shift = random.randint(-2, 2)
        spec = torch.roll(spec, shifts=shift, dims=0)

        # 2) Frequency Masking
        freq_mask_size = random.randint(0, min(max_freq_mask, freq_bins))
        if freq_mask_size > 0:
            freq_start = random.randint(0, freq_bins - freq_mask_size)
            spec[:, freq_start:freq_start + freq_mask_size] = 0

        # 3) Time Masking
        time_mask_size = random.randint(0, min(max_time_mask, time_steps))
        if time_mask_size > 0:
            time_start = random.randint(0, time_steps - time_mask_size)
            spec[time_start:time_start + time_mask_size, :] = 0

        augmented[i] = spec

    if single_sample:
        augmented = augmented.squeeze(0)

    return augmented


# -------- PREPROCESS --------
def batch_data(data, batch_size=32, shuffle=True):
    """
    Split data into mini-batches.

    data can be:
    - list
    - tuple
    - torch Tensor

    Training is done batch-wise.
    """

    if shuffle:
        if torch.is_tensor(data):
            indices = torch.randperm(len(data))
            data = data[indices]
        else:
            data = list(data)
            random.shuffle(data)

    batches = []

    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])

    return batches


# -------- MODEL --------
def compile_model(model, learning_rate=0.001):
    """
    Prepare the model for training.

    Includes:
    - Loss function
    - Optimizer
    """

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_function, optimizer


# -------- TRAIN --------
def update_weights(optimizer):
    """
    Update model weights using optimizer.

    This function should be called after:
    - forward pass
    - loss calculation
    - loss.backward()
    """

    optimizer.step()
    optimizer.zero_grad()


# -------- EVALUATE --------
def calculate_wer(reference, prediction):
    """
    Calculate Word Error Rate.

    WER = (Substitutions + Deletions + Insertions) / Number of reference words
    """

    ref_words = reference.split()
    pred_words = prediction.split()

    n = len(ref_words)
    m = len(pred_words)

    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == pred_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                dp[i][j] = min(substitution, deletion, insertion)

    if n == 0:
        return 0.0

    return dp[n][m] / n


def generate_report(accuracy=None, loss=None, wer=None):
    """
    Generate final evaluation report.

    Includes:
    - Accuracy
    - Loss
    - WER
    """

    print("========== Final Report ==========")

    if accuracy is not None:
        print(f"Accuracy: {accuracy:.4f}")

    if loss is not None:
        print(f"Loss: {loss:.4f}")

    if wer is not None:
        print(f"WER: {wer:.4f}")

    print("==================================")
