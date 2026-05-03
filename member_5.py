import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


# -------- DATA --------
_SPEC_POLICY = {
    # Table 1 from SpecAugment paper (1904.08779v3)
    "None": {"W": 0, "F": 0, "mF": 0, "T": 0, "p": 0.0, "mT": 0},
    "LB": {"W": 80, "F": 27, "mF": 1, "T": 100, "p": 1.0, "mT": 1},
    "LD": {"W": 80, "F": 27, "mF": 2, "T": 100, "p": 1.0, "mT": 2},
    "SM": {"W": 40, "F": 15, "mF": 2, "T": 70, "p": 0.2, "mT": 2},
    "SS": {"W": 40, "F": 27, "mF": 2, "T": 70, "p": 0.2, "mT": 2},
}


def _time_warp_piecewise(spec_tf, max_warp):
    """Lightweight sparse-image-warp approximation without wrapping audio in time."""
    tau, _ = spec_tf.shape
    if max_warp <= 0 or tau < 4:
        return spec_tf

    center = tau // 2
    warp = random.randint(-min(max_warp, center - 1), min(max_warp, tau - center - 2))
    if warp == 0:
        return spec_tf

    dest = center + warp
    left = spec_tf[: center + 1].transpose(0, 1).unsqueeze(0)
    right = spec_tf[center:].transpose(0, 1).unsqueeze(0)
    left_warped = F.interpolate(left, size=dest + 1, mode="linear", align_corners=True).squeeze(0).transpose(0, 1)
    right_warped = F.interpolate(right, size=tau - dest, mode="linear", align_corners=True).squeeze(0).transpose(0, 1)
    warped = torch.cat([left_warped[:-1], right_warped], dim=0)
    return warped[:tau] if warped.shape[0] >= tau else F.pad(warped.transpose(0, 1), (0, tau - warped.shape[0])).transpose(0, 1)


def _apply_specaugment_to_one(spec_tf, params):
    """
    Apply SpecAugment on one spectrogram in (time, freq).
    Notes:
    - Time warping in paper uses TF sparse_image_warp; here we use a piecewise
      interpolation approximation because it is lightweight and framework-agnostic.
    - Frequency and time masks match paper stochastic definitions and can overlap.
    """
    out = spec_tf.clone()
    tau, nu = out.shape[0], out.shape[1]

    # 1) Time warping approximation
    W = int(params["W"])
    if W > 0 and tau > 3:
        out = _time_warp_piecewise(out, min(W, tau // 3))

    # 2) Frequency masking: mF masks with f ~ U(0, F)
    F = int(params["F"])
    mF = int(params["mF"])
    for _ in range(mF):
        if nu <= 1:
            break
        f = random.randint(0, min(F, nu - 1))
        if f <= 0:
            continue
        f0 = random.randint(0, nu - f)
        out[:, f0 : f0 + f] = 0.0

    # 3) Time masking: mT masks with t ~ U(0, min(T, p*tau))
    T = int(params["T"])
    p = float(params["p"])
    mT = int(params["mT"])
    t_upper = min(T, int(p * tau)) if p > 0 else T
    t_upper = max(0, min(t_upper, max(0, tau - 1)))
    for _ in range(mT):
        if tau <= 1 or t_upper <= 0:
            break
        t = random.randint(0, t_upper)
        if t <= 0:
            continue
        t0 = random.randint(0, tau - t)
        out[t0 : t0 + t, :] = 0.0

    return out


def augment_data(data, mode="train", policy="LB", max_freq_mask=None, max_time_mask=None):
    """Bypass augmentation for Hybrid Wav2Vec2 since it requires 1D waveforms."""
    # الإلغاء الإجباري لأن Wav2Vec2 بياخد 1D waveform والـ Augment القديم بيحتاج 2D
    return data


def smart_collate_fn(batch):
    xs = [item[0] for item in batch]
    ys = []
    for item in batch:
        y = item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else 0
        if torch.is_tensor(y):
            ys.append(y.to(dtype=torch.long).view(-1) if y.dim() > 0 else y.to(dtype=torch.long))
        elif isinstance(y, (list, tuple)):
            ys.append(torch.tensor([int(v) for v in y], dtype=torch.long))
        else:
            ys.append(torch.tensor(int(y), dtype=torch.long))

    # Padding on the fly for this specific batch only! (بيوفر جيجابايتات من الرام)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    
    if ys and any(y.dim() > 0 for y in ys):
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=2) # 2 is PAD_ID
    else:
        ys_padded = torch.stack(ys, dim=0)

    return xs_padded, ys_padded

def batch_data(data, batch_size=2, shuffle=True): 
    # إجبار الكود على عدم تخطي batch_size=2 لحماية الـ RTX 3050 Ti
    if batch_size > 1:
        print(f"[Memory Protect] Reducing batch_size from {batch_size} to 2 to prevent CUDA OOM.")
        batch_size = 1

    if isinstance(data, dict) and ("train" in data or "test" in data):
        train_items = data.get("train") or []
        test_items = data.get("test") if data.get("test") else data.get("valid") or []

        gen = torch.Generator().manual_seed(42)
        # استخدمنا DataLoader مباشر مع smart_collate_fn بدل TensorDataset اللي كان بيفجر الذاكرة
        train_loader = (
            DataLoader(train_items, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=smart_collate_fn, generator=gen if shuffle else None)
            if train_items else None
        )
        test_loader = (
            DataLoader(test_items, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=smart_collate_fn)
            if test_items else None
        )

        out = {"train_loader": train_loader, "test_loader": test_loader, "num_classes": 31}
        return out

    return data

# -------- MODEL --------
def compile_model(model, learning_rate=0.001):
    if isinstance(model, nn.Module):
        if not hasattr(model, "criterion"):
            model.criterion = nn.CrossEntropyLoss(label_smoothing=float(getattr(model, "label_smoothing", 0.1)))
        if not hasattr(model, "optimizer"):
            model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return model
    return model
# -------- TRAIN --------
def update_weights(model):
    if isinstance(model, nn.Module) and getattr(model, "_integrated_training", False):
        return model
    if isinstance(model, nn.Module):
        opt = getattr(model, "optimizer", None)
        if opt is not None:
            opt.step()
            opt.zero_grad(set_to_none=True)
        return model

    optimizer = model
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return model

# -------- EVALUATE --------
def calculate_wer(reference, prediction):
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


def generate_report(model=None, accuracy=None, loss=None, wer=None):
    if isinstance(model, dict) and accuracy is None:
        accuracy = model.get("accuracy")
        loss = model.get("test_loss", model.get("loss"))
        wer = model.get("wer")

    if isinstance(model, nn.Module):
        accuracy = getattr(model, "last_accuracy", accuracy)
        loss = getattr(model, "last_test_loss", getattr(model, "last_train_loss", loss))
        if wer is None:
            wer = getattr(model, "last_wer", None)
        sequence_accuracy = getattr(model, "last_sequence_accuracy", None)
    else:
        sequence_accuracy = None

    print("========== Final Report ==========")
    if accuracy is not None:
        print(f"Accuracy: {float(accuracy):.4f}")
    if loss is not None:
        print(f"Loss: {float(loss):.4f}")
    if wer is not None:
        print(f"WER: {float(wer):.4f}")
    print("==================================")
    return {"accuracy": float(accuracy) if accuracy else 0.0}