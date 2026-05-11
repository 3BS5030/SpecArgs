import random


def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        if f_zero == f_zero + f:
            return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if replace_with_zero:
            cloned[0][f_zero:mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned


def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        if t_zero == t_zero + t:
            return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if replace_with_zero:
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[0][:, t_zero:mask_end] = cloned.mean()

    return cloned
