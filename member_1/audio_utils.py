import torch
import torchaudio
from torchaudio import transforms
from collections import namedtuple

AudioData = namedtuple("AudioData", ["sig", "sr"])


def load_audio(path):
    return AudioData(*torchaudio.load(path))


class SpectrogramToDB(object):
    def __init__(self, stype="power", top_db=None):
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = torch.log10(torch.maximum(torch.tensor(self.amin), torch.tensor(self.ref_value)))

    def __call__(self, spec):
        spec_db = self.multiplier * torch.log10(torch.clamp(spec, min=self.amin))
        spec_db -= self.multiplier * self.db_multiplier
        if self.top_db is not None:
            spec_db = torch.max(
                spec_db,
                spec_db.new_full((1,), spec_db.max() - self.top_db),
            )
        return spec_db


def tfm_spectro(ad, sr=16000, to_db_scale=False, n_fft=1024,
                ws=None, hop=None, f_min=0.0, f_max=-80, pad=0, n_mels=128):
    mel = transforms.MelSpectrogram(
        sample_rate=ad.sr, n_mels=n_mels, n_fft=n_fft,
        win_length=ws, hop_length=hop,
        f_min=f_min, f_max=f_max, pad=pad,
    )(ad.sig.reshape(1, -1))
    if to_db_scale:
        mel = SpectrogramToDB(stype="magnitude", top_db=f_max)(mel)
    return mel
