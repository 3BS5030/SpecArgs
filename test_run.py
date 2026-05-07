import torchaudio
import matplotlib.pyplot as plt
from spec_augment import tfm_spectro, time_warp, freq_mask, time_mask

# 1. Load the audio file
sample_path = './party-crowd.wav'
sig, sr = torchaudio.load(sample_path)

# 2. Generate the spectrogram
print("Generating Spectrogram...")
# tfm_spectro expects the signal and sample rate.
spectro = tfm_spectro(sig, sr=sr, ws=512, hop=256, n_mels=128, to_db_scale=True, f_max=8000, f_min=-80)

# 3. Apply SpecAugment techniques
print("Applying Time Warp...")
warped_spectro = time_warp(spectro)

print("Applying Frequency Masking...")
freq_masked_spectro = freq_mask(spectro, num_masks=2)

print("Applying Time Masking...")
time_masked_spectro = time_mask(spectro, num_masks=2)

print("Applying Combined Augmentation (Time Warp + Freq Mask + Time Mask)...")
combined_spectro = time_mask(freq_mask(time_warp(spectro), num_masks=2), num_masks=2)

# 4. (Optional) Visualize the results
def show_spectrogram(spec, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec[0].numpy(), origin='lower', aspect='auto')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# Uncomment the following lines to see the plots if you have matplotlib installed
# show_spectrogram(spectro, "Original Spectrogram")
# show_spectrogram(combined_spectro, "Augmented Spectrogram")

print("Successfully applied SpecAugment to the audio sample!")
