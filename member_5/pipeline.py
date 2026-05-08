from member_1.audio_utils import load_audio, tfm_spectro
from member_3.time_warp import time_warp
from member_4.masking import freq_mask, time_mask
from member_5.visualization import run_benchmark


def run_pipeline():
    print("=" * 60)
    print("SpecAugment — Full Pipeline")
    print("=" * 60)

    print("\n[Phase 1] Loading audio and creating mel spectrogram...")
    sample = "./party-crowd.wav"
    audio = load_audio(sample)
    print(f"  Loaded: sample rate={audio.sr} Hz, shape={audio.sig.shape}")
    mel = tfm_spectro(audio, ws=512, hop=256, n_mels=128, to_db_scale=True, f_max=8000)
    print(f"  Mel spectrogram shape: {mel.shape}")

    print("\n[Phase 2] Sparse image warp utilities loaded (member_2)")

    print("\n[Phase 3] Applying time warp...")
    warped = time_warp(mel, W=50)
    print(f"  Time warped shape: {warped.shape}")

    print("\n[Phase 4] Applying frequency and time masking...")
    freq_masked = freq_mask(warped, F=30, num_masks=2)
    print(f"  After freq_mask: {freq_masked.shape}")
    final = time_mask(freq_masked, T=40, num_masks=2)
    print(f"  After time_mask: {final.shape}")

    print("\n[Phase 5] Visualization and benchmark...")
    import matplotlib.pyplot as plt
    plt.imsave("augmented_spectrogram.png", final[0].numpy(), cmap="viridis")
    print("  Saved augmented_spectrogram.png")
    run_benchmark(mel, iterations=5)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
