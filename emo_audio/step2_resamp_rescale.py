import librosa
import os
import numpy as np
import soundfile as sf

raw_audio_conditions = [
    'Male CDS',
    'Male ADS',
    'Fem CDS',
    'Fem ADS'
]

TARGET_SF = 48000
TARGET_SCALE = 80  # Praat-like intensity target in dB

for condition in raw_audio_conditions:
    prefix = condition
    style = prefix.split()[1]
    prefix = prefix.replace(' ', '_')

    raw_audio_dir = f"C:/projects/emo_EEG/emo_audio/trimmed/{style}/{prefix}"
    out_audio_dir = f"C:/projects/emo_EEG/emo_audio/trimmed/{style}/{prefix}/{TARGET_SF}Hz_{TARGET_SCALE}dB"
    os.makedirs(out_audio_dir, exist_ok=True)

    wav_files = librosa.util.find_files(raw_audio_dir, ext='wav')

    for wav_path in wav_files:
        name, ext = os.path.splitext(os.path.basename(wav_path))
        out_filename = f"{name}_{TARGET_SF}Hz_{TARGET_SCALE}dB{ext}"
        out_file_path = os.path.join(out_audio_dir, out_filename)

        if os.path.exists(out_file_path):
            print(f"Skip already processed file: {out_file_path}")
            continue

        # keep original sampling rate
        wav_y, wav_sr = librosa.load(wav_path, sr=None)

        # 1) resample
        if wav_sr != TARGET_SF:
            wav_y = librosa.resample(wav_y, orig_sr=wav_sr, target_sr=TARGET_SF)
            wav_sr = TARGET_SF

        # 2) scale like Praat intensity
        current_rms = np.sqrt(np.mean(wav_y ** 2))
        target_rms = 2e-5 * (10 ** (TARGET_SCALE / 20))
        gain = target_rms / current_rms
        wav_y = wav_y * gain

        sf.write(out_file_path, wav_y, wav_sr)

    