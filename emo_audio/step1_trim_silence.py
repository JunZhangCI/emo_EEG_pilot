import librosa
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

raw_audio_conditions = [
    'Male CDS',
    'Male ADS',
    'Fem CDS',
    'Fem ADS'
]
for condition in raw_audio_conditions:  
    prefix = condition # -> e.g. "Fem ADS"
    style = prefix.split()[1]  
    prefix = prefix.replace(' ', '_') 
    raw_audio_path = f"C:/projects/emo_EEG/emo_audio/raw/{style} Stimuli/{condition}/Test"   
    out_audio_path = f"C:/projects/emo_EEG/emo_audio/trimmed/{style}/{prefix}"
    os.makedirs(out_audio_path, exist_ok = True)
    wav_files = librosa.util.find_files(raw_audio_path, ext='wav')
    # Create text file to store boundary times
    boundary_filename = f"{prefix}_boundaries.txt"
    boundary_file_path = os.path.join(out_audio_path, boundary_filename)

    for wav_path in wav_files:
        name, ext = os.path.splitext(os.path.basename(wav_path))
        trimmed_filename = f"{name}_trim{ext}"
        trimmed_file_path = os.path.join(out_audio_path, trimmed_filename)
        if os.path.exists(trimmed_file_path):
            print(f"Skip already trimmed file: {trimmed_file_path}")
            continue
        wav_y, wav_sr = librosa.load(path = wav_path, sr = None)
        yt, index = librosa.effects.trim(
            wav_y, 
            top_db=20,
            frame_length = 512,
            hop_length = 128
            )
        
        # Print original and cleaned durations
        original_duration = librosa.get_duration(y=wav_y, sr=wav_sr)
        trimmed_duration = librosa.get_duration(y=yt, sr=wav_sr)
        # Get boundary times in seconds
        start_time = index[0] / wav_sr 
        end_time = index[1] / wav_sr
        # Save cleaned audio file
        sf.write(trimmed_file_path, yt, wav_sr)
        # Save durations and boundary times to a text file
        with open(boundary_file_path, 'a') as f:
            f.write(f"{trimmed_filename}: \n - Original Duration = {original_duration:.2f} s, "
                    f"Cleaned Duration = {trimmed_duration:.2f} s, Start Time = {start_time:.2f} s, End Time = {end_time:.2f} s\n")