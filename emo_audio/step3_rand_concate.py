import librosa
import soundfile as sf
import os
import random
import numpy as np
import csv

raw_audio_conditions = [
    'Male CDS',
    'Male ADS',
    'Fem CDS',
    'Fem ADS'
]

TARGET_SF = 48000
TARGET_SCALE = 80  # Praat-like intensity target in dB
emotions = ['hap', 'sad'] # possible values: 'hap', 'sad', 'ang', 'neu', 'sca'
recordings = 4  # number of times to create the concatenated file
repeatition = 2  # number of times to repeat the senetences in the concatenated file
ISI = 0.5  # inter-stimulus interval in seconds

for condition in raw_audio_conditions:
    prefix = condition
    style = prefix.split()[1]
    prefix = prefix.replace(' ', '_')

    input_audio_dir = f"C:/projects/emo_EEG/emo_audio/trimmed/{style}/{prefix}/{TARGET_SF}Hz_{TARGET_SCALE}dB"
    out_audio_dir = f"C:/projects/emo_EEG/emo_audio/random_cont/{TARGET_SF}Hz_{TARGET_SCALE}dB"
    os.makedirs(out_audio_dir, exist_ok=True)

    csv_filename = "random_cont_order.csv"
    csv_file_path = os.path.join(out_audio_dir, csv_filename)
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["emotion", "recording_index", "output_filename", "file_order"])
    wav_files = librosa.util.find_files(input_audio_dir, ext='wav')

    for emotion in emotions:
        selected_files = []
        # Select files with the specified emotion
        for wav_file in wav_files:
            name, ext = os.path.splitext(os.path.basename(wav_file))
            parts = name.split('_')
            file_emotion = parts[2]
            if file_emotion == emotion:
                # Save file path for selected emotion
                selected_files.append(wav_file)
        # Repeat the selected files as specified
        selected_files = selected_files * repeatition
        # Create concatenated recordings with random order
        for i in range(recordings):
            # Shuffle the selected files randomly
            random_files = random.sample(selected_files, len(selected_files))
            # Concatenate the audio files   
            concatenated_audio = []
            for wav_file in random_files:
                y, sr = librosa.load(wav_file, sr = None)
                concatenated_audio.append(y)
                concatenated_audio.append(np.zeros(int(ISI * sr)))  # Add ISI of silence
            final_audio = np.concatenate(concatenated_audio)
            # Save the concatenated audio file  
            out_filename = f"{prefix}_{emotion}_cont_{i + 1}.wav" 
            out_file_path = os.path.join(out_audio_dir, out_filename)
            sf.write(out_file_path, final_audio, sr)  
            # Write the order of files to the CSV
            with open(csv_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    emotion,
                    i + 1,
                    out_filename,
                    ";\n".join(os.path.basename(f) for f in random_files)
                ])


