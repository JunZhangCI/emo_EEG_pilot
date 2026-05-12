import os
from pathlib import Path

raw_audio_conditions = [
    'Male CDS',
    'Male ADS',
    'Fem CDS',
    'Fem ADS'
]

sentence_map = {
    "01": "her coat is on the chair",
    "02": "the road goes up the hill",
    "03": "they're going out tonight",
    "04": "her wore his yellow shirt",
    "05": "they took some food outside",
    "06": "the truck drove up the road",
    "07": "the tall man tied his shoes",
    "08": "the mailman shut the gate",
    "09": "the lady wore a coat",
    "10": "the chicken laid some eggs",
    "11": "a fish swam in the pond",
    "12": "snow falls in the winter",
    "13": "the shirts are in the closet",
    "14": "the broom is in the corner",
}

for condition in raw_audio_conditions:
    prefix = condition
    style = prefix.split()[1]
    prefix = prefix.replace(' ', '_')

    raw_audio_dir = f"C:/projects/emo_EEG/emo_audio/trimmed/{style}/{prefix}"
    raw_audio_dir = Path(raw_audio_dir)
    out_audio_dir = raw_audio_dir
    os.makedirs(out_audio_dir, exist_ok=True)
    
    # 1. read in all wav files in target folder
    for wav_path in raw_audio_dir.glob("*.wav"):
        # 2. separate the file name by "_"
        parts = wav_path.stem.split("_")
        # 3. identify the sentence index (second term)
        if len(parts) < 2:
            print(f"Skipping {wav_path.name}: not enough '_' parts")
            continue
        sent_idx = parts[1]
        # 4. determine content based on second term
        sentence = sentence_map.get(sent_idx)
        if sentence is None:
            print(f"Skipping {wav_path.name}: unknown code '{sent_idx}'")
            continue
        # 5. create a txt file with the same name
        txt_path = wav_path.with_suffix(".txt")
        #skip if txt already exists
        if txt_path.exists():
            continue
        # 6. save txt file in the same folder
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(sentence)

