import os
import soundfile as sf
from datasets import load_dataset

print("Loading Datasets...")

dataset = load_dataset('amanteur/CHAD_hummings')

train_data = dataset['train']

output_dir = 'data/raw/hummings'

os.makedirs(output_dir, exist_ok = True)

for i, item in enumerate(train_data):
    audio_array = item['wav']['array']
    sample_rate = item['wav']['sampling_rate']

    file_path = os.path.join(output_dir, f"hum_{i}.wav")

    sf.write(file_path, audio_array, sample_rate)

    if i > 0 and i % 500 == 0:
        print(f"Successfully saved {i} files")