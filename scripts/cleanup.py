import os
import re

hummings_dir = "data/raw/hummings"
songs_dir = "data/raw/songs"

print("Scanning for unpaired hummings...")

hum_files = os.listdir(hummings_dir)
deleted_count = 0

for hum_filename in hum_files:
    if not hum_filename.endswith(".wav"):
        continue
        
    numbers = re.findall(r'\d+', hum_filename)
    if not numbers:
        continue
        
    file_id = numbers[0]
    
    matching_song_path = os.path.join(songs_dir, f"song_{file_id}.wav")
    
    if not os.path.exists(matching_song_path):
        hum_path = os.path.join(hummings_dir, hum_filename)
        os.remove(hum_path)
        print(f"[-] Deleted {hum_filename} (No matching song found)")
        deleted_count += 1

print(f"\nCleanup Complete. Removed {deleted_count} orphaned hums.")

remaining_hums = len(os.listdir(hummings_dir))
remaining_songs = len(os.listdir(songs_dir))

print(f"Total Hummings Left: {remaining_hums}")
print(f"Total Songs Left: {remaining_songs}")

if remaining_hums == remaining_songs:
    print("SUCCESS: Your dataset is perfectly balanced.")
else:
    print("WARNING: Folders are still uneven. Check for hidden files.")