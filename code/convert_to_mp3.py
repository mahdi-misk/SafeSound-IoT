import os
import subprocess
import glob

test_dir = r"c:\Users\mahdi\OneDrive\Desktop\0096.00\code\test"
wav_files = glob.glob(os.path.join(test_dir, "*.wav"))

if not wav_files:
    print("No .wav files found to convert.")
else:
    print(f"Converting {len(wav_files)} .wav files to .mp3...")
    for wav_file in wav_files:
        mp3_file = os.path.splitext(wav_file)[0] + ".mp3"
        print(f"Converting {os.path.basename(wav_file)}...")
        
        cmd = [
            "ffmpeg", "-y", 
            "-i", wav_file, 
            mp3_file
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            os.remove(wav_file)  # Delete original wav file
            print(f"  -> Successfully converted to {os.path.basename(mp3_file)}")
        else:
            print(f"  -> Failed to convert {os.path.basename(wav_file)}")

print("All files processed.")
