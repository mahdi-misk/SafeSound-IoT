import os
import subprocess
import glob

test_dir = r"c:\Users\mahdi\OneDrive\Desktop\0096.00\code\test"
wav_files = glob.glob(os.path.join(test_dir, "*.wav"))

print("Extending files to 100 seconds...")
for wav_file in wav_files:
    tmp_file = wav_file + ".tmp.wav"
    print(f"Processing {os.path.basename(wav_file)}...")
    
    # Run ffmpeg to loop the file until it reaches 100 seconds
    cmd = [
        "ffmpeg", "-y", 
        "-stream_loop", "-1", 
        "-i", wav_file, 
        "-t", "100", 
        tmp_file
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        os.replace(tmp_file, wav_file)
        print(f"  -> Successfully extended {os.path.basename(wav_file)} to 100s.")
    else:
        print(f"  -> Failed to process {os.path.basename(wav_file)}")
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

print("All files processed.")
