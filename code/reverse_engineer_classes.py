import os
import random
import numpy as np
import librosa
import onnxruntime as ort
from collections import defaultdict

# Path setup
base_path = r"c:\Users\mahdi\OneDrive\Desktop\0096.00\code\ai_training\balanced_dataset"
onnx_path = r"c:\Users\mahdi\OneDrive\Desktop\0096.00\code\ai_training\audio_classification_model.onnx"

def extract_mfcc_40(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=3.0)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        return None

print("Loading ONNX Model...")
sess = ort.InferenceSession(onnx_path)
input_name = sess.get_inputs()[0].name

mapping_counts = defaultdict(lambda: defaultdict(int))

devices = ['id_fan_00', 'id_pump_00', 'id_slider_00', 'id_valve_00']
states = ['normal', 'abnormal']

for device in devices:
    for state in states:
        folder_path = os.path.join(base_path, device, state)
        if not os.path.exists(folder_path):
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not files:
            continue
            
        # Select up to 10 random files to test
        sample_files = random.sample(files, min(10, len(files)))
        
        for f in sample_files:
            file_path = os.path.join(folder_path, f)
            features = extract_mfcc_40(file_path)
            if features is not None:
                input_data = features.reshape(1, 40, 1).astype(np.float32)
                raw_output = sess.run(None, {input_name: input_data})[0]
                pred_class = int(np.argmax(raw_output[0]))
                
                label_name = f"{device}_{state}"
                mapping_counts[label_name][pred_class] += 1

print("\n--- Discovery Results ---")
for label_name, counts in mapping_counts.items():
    most_frequent_class = max(counts, key=counts.get)
    print(f"Folder '{label_name}' strongly predicts Class: {most_frequent_class} (Counts: {dict(counts)})")
