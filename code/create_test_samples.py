import os
import shutil
import glob

dataset_dir = r"c:\Users\mahdi\OneDrive\Desktop\0096.00\code\ai_training\balanced_dataset"
test_dir = r"c:\Users\mahdi\OneDrive\Desktop\0096.00\code\test"

os.makedirs(test_dir, exist_ok=True)

categories = ['id_fan_00', 'id_pump_00', 'id_slider_00', 'id_valve_00']
subcategories = ['normal', 'abnormal']

print("Creating test samples...")

for cat in categories:
    for subcat in subcategories:
        search_path = os.path.join(dataset_dir, cat, subcat, "*.wav")
        files = glob.glob(search_path)
        if files:
            first_file = files[0]
            clean_cat = cat.split('_')[1] 
            new_name = f"{clean_cat}_{subcat}.wav"
            dest_path = os.path.join(test_dir, new_name)
            shutil.copy(first_file, dest_path)
            print(f"Copied: {new_name}")
        else:
            print(f"No files found in {search_path}")

print("Done!")
