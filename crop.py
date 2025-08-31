import cv2
import os
import tkinter as tk
from tkinter import filedialog
from inference import get_model

# 🚀 Load model
model = get_model(
    model_id="mangosteen-markersc-detection-c7wdu/1",
    api_key="jwFwJKYOs7A8qwQzEb9h"
)

# 📁 เลือกโฟลเดอร์ A (input images)
root = tk.Tk()
root.withdraw()
folder_A = filedialog.askdirectory(title="Select Folder A (Images)", initialdir="./Datasets/")

if not folder_A:
    print("❌ No input folder selected.")
    exit()

# 📁 โฟลเดอร์ B (output crops)
folder_B = os.path.join(folder_A, "CROPPED")
os.makedirs(folder_B, exist_ok=True)

# 🔄 loop ภาพทั้งหมด
filenames = sorted([f for f in os.listdir(folder_A) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

for filename in filenames:
    filepath = os.path.join(folder_A, filename)
    image = cv2.imread(filepath)
    if image is None:
        print(f"⚠️ Failed to load {filename}")
        continue

    # 🔍 detect
    result = model.infer(image)[0]
    if not result.predictions:
        print(f"📷 {filename} → No detections.")
        continue

    mangosteen_count = 0
    for pred in result.predictions:
        label = pred.class_name.strip().lower()
        if label == "mangosteen":
            x, y, w, h = int(pred.x), int(pred.y), int(pred.width), int(pred.height)

            # แปลงเป็นพิกัด crop
            x1, y1 = max(0, x - w//2), max(0, y - h//2)
            x2, y2 = min(image.shape[1], x + w//2), min(image.shape[0], y + h//2)

            crop = image[y1:y2, x1:x2]

            if crop.size > 0:
                mangosteen_count += 1
                out_name = f"{os.path.splitext(filename)[0]}_mangosteen_{mangosteen_count}.jpg"
                out_path = os.path.join(folder_B, out_name)
                cv2.imwrite(out_path, crop)
                print(f"✅ Saved {out_name}")
    
    if mangosteen_count == 0:
        print(f"⚠️ {filename} → No mangosteen detected.")

print("\n🎉 Cropping complete! Check folder:", folder_B)
