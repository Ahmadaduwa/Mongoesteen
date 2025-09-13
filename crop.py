import cv2
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
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

    # หา marker เพื่อนำมาคำนวณ pixels per cm
    marker_ws = [p.width for p in result.predictions if p.class_name.strip().lower() == "marker"]
    pixels_per_cm = None
    if marker_ws:
        # ใช้ค่า median เพื่อความทนต่อ outlier
        median_marker_w = float(np.median(marker_ws))
        if median_marker_w >= 1.0:
            pixels_per_cm = median_marker_w / 3.0
            print(f"ℹ️ {filename} → Found marker, marker width px = {median_marker_w:.1f}, pixels/cm = {pixels_per_cm:.3f}")
    else:
        print(f"ℹ️ {filename} → No marker found (will use fallback).")

    mangosteen_count = 0
    for pred in result.predictions:
        label = pred.class_name.strip().lower()
        if label == "mangosteen":
            cx, cy = int(pred.x), int(pred.y)
            mw, mh = int(pred.width), int(pred.height)

            if pixels_per_cm:
                # ขนาดเป้าหมาย: 8 cm -> พิกเซล
                target_size = int(round(8.0 * pixels_per_cm))
                if target_size < 10:
                    # ป้องกันค่าผิดปกติ
                    target_size = max(mw, mh)
            else:
                # fallback: หากไม่มี marker -> ขยาย bbox ของ mangosteen ประมาณ 1.6 เท่า (ไม่เที่ยงเท่าการมี marker)
                scale = 1.6
                target_size = int(round(max(mw, mh) * scale))
                print(f"⚠️ Using fallback crop size {target_size}px for {filename} (no marker)")

            # สร้าง crop สี่เหลี่ยมจตุรัสกึ่งกลางที่ (cx, cy)
            half = target_size // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(image.shape[1], cx + half)
            y2 = min(image.shape[0], cy + half)

            # หาก crop ได้ขนาด 0 ให้พยายามใช้ bbox เดิม
            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ Invalid crop coords for {filename}, skipping this detection.")
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                mangosteen_count += 1
                out_name = f"{os.path.splitext(filename)[0]}_mangosteen_{mangosteen_count}.jpg"
                out_path = os.path.join(folder_B, out_name)
                cv2.imwrite(out_path, crop)
                print(f"✅ Saved {out_name} (size: {crop.shape[1]}x{crop.shape[0]} px)")
    
    if mangosteen_count == 0:
        print(f"⚠️ {filename} → No mangosteen detected.")

print("\n🎉 Cropping complete! Check folder:", folder_B)
