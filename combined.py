import cv2
import os
import numpy as np

# 📁 กำหนด path ของโฟลเดอร์
folder_up = "./Datasets/Image Data up/CROPPED"
folder_down = "./Datasets/Image Data down/CROPPED"
folder_side = "./Datasets/Image Data side/CROPPED"
output_folder = "./Datasets/combined2"

os.makedirs(output_folder, exist_ok=True)

# 📌 ฟังก์ชัน resize พร้อม padding (letterbox)
def resize_with_padding(img, target_size=(112, 112)):
    h, w = img.shape[:2]
    # scale โดยให้ด้านที่ยาวสุดไม่เกิน 112
    scale = min(target_size[0] / h, target_size[1] / w, 1.0)  
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # สร้าง canvas ขนาด 112×112 (สีขาว)
    canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

# 📌 ฟังก์ชันรวมภาพ 4 ช่อง
def combine_images(up_img, down_img, side_img, idx):
    # resize ทุกภาพเป็น 112×112
    up_resized = resize_with_padding(up_img, (112, 112))
    down_resized = resize_with_padding(down_img, (112, 112))
    side_resized = resize_with_padding(side_img, (112, 112))
    side_flip = cv2.flip(side_resized, 1)

    # รวมเป็น 224×224
    top_row = np.hstack((up_resized, down_resized))
    bottom_row = np.hstack((side_resized, side_flip))
    combined = np.vstack((top_row, bottom_row))

    # บันทึก
    out_path = os.path.join(output_folder, f"combined-{idx}.jpg")
    cv2.imwrite(out_path, combined)
    print(f"✅ Saved {out_path}")

# 🔄 loop ตามจำนวนไฟล์
up_files = sorted([f for f in os.listdir(folder_up) if f.endswith(".jpg")])
down_files = sorted([f for f in os.listdir(folder_down) if f.endswith(".jpg")])
side_files = sorted([f for f in os.listdir(folder_side) if f.endswith(".jpg")])

# ใช้เลขท้ายไฟล์เป็นตัวจับคู่
for i in range(1, len(up_files) + 1):
    up_path = os.path.join(folder_up, f"mangosteen-vertical-{i}.jpg")
    down_path = os.path.join(folder_down, f"mangosteen-upsidedown-{i}.jpg")
    side_path = os.path.join(folder_side, f"mangosteen-lateral-{i}.jpg")

    if not (os.path.exists(up_path) and os.path.exists(down_path) and os.path.exists(side_path)):
        print(f"⚠️ Missing files for index {i}, skipping")
        continue

    up_img = cv2.imread(up_path)
    down_img = cv2.imread(down_path)
    side_img = cv2.imread(side_path)

    if up_img is None or down_img is None or side_img is None:
        print(f"⚠️ Failed to load images for index {i}")
        continue

    combine_images(up_img, down_img, side_img, i)

print("\n🎉 All images combined! Check folder:", output_folder)
