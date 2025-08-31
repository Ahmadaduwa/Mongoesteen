import os

# 📁 กำหนดโฟลเดอร์ที่เก็บไฟล์ crop
folder = "./Datasets/Image Data down/CROPPED"

for filename in os.listdir(folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # ลบ `_mangosteen_x` ออก
    new_name = filename.split("_mangosteen_")[0] + ".jpg"
    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)

    # ถ้าไฟล์ใหม่มีอยู่แล้ว ให้ข้าม (กันทับไฟล์)
    if os.path.exists(new_path):
        print(f"⚠️ Skip {filename}, already exists as {new_name}")
        continue

    os.rename(old_path, new_path)
    print(f"✅ {filename} → {new_name}")

print("\n🎉 Rename complete!")
