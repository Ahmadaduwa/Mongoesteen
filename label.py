import pandas as pd
import os

# 1. อ่านไฟล์ Excel
excel_file = './Datasets/sheet-data.xlsx'  # เปลี่ยนเป็นชื่อไฟล์ของคุณ
df = pd.read_excel(excel_file)

# 2. สร้างโฟลเดอร์เก็บไฟล์ output (optional)
output_folder = './Datasets/labels'
os.makedirs(output_folder, exist_ok=True)

# 3. ลูปแต่ละแถว สร้างไฟล์และเขียนค่า weight g
for index, row in df.iterrows():
    number = row['Number']      # คอลัมน์ number
    weight = row['weight g']    # คอลัมน์ weight g

    filename = f"combined-{number}.txt"
    filepath = os.path.join(output_folder, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(weight))

print("สร้างไฟล์เสร็จเรียบร้อยแล้ว")
