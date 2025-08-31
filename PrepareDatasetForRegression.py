# conda activate RoboflowOak

import cv2
import os
import tkinter as tk
from tkinter import filedialog
from inference import get_model # Assuming 'inference.py' contains the get_model function
import math # Import math for isnan check
import pandas as pd
import numpy as np

# 🚀 Load model
# Ensure your Roboflow model ID and API key are correct and active.
model = get_model(
    model_id="mangosteen-markersc-detection-c7wdu/1",
    api_key="jwFwJKYOs7A8qwQzEb9h"
)

# 📁 Ask user to select folder (default path)
root = tk.Tk()
root.withdraw() # Hide the main window
folder_path = filedialog.askdirectory(
    title="Select Folder of Images",
    initialdir="./Datasets/"
)

if not folder_path:
    print("❌ No folder selected. Exiting.")
    exit()

# Define physical dimensions of the markers in millimeters
MARKER_PHYSICAL_SIZE_MM = 30.0 # Both square and circle markers are 30x30mm

# Define a tolerance for filtering out misidentified markers
SIZE_TOLERANCE_MM = 5.0 # Mangosteen detections with dimensions within this tolerance of 30mm will be ignored.

# List to store the results
results_list = []

# --- 🔄 Sort the filenames before processing ---
# This ensures a consistent processing order (e.g., 1.jpg, 2.jpg, ...).
filenames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


# 🔁 Process each image in the sorted list
for filename in filenames:

    filepath = os.path.join(folder_path, filename)

    # Load the original image once
    original_image = cv2.imread(filepath)
    if original_image is None:
        print(f"⚠️ Failed to load image: {filename}")
        continue

    # Create a copy for drawing annotations
    display_image = original_image.copy()

    h_img, w_img = display_image.shape[:2]

    # --- Dynamic Scaling Parameters ---
    scale_factor = min(h_img, w_img) / 720.0
    scaled_font_scale = max(0.4, 0.75 * scale_factor)
    scaled_font_thickness = max(1, int(1 * scale_factor))
    scaled_line_thickness_boxes = max(1, int(2 * scale_factor))
    scaled_line_thickness_arrows = max(1, int(2 * scale_factor))
    scaled_tip_length = max(0.01, 0.03 * scale_factor)

    # 🔍 Inference
    result = model.infer(display_image)[0]

    mangosteen_boxes = []
    potential_markers = []

    print(f"\n📷 File: {filename}")
    if not result.predictions:
        print("    → No detections.")
        # --- Add filename annotation to the image even if no detections are found ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255) # White color
        bg_color = (0, 0, 0) # Black background
        font_scale = max(0.4, 0.75 * scale_factor)
        font_thickness = max(1, int(2 * scale_factor))
        text_pos = (int(10 * scale_factor), int(h_img - 10 * scale_factor))
        (text_w, text_h), _ = cv2.getTextSize(filename, font, font_scale, font_thickness)
        cv2.rectangle(display_image, (text_pos[0] - 5, text_pos[1] + 5), 
                      (text_pos[0] + text_w + 5, text_pos[1] - text_h - 5), bg_color, -1)
        cv2.putText(display_image, filename, text_pos, font, font_scale, text_color, font_thickness)
        
        cv2.imshow("Mangosteen Size Estimation (mm)", display_image)
        key = cv2.waitKey(0)
        if key == 27:
            print("👋 ESC pressed. Exiting.")
            break
        continue

    class_names_detected = set()

    for pred in result.predictions:
        x, y, w, h = pred.x, pred.y, pred.width, pred.height
        label = pred.class_name.strip().lower()
        confidence = pred.confidence

        class_names_detected.add(label)

        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        x1, x2 = max(0, min(x1, w_img - 1)), max(0, min(x2, w_img - 1))
        y1, y2 = max(0, min(y1, h_img - 1)), max(0, min(y2, h_img - 1))

        w_clamped = x2 - x1
        h_clamped = y2 - y1

        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), scaled_line_thickness_boxes // 2 + 1)
        cv2.putText(display_image, label, (x1, max(0, y1 - int(10 * scale_factor))),
                                 cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale * 0.8, (0, 0, 255), scaled_font_thickness)

        if label in ["marker_square", "marker_circle"]:
            if w_clamped > 0:
                ppm_w = w_clamped / MARKER_PHYSICAL_SIZE_MM
                ppm_error_w = abs((w_clamped + 1) / MARKER_PHYSICAL_SIZE_MM - ppm_w)
            else:
                ppm_w = 0
                ppm_error_w = float('inf')

            if h_clamped > 0:
                ppm_h = h_clamped / MARKER_PHYSICAL_SIZE_MM
                ppm_error_h = abs((h_clamped + 1) / MARKER_PHYSICAL_SIZE_MM - ppm_h)
            else:
                ppm_h = 0
                ppm_error_h = float('inf')

            if ppm_error_w <= ppm_error_h:
                best_ppm_for_this_marker = ppm_w
                best_ppm_error_for_this_marker = ppm_error_w
            else:
                best_ppm_for_this_marker = ppm_h
                best_ppm_error_for_this_marker = ppm_error_h

            if not math.isinf(best_ppm_error_for_this_marker):
                potential_markers.append({
                    'class_name': label,
                    'pixel_per_mm': best_ppm_for_this_marker,
                    'pixel_per_mm_error': best_ppm_error_for_this_marker,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })

        elif label == "mangosteen":
            mangosteen_boxes.append((x1, y1, x2, y2))

    print(f"    🔎 Detected classes: {class_names_detected}")

    best_marker = None
    if potential_markers:
        potential_markers.sort(key=lambda m: m['pixel_per_mm_error'])
        best_marker = potential_markers[0]

        print(f"    🏆 Best marker selected: {best_marker['class_name']} with error {best_marker['pixel_per_mm_error']:.4f}")

        bx1, by1, bx2, by2 = best_marker['bbox']
        cv2.rectangle(display_image, (bx1, by1), (bx2, by2), (0, 255, 255), scaled_line_thickness_boxes)
        cv2.putText(display_image, f"BEST MARKER: {best_marker['class_name']}",
                    (bx1, max(0, by1 - int(20 * scale_factor))),
                    cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale, (0, 255, 255), scaled_font_thickness)

    # --- Add filename annotation to the image ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255) # White color
    bg_color = (0, 0, 0) # Black background
    font_scale = max(0.4, 0.75 * scale_factor)
    font_thickness = max(1, int(2 * scale_factor))
    text_pos = (int(10 * scale_factor), int(h_img - 10 * scale_factor))

    # Get text size to draw a background rectangle
    (text_w, text_h), _ = cv2.getTextSize(filename, font, font_scale, font_thickness)
    cv2.rectangle(display_image, (text_pos[0] - 5, text_pos[1] + 5), 
                  (text_pos[0] + text_w + 5, text_pos[1] - text_h - 5), bg_color, -1)
    
    cv2.putText(display_image, filename, text_pos, font, font_scale, text_color, font_thickness)


    if best_marker and best_marker['pixel_per_mm'] > 0:
        pixel_per_mm = best_marker['pixel_per_mm']
        pixel_per_mm_error = best_marker['pixel_per_mm_error']

        marker_mm_est = MARKER_PHYSICAL_SIZE_MM
        marker_pixel_width_used = best_marker['bbox'][2] - best_marker['bbox'][0]
        marker_pixel_height_used = best_marker['bbox'][3] - best_marker['bbox'][1]

        marker_mm_text = f"Ref: {best_marker['class_name']} {marker_pixel_width_used}x{marker_pixel_height_used}px -> {marker_mm_est:.1f}mm"
        marker_ppm_text = f"PPM: {pixel_per_mm:.2f} +/- {pixel_per_mm_error:.4f}"

        pos_x = int(10 * scale_factor)
        pos_y_line1 = int(30 * scale_factor)
        pos_y_line2 = int(pos_y_line1 + 25 * scale_factor)

        (tw1, th1), _ = cv2.getTextSize(marker_mm_text, cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale, scaled_font_thickness)
        (tw2, th2), _ = cv2.getTextSize(marker_ppm_text, cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale, scaled_font_thickness)
        max_tw = max(tw1, tw2)

        cv2.rectangle(display_image,
                      (pos_x - int(5 * scale_factor), pos_y_line1 - th1 - int(5 * scale_factor)),
                      (pos_x + max_tw + int(5 * scale_factor), pos_y_line2 + th2 + int(5 * scale_factor)),
                      (255, 255, 255), -1)

        cv2.putText(display_image, marker_mm_text, (pos_x, pos_y_line1), cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale, (0, 0, 0), scaled_font_thickness)
        cv2.putText(display_image, marker_ppm_text, (pos_x, pos_y_line2), cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale, (0, 0, 0), scaled_font_thickness)

        # Iterate through detected mangosteen boxes and estimate their size
        for i, (x1, y1, x2, y2) in enumerate(mangosteen_boxes):
            width_px = x2 - x1
            height_px = y2 - y1

            width_mm = width_px / pixel_per_mm
            height_mm = height_px / pixel_per_mm

            # ⭐ NEW: Filter out mangosteen detections that are too close to the marker's size
            if (abs(width_mm - MARKER_PHYSICAL_SIZE_MM) < SIZE_TOLERANCE_MM and
                abs(height_mm - MARKER_PHYSICAL_SIZE_MM) < SIZE_TOLERANCE_MM):
                print(f"    ➡️ Skipping mangosteen {i+1}: Detected size ({width_mm:.1f}x{height_mm:.1f}mm) is too close to marker size.")
                continue
            
            # --- Store the results for this mangosteen
            results_list.append({
                'filename': filename,
                'width_mm': width_mm,
                'height_mm': height_mm
            })

            width_error_mm = abs(width_px * pixel_per_mm_error / (pixel_per_mm ** 2))
            height_error_mm = abs(height_px * pixel_per_mm_error / (pixel_per_mm ** 2))

            # ↔️ Horizontal width arrow (yellow)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            cv2.arrowedLine(display_image, (x1, mid_y), (x2, mid_y), (255, 255, 0), scaled_line_thickness_arrows, tipLength=scaled_tip_length)
            cv2.arrowedLine(display_image, (x2, mid_y), (x1, mid_y), (255, 255, 0), scaled_line_thickness_arrows, tipLength=scaled_tip_length)

            text_width = f"W: {width_mm:.1f} +/- {width_error_mm:.1f} mm"
            (w_text, h_text), _ = cv2.getTextSize(text_width, cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale, scaled_font_thickness)
            text_x = mid_x - (w_text // 2)
            text_y = y1 - int(10 * scale_factor)

            if i == 0 and text_y < (pos_y_line2 + th2 + int(10 * scale_factor)):
                text_y = pos_y_line2 + th2 + int(20 * scale_factor)

            cv2.rectangle(display_image,
                          (text_x - int(5 * scale_factor), text_y - h_text - int(5 * scale_factor)),
                          (text_x + w_text + int(5 * scale_factor), text_y + int(5 * scale_factor)),
                          (255, 255, 255), -1)
            cv2.putText(display_image, text_width, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale, (255, 255, 0), scaled_font_thickness)

            # ↕️ Vertical height arrow (green)
            cv2.arrowedLine(display_image, (mid_x, y1), (mid_x, y2), (0, 255, 0), scaled_line_thickness_arrows, tipLength=scaled_tip_length)
            cv2.arrowedLine(display_image, (mid_x, y2), (mid_x, y1), (0, 255, 0), scaled_line_thickness_arrows, tipLength=scaled_tip_length)

            text_height = f"H: {height_mm:.1f} +/- {height_error_mm:.1f} mm"
            (w2_text, h2_text), _ = cv2.getTextSize(text_height, cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale, scaled_font_thickness)
            text2_x = x2 + int(10 * scale_factor)
            text2_y = mid_y

            cv2.rectangle(display_image,
                          (text2_x - int(5 * scale_factor), text2_y - h2_text - int(5 * scale_factor)),
                          (text2_x + w2_text + int(5 * scale_factor), text2_y + int(5 * scale_factor)),
                          (255, 255, 255), -1)
            cv2.putText(display_image, text_height, (text2_x, text2_y), cv2.FONT_HERSHEY_SIMPLEX, scaled_font_scale, (0, 255, 0), scaled_font_thickness)

            print(f"    ✅ Mangosteen {i+1} Estimated: W={width_mm:.1f} +/- {width_error_mm:.1f} mm, H={height_mm:.1f} +/- {height_error_mm:.1f} mm")

    else:
        print("    ⚠️ No suitable 'marker_square' or 'marker_circle' found with finite error – cannot estimate mm.")

    cv2.imshow("Mangosteen Size Estimation (mm)", display_image)
    key = cv2.waitKey(0)
    if key == 27:
        print("👋 ESC pressed. Exiting.")
        break

cv2.destroyAllWindows()

# --- 💾 SAVE RESULTS TO EXCEL FILE ---
if results_list:
    df = pd.DataFrame(results_list)
    output_excel_path = os.path.join(folder_path, "mangosteen_dimensions.xlsx")
    df.to_excel(output_excel_path, index=False)
    print(f"\n🎉 Successfully saved results to: {output_excel_path}")
else:
    print("\n❌ No valid mangosteen data was processed. No Excel file created.")
