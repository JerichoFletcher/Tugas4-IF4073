import cv2
import numpy as np

def visualize_bboxes(thresh_image, original_image, classname):
    output_image = original_image.copy()
    y_non_zero, x_non_zero = np.nonzero(thresh_image)

    if len(x_non_zero) > 0 and len(y_non_zero) > 0:
        x_min, x_max = x_non_zero.min(), x_non_zero.max()
        y_min, y_max = y_non_zero.min(), y_non_zero.max()

    # Gambar bounding box pada gambar asli
    cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Tentukan posisi teks
    text_x = x_min
    text_y = y_min - 10
    
    if text_y < 10:
        text_y = y_max - 10
        text_x = text_x + 20

    # Tambahkan label
    cv2.putText(output_image, classname, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return (output_image)
