import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

from segmentation import segment_image
from operator_classifier import predict_character
from expression_parser import build_and_evaluate


def run_expression_pipeline(image_path):
    """
    Quy trình đầy đủ: phân đoạn → phân loại → phân tích → tính toán → trực quan hóa.
    """

    # --- Bước 1: Phân đoạn hình ảnh ---
    print("\n[STEP 1] Segmenting image...")
    roi_images, rects, thresh, img_display = segment_image(image_path)
    
    if not roi_images:
        return "", None, "No characters detected."
    
    print(f"         Found {len(roi_images)} character(s).")


    # --- Bước 2: Phân loại từng ký tự ---
    print("\n[STEP 2] Classifying characters...")
    predictions = []
    for i, roi in enumerate(roi_images):
        char, conf = predict_character(roi)
        predictions.append((char, conf))
        print(f"  ROI {i}: predicted '{char}' (confidence {conf:.3f})")

    # --- Bước 3: Tính toán biểu thức ---
    print("\n[STEP 3] Building and evaluating expression...")
    raw_chars = [char for char, _ in predictions]
    expression_str, result_str, error = build_and_evaluate(raw_chars)
    
    print(f"\n  Expression : {expression_str}")
    if error:
        print(f"  Error      : {error}")
    else:
        print(f"  Result     : {result_str}")
    # --- Bước 4: Trực quan hóa ---
    print("\n[STEP 4] Showing results...")
    _visualize(img_display, rects, predictions, roi_images, thresh, expression_str, result_str)
    
    print("\n" + "=" * 60)
    return expression_str, result_str, error


def _visualize(img_display, rects, predictions, roi_images, thresh, expr_str, result_str):
    vis = img_display.copy()
    
    # Chú thích hộp 
    for (x, y, w, h), (char, conf) in zip(rects, predictions):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{char} ({conf:.2f})"
        cv2.putText(vis, label, (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Tạo hình
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mở rộng Tùy chọn 2 — Nhận dạng Biểu thức Số học",
                 fontsize=14, fontweight='bold')

    # Bảng 1: Ảnh gốc 
    axes[0, 0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Final Result (Merged Boxes)")
    axes[0, 0].axis('off')

    # Bảng 2: Ngưỡng nhị phân
    axes[0, 1].imshow(thresh, cmap='gray')
    axes[0, 1].set_title("Binary Input")
    axes[0, 1].axis('off')

    # Bảng 3: Dải ROI
    if roi_images:
        strip_h = 28
        strip_w = 28 * len(roi_images) + 4 * (len(roi_images) - 1)
        strip = np.zeros((strip_h, strip_w), dtype=np.uint8)
        for idx, roi in enumerate(roi_images):
            strip[:, idx * 32:idx * 32 + 28] = roi
        axes[1, 0].imshow(strip, cmap='gray')
        axes[1, 0].set_title("Segmented ROIs (left → right)")
        axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, "No ROIs", ha='center', va='center')
        axes[1, 0].axis('off')

    # Bảng 4: Văn bản kết quả
    axes[1, 1].axis('off')
    result_display = result_str if result_str else "ERROR"
    text_block = (
        f"Recognised Expression:\n"
        f"    {expr_str}\n\n"
        f"Computed Result:\n"
        f"    {result_display}"
    )
    axes[1, 1].text(0.1, 0.5, text_block,
                    fontsize=16, fontfamily='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'input_image/test.jpg'
    run_expression_pipeline(path)