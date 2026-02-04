import cv2
import numpy as np

# --- HÀM 1: ĐỌC ẢNH AN TOÀN ---
def read_image_safe(path):
    try:
        stream = open(path, "rb")
        bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

# --- HÀM 2: GỘP CÁC BOX BỊ ĐỨT (Logic mới thêm) ---
def merge_broken_parts(rects):
    if len(rects) < 2:
        return rects

    rects.sort(key=lambda r: r[0])  # sắp xếp từ trái sang phải
    
    merged_rects = []
    skip_next = False
    
    for i in range(len(rects)):
        if skip_next:
            skip_next = False
            continue
            
        x1, y1, w1, h1 = rects[i]
        
        if i == len(rects) - 1:
            merged_rects.append((x1, y1, w1, h1))
            break
            
        x2, y2, w2, h2 = rects[i + 1]
        
        # Kiểm tra nếu căn chỉnh theo chiều dọc (tâm gần nhau)
        center1 = x1 + w1 // 2
        center2 = x2 + w2 // 2
        dist_centers = abs(center1 - center2)
        
        if dist_centers < 20:
            # Gộp hai hộp
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1 + w1, x2 + w2) - new_x
            new_h = max(y1 + h1, y2 + h2) - new_y
            merged_rects.append((new_x, new_y, new_w, new_h))
            skip_next = True
        else:
            merged_rects.append((x1, y1, w1, h1))
            
    return merged_rects


def segment_image(image_path):
    """
    Phân đoạn hình ảnh chứa chữ số/toán tử viết tay thành các ROI 28×28 riêng lẻ.
    
    Tham số
    ----------
    image_path : str
        Đường dẫn đến hình ảnh đầu vào
    
    Trả về
    -------
    roi_images : list of np.ndarray (28×28 uint8)
        Hình ảnh ký tự riêng lẻ, được sắp xếp từ trái sang phải
    rects : list of (x, y, w, h)
        Hộp giới hạn tương ứng với mỗi ROI
    thresh : np.ndarray
        Hình ảnh nhị phân đã ngưỡng hóa (để trực quan hóa)
    img_display : np.ndarray
        Hình ảnh màu gốc (để trực quan hóa)
    
    Ngoại lệ
    ------
    FileNotFoundError : nếu không thể đọc hình ảnh
    """
    img = read_image_safe(image_path)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc hình ảnh: {image_path}")

    img_display = img.copy()
    
    # Xử lý hình ảnh BGRA
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_display = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- Quy trình tiền xử lý ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 5
    )
    
    # Các thao tác hình thái học: làm sạch nhiễu + chữa lành các vết nứt
    kernel_clean = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    kernel_heal = np.ones((5, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_heal)

    # --- Tìm đường viền ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc ban đầu: bỏ qua các hộp rất lớn (có thể là đường viền) và nhiễu nhỏ
    initial_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 300 and h < 300 and h > 5 and w > 5:
            initial_rects.append((x, y, w, h))

    # Gộp các phần ký tự bị vỡ (chạy hai lần để đảm bảo)
    final_rects = merge_broken_parts(initial_rects)
    final_rects = merge_broken_parts(final_rects)

    # Sắp xếp từ trái sang phải (thứ tự đọc)
    final_rects.sort(key=lambda r: r[0])

    # --- Trích xuất và chuẩn hóa ROI ---
    roi_images = []
    valid_rects = []
    
    for (x, y, w, h) in final_rects:
        # Lọc: bỏ qua các hộp nhỏ (nhiễu)
        if h < 15 and w < 15:
            continue

        # Trích xuất ROI với padding
        pad = 5
        roi = thresh[
            max(0, y - pad):min(thresh.shape[0], y + h + pad),
            max(0, x - pad):min(thresh.shape[1], x + w + pad)
        ]
        
        if roi.size == 0:
            continue

        # Padding hình vuông để duy trì tỷ lệ khung hình
        h_roi, w_roi = roi.shape
        max_dim = max(h_roi, w_roi)
        square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
        start_x = (max_dim - w_roi) // 2
        start_y = (max_dim - h_roi) // 2
        square_img[start_y:start_y + h_roi, start_x:start_x + w_roi] = roi

        # Thay đổi kích thước thành 28×28
        final_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
        
        roi_images.append(final_img)
        valid_rects.append((x, y, w, h))

    return roi_images, valid_rects, thresh, img_display