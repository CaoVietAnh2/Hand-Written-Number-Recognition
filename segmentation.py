import cv2
import numpy as np


def read_image_safe(path):
    try: 
        stream = open(path, "rb")
        bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def merge_broken_parts(rects):
    if len(rects) < 2:
        return rects

    rects.sort(key=lambda r: r[0])
    
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
            
        x2, y2, w2, h2 = rects[i+1]
        
        center1 = x1 + w1 // 2
        center2 = x2 + w2 // 2
        
        dist_centers = abs(center1 - center2)
        dist_x = x2 - (x1 + w1)
        y_overlap = not (y1 + h1 < y2 or y2 + h2 < y1)
        x_overlap = (x2 < x1 + w1)
        
        if dist_centers < 15 or (dist_x < 4 and dist_x > -4 and y_overlap):
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1+w1, x2+w2) - new_x
            new_h = max(y1+h1, y2+h2) - new_y
            
            merged_rects.append((new_x, new_y, new_w, new_h))
            skip_next = True
            print(f"[LOGIC] Da gop 2 manh vo cua so tai vi tri X={new_x}")
        else:
            merged_rects.append((x1, y1, w1, h1))
            
    return merged_rects


def remove_inside_boxes(rects):
    valid_rects = []
    
    for i in range(len(rects)):
        x1, y1, w1, h1 = rects[i]
        is_inside = False
        
        for j in range(len(rects)):
            if i == j:
                continue
            
            x2, y2, w2, h2 = rects[j]
            
            if (x1 >= x2 and y1 >= y2 and 
                (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)):
                is_inside = True
                print(f"[LOGIC] Phat hien hop nho tai X={x1} nam trong hop to -> XOA!")
                break
                
        if not is_inside:
            valid_rects.append((x1, y1, w1, h1))
            
    return valid_rects


def segment_image(image_path):
    print(f"[INFO] Dang xu ly anh: {image_path}")
    
    img = read_image_safe(image_path)
    
    if img is None:
        raise FileNotFoundError(f"[ERROR] Cannot read the image. Please check the file path.")

    img_display = img.copy()
    
    if len(img.shape) == 3 and img.shape[2] == 4:
         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
         img_display = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- PRE-PROCESSING ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)
    
    kernel_clean = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    kernel_heal = np.ones((5,3), np.uint8) 
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_heal)

    # --- TÌM CONTOUR ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- LỌC BAN ĐẦU ---
    initial_rects = [] 
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 300 and h < 300:
            if (h > 5 and w > 5): 
                initial_rects.append((x, y, w, h))

    print(f"[INFO] Tim thay {len(initial_rects)} manh vo ban dau.")
    
    # --- GỘP BOX ---
    final_rects = merge_broken_parts(initial_rects)
    final_rects = merge_broken_parts(final_rects)
    final_rects = remove_inside_boxes(final_rects)
    
    print(f"[INFO] Sau khi gop manh vo, con lai {len(final_rects)} so hoan chinh.")

    # Sắp xếp từ trái sang phải
    final_rects.sort(key=lambda r: r[0])

    # --- CẮT & XỬ LÝ ROI ---
    roi_images = []
    valid_rects = []
    
    for (x, y, w, h) in final_rects:
        if (w * h < 150) or (h < 25 and w < 25) or (w < 8): 
            print(f"[-] Vut 1 manh vun (rac) tai X={x}")
            continue

        pad = 5
        roi = thresh[max(0, y-pad):min(thresh.shape[0], y+h+pad),
                     max(0, x-pad):min(thresh.shape[1], x+w+pad)]

        if roi.size == 0:
            continue

        # Square Padding 
        h_roi, w_roi = roi.shape
        max_dim = max(h_roi, w_roi)
        square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
        start_x = (max_dim - w_roi) // 2
        start_y = (max_dim - h_roi) // 2
        square_img[start_y:start_y+h_roi, start_x:start_x+w_roi] = roi

        final_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
        
        roi_images.append(final_img)
        valid_rects.append((x, y, w, h))

    return roi_images, valid_rects, thresh, img_display
