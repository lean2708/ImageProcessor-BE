import cv2

def apply_box_filter(input_path, ksize=5, output_path="output.png"):
    """
    Làm mờ ảnh bằng Box Filter.
    ksize: kích thước kernel (mặc định 5)
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {input_path}")

    # Áp dụng box filter
    blurred = cv2.boxFilter(image, -1, (ksize, ksize))
    cv2.imwrite(output_path, blurred)

    return {
        "message": "Làm mờ ảnh (Box Filter) thành công",
        "output_path": output_path
    }
