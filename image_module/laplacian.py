import cv2
import numpy as np

def apply_laplacian(input_path, output_path="output.png"):
    """
    Làm sắc nét ảnh bằng bộ lọc Laplacian.
    """
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))

    # Chuyển về 3 kênh để lưu lại ảnh màu
    lap_color = cv2.merge([lap, lap, lap])
    cv2.imwrite(output_path, lap_color)

    return {
        "message": "Làm sắc nét ảnh (Laplacian) thành công",
        "output_path": output_path
    }
