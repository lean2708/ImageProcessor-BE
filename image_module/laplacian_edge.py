import cv2
import numpy as np
import logging
from PIL import Image
from image_module.encode_image import encode_image


def laplacian_edge(input_path, output_path):
    """
    Áp dụng toán tử Laplacian để phát hiện cạnh.
    :param input_path: Đường dẫn đến ảnh PNG đầu vào
    :param output_path: Đường dẫn để lưu ảnh PNG đầu ra
    :return: dictionary chứa base64 string và kích thước ảnh
    """
    try:
        img = cv2.imread(input_path, 0) # Đọc ảnh xám
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {input_path}")

        # Áp dụng Laplacian
        # Dùng cv2.CV_64F để có độ chính xác cao, ksize=3
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
        
        # Chuyển đổi lại sang uint8 (ảnh 8-bit) để lưu
        laplacian_abs = cv2.convertScaleAbs(laplacian)

        cv2.imwrite(output_path, laplacian_abs)

        (h, w) = laplacian_abs.shape[:2]
        a2 = str(encode_image(output_path))

        processed_image = {
            'base64': a2,
            'image_size': (w, h),
        }
        logging.info(f"Bộ lọc Laplacian thành công cho {input_path}")
        return processed_image

    except Exception as e:
        logging.error(f"Lỗi trong laplacian_edge: {e}")
        raise