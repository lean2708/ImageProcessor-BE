import cv2
import numpy as np
import logging
from PIL import Image
from image_module.encode_image import encode_image


def sobel_edge(input_path, output_path):
    """
    Áp dụng toán tử Sobel (kết hợp X và Y) để phát hiện cạnh.
    :param input_path: Đường dẫn đến ảnh PNG đầu vào
    :param output_path: Đường dẫn để lưu ảnh PNG đầu ra
    :return: dictionary chứa base64 string và kích thước ảnh
    """
    try:
        logging.basicConfig(filename='encode_image.log', level=logging.DEBUG,
                        filemode='w')
        img = cv2.imread(input_path, 0) # Đọc ảnh xám
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {input_path}")

        # Áp dụng Sobel X và Y
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Kết hợp hai đạo hàm
        # Lấy giá trị tuyệt đối và kết hợp
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)
        sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

        cv2.imwrite(output_path, sobel_combined)

        (h, w) = sobel_combined.shape[:2]
        a2 = str(encode_image(output_path))

        processed_image = {
            'base64': a2,
            'image_size': (w, h),
        }
        logging.info(f"Bộ lọc Sobel thành công cho {input_path}")
        return processed_image

    except Exception as e:
        logging.error(f"Lỗi trong sobel_edge: {e}")
        raise