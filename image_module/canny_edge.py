import cv2
import logging
from PIL import Image
from image_module.encode_image import encode_image


def canny_edge(input_path, output_path):
    """
    Áp dụng thuật toán Canny để phát hiện cạnh.
    Sử dụng ngưỡng 100 và 200 từ ví dụ của bạn.
    :param input_path: Đường dẫn đến ảnh PNG đầu vào
    :param output_path: Đường dẫn để lưu ảnh PNG đầu ra
    :return: dictionary chứa base64 string và kích thước ảnh
    """
    try:
        # Đọc ảnh, 0 nghĩa là đọc ảnh xám
        img = cv2.imread(input_path, 0) 
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {input_path}")

        # Áp dụng Canny, 100 và 200 là ngưỡng min và max
        edges = cv2.Canny(img, 100, 200)

        # Lưu ảnh đã xử lý
        cv2.imwrite(output_path, edges)

        # Lấy kích thước ảnh
        (h, w) = edges.shape[:2]

        # Mã hóa ảnh output sang base64
        a2 = str(encode_image(output_path))

        processed_image = {
            'base64': a2,
            'image_size': (w, h),
        }
        logging.info(f"Phát hiện cạnh Canny thành công cho {input_path}")
        return processed_image

    except Exception as e:
        logging.error(f"Lỗi trong canny_edge: {e}")
        raise