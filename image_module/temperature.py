import cv2
import numpy as np

def apply_temperature(input_path, intensity=30, warm=True, output_path="output.png"):
    """
    Thay đổi nhiệt độ màu của ảnh.
    warm=True  →  Ảnh ấm hơn (vàng)
    warm=False →  Ảnh lạnh hơn (xanh)
    intensity: độ mạnh của hiệu ứng (mặc định 30)
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {input_path}")

    # Chuyển sang float để tránh tràn khi cộng/trừ
    result = image.astype(np.float32)

    if warm:
        result[:, :, 2] += intensity   # Tăng kênh đỏ
        result[:, :, 1] += intensity / 2  # Tăng nhẹ kênh xanh lá
    else:
        result[:, :, 0] += intensity   # Tăng kênh xanh dương
        result[:, :, 2] -= intensity / 2  # Giảm nhẹ kênh đỏ

    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, result)

    return {
        "message": "Thay đổi nhiệt độ màu thành công",
        "output_path": output_path
    }
