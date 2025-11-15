from image_module.encode_image import encode_image


def log_compression(id1, output_folder="output_images/"):
    """
    Process the input image with log_compression and save
    the processed image into the output folder with a timestamp.

    :param id1: input image file path (string)
    :param output_folder: folder để lưu ảnh output (string)
    :return: dictionary chứa base64 string của ảnh và kích thước
    """
    import os
    import time
    import logging
    from PIL import Image
    import numpy as np
    from skimage import exposure

    os.makedirs(output_folder, exist_ok=True)

    logging.basicConfig(filename='log_compression.log',
                        level=logging.DEBUG, filemode='w')

    try:
        i = np.asarray(Image.open(id1))
    except FileNotFoundError:
        logging.debug("The image file does not exist")
        raise FileNotFoundError

    try:
        assert len(i.shape) == 3
    except AssertionError:
        logging.debug("The shape of image array is not 3 layers")
        print("image numpy array is not in the right shape")

    # Apply log compression
    i_compression = exposure.adjust_log(i)
    ima = Image.fromarray(np.uint8(i_compression))
    (w, h) = ima.size

    # Tạo tên file output: gốc + kỹ thuật + timestamp
    base_name = os.path.splitext(os.path.basename(id1))[0]
    timestamp = int(time.time())
    output_filename = f"{base_name}_log_{timestamp}.png"
    output_path = os.path.join(output_folder, output_filename)

    # Lưu file output
    ima.save(output_path)
    a2 = str(encode_image(output_path))

    processed_image = {
        'base64': a2,
        'image_size': (w, h),
        'output_file': output_filename
    }
    logging.info(f"Processed image saved as {output_filename}")
    return processed_image
