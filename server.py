import pymodm
from pymodm import connect
from flask import Flask, jsonify, request
from flask_cors import CORS
import datetime
from skimage import exposure
import base64
import numpy as np
import uuid
import os
import math
from skimage import util
import PIL
# FIX: Đã sửa lại các import này để trỏ đúng vào hàm trong file,
# thay vì import module
from image_module.decode_image import decode_image
from image_module.log_compression import log_compression
from image_module.strip_image import strip_image
from image_module.contrast_stretching import contrast_stretching
from image_module.histogram_equalization import histogram_equalization
from image_module.reverse_video import reverse_video
from image_module.temperature import apply_temperature
from image_module.laplacian import apply_laplacian
from image_module.box_filter import apply_box_filter
from user import create_user, already_user, add_uploadimage, add_image_hist
from user import add_image_contrast, add_image_log, add_image_reverse
from models import User
import logging
import models
from PIL import Image
import time


app = Flask(__name__)
CORS(app)
connect("mongodb://localhost:27017/image_app")


INPUT_FOLDER = "input_images/"
OUTPUT_FOLDER = "output_images/"  # FIX: Thêm folder output


@app.route("/api/user_exists/<username>", methods=["GET"])
def user_exists(username):
    """
    Returns whether username is already taken
    :return: json dict of new user initial info
    :rtype: Request
    :return: 4xx error with json error dict if missing key
             or incorrect type given
    """
    user_exists = already_user(username)
    return jsonify(user_exists), 200


@app.route("/api/new_user", methods=["POST"])
def post_new_user():
    """
    Posts new user
    :return: json dict of new user initial info
    :rtype: Request
    :return: 4xx error with json error dict if missing key
             or incorrect type given
    """
    r = request.get_json()
    try:
        username = r["username"]
    except KeyError as e:
        logging.warning("Incorrect JSON input:{}".format(e))
        err = {"error": "Incorrect JSON input"}
        return jsonify(err), 400
    if already_user(username):
        u_vals = {"warning": "This user_name is already existed"}
    else:
        u_vals = create_user(username)
    return jsonify(u_vals), 200


@app.route("/api/upload", methods=["POST"])
def image_post():
    """
    Posts new user with given image
    Lưu file vào folder INPUT với tên: username_timestamp.filetype
    """
    r = request.get_json()
    try:
        username = r["username"]
        image_new = r["image"]
        file_type = r.get("file_type", "png")  # lấy file_type, mặc định png
        assert type(image_new) is str
    except KeyError as e:
        logging.warning("Incorrect JSON input:{}".format(e))
        return jsonify({"error": "Incorrect JSON input"}), 400
    except AssertionError as e:
        logging.warning("Incorrect image type given: {}".format(e))
        return jsonify({"error": "Incorrect image type given"}), 400

    # tạo tên file = username + timestamp
    timestamp = int(time.time())
    filename = f"{username}_{timestamp}.{file_type}"
    filepath = os.path.join(INPUT_FOLDER, filename)

    # giải mã base64 và lưu file vào INPUT_FOLDER
    stripped_image = strip_image(image_new, file_type)
    decode_image(stripped_image, filepath)

    # lưu thông tin user
    if not already_user(username):
        create_user(username)
    u_vals = add_uploadimage(username, filename, datetime.datetime.now())

    logging.debug(f"Saved new image {filename} for user {username}")
    return jsonify(u_vals), 200


@app.route("/api/<username>", methods=["GET"])
def get_user(username):
    """
    Get the user with the whole info
    :return: json dict of user values
    :rtype: Request
    """
    if already_user(username):
        user = models.User.objects.raw({"_id": username}).first()
        u_values = {"user_name": user.username,
                    "upload_image": user.image_original,
                    "upload_time": user.upload_time
                    }
        return jsonify(u_values), 200
    else:
        u_values = {"warning": "This user does not exist"}
        return jsonify(u_values), 200


@app.route("/api/histogram_equalization", methods=["POST"])
def histogram_equalization_processing():
    """
    Get the processed image with histogram
    :return: json dict of image
    :rtype: Request
    """
    r = request.get_json()
    try:
        username = r["username"]
        image = r["image"]
        file_type = r["file_type"]
        assert type(image) == str
    except KeyError as e:
        logging.warning("Incorrect JSON input: {}".format(e))
        err = {"error": "Incorrect JSON input"}
        return jsonify(err), 400
    except AssertionError as e:
        logging.warning("Incorrect image type given: {}".format(e))
        err = {"error": "Incorrect image type given"}
        return jsonify(err), 400
    stripped_image = strip_image(image, file_type)

    suffix = "." + file_type

    # FIX: Tạo đường dẫn file rõ ràng dùng os.path.join
    # Tạo tên file cơ sở (base filenames)
    input_base_name = str(uuid.uuid4())
    output_base_name = str(uuid.uuid4())

    # Đường dẫn cho file input tạm (ví dụ: input_images/uuid1.jpg)
    input_temp_path = os.path.join(INPUT_FOLDER, input_base_name + suffix)
    # Đường dẫn cho file input đã chuẩn hóa PNG (ví dụ: input_images/uuid1.png)
    input_png_path = os.path.join(INPUT_FOLDER, input_base_name + ".png")

    # Tên file và đường dẫn đầy đủ cho file output
    output_filename = output_base_name + ".png"
    output_full_path = os.path.join(OUTPUT_FOLDER, output_filename)

    start_time = datetime.datetime.now()
    decode_image(stripped_image, input_temp_path)
    im = Image.open(input_temp_path)
    im.save(input_png_path)

    # Gọi hàm xử lý với đường dẫn đầy đủ
    processed_image = histogram_equalization(input_png_path, output_full_path)
    
    end_time = datetime.datetime.now()
    process_time = str(end_time - start_time)

    if username != 'Visitor':
        # FIX: Chỉ lưu tên file (output_filename) vào DB
        num_hist = add_image_hist(username, output_filename, datetime.datetime.now())
        processed_image["process_count"] = num_hist

    processed_image["process_time"] = process_time

    print("returning processed image")
    return jsonify(processed_image), 200


@app.route("/api/contrast_stretching", methods=["POST"])
def contrast_stretching_processing():
    """
    Get the processed image with contrast-stretching
    :return: json dict of image
    :rtype: Request
    """
    r = request.get_json()
    try:
        username = r["username"]
        image_new = r["image"]
        file_type = r["file_type"]
        assert type(image_new) is str
    except KeyError as e:
        logging.warning("Incorrect JSON input: {}".format(e))
        err = {"error": "Incorrect JSON input"}
        return jsonify(err), 400
    except AssertionError as e:
        logging.warning("Incorrect image type given: {}".format(e))
        err = {"error": "Incorrect image type given"}
        return jsonify(err), 400
    stripped_string = strip_image(image_new, file_type)

    suffix = "." + file_type

    # FIX: Tạo đường dẫn file rõ ràng dùng os.path.join
    input_base_name = str(uuid.uuid4())
    output_base_name = str(uuid.uuid4())

    input_temp_path = os.path.join(INPUT_FOLDER, input_base_name + suffix)
    input_png_path = os.path.join(INPUT_FOLDER, input_base_name + ".png")

    output_filename = output_base_name + ".png"
    output_full_path = os.path.join(OUTPUT_FOLDER, output_filename)

    start_time = datetime.datetime.now()
    decode_image(stripped_string, input_temp_path)
    im = Image.open(input_temp_path)
    im.save(input_png_path)
    
    processed_image = contrast_stretching(input_png_path, output_full_path)
    
    end_time = datetime.datetime.now()
    process_time = str(end_time - start_time)

    if username != 'Visitor':
        # FIX: Chỉ lưu tên file (output_filename) vào DB
        num_contrast = add_image_contrast(username,
                                          output_filename, datetime.datetime.now())
        processed_image["process_count"] = num_contrast

    processed_image["process_time"] = process_time
    return jsonify(processed_image), 200


@app.route("/api/log_compression", methods=["POST"])
def log_compression_processing():
    """
    Get the processed image with log_compression
    :return: json dict of image
    :rtype: Request
    """
    r = request.get_json()
    try:
        username = r["username"]
        image_new = r["image"]
        file_type = r["file_type"]
        assert type(image_new) is str
    except KeyError as e:
        logging.warning("Incorrect JSON input: {}".format(e))
        err = {"error": "Incorrect JSON input"}
        return jsonify(err), 400
    except AssertionError as e:
        logging.warning("Incorrect image type given: {}".format(e))
        err = {"error": "Incorrect image type given"}
        return jsonify(err), 400
    stripped_string = strip_image(image_new, file_type)

    suffix = "." + file_type
    
    # FIX: Tạo đường dẫn file rõ ràng dùng os.path.join
    input_base_name = str(uuid.uuid4())
    output_base_name = str(uuid.uuid4())

    input_temp_path = os.path.join(INPUT_FOLDER, input_base_name + suffix)
    input_png_path = os.path.join(INPUT_FOLDER, input_base_name + ".png")

    output_filename = output_base_name + ".png"
    output_full_path = os.path.join(OUTPUT_FOLDER, output_filename)

    start_time = datetime.datetime.now()
    decode_image(stripped_string, input_temp_path)
    im = Image.open(input_temp_path)
    im.save(input_png_path)
    
    processed_image = log_compression(input_png_path, output_full_path)
    
    end_time = datetime.datetime.now()
    process_time = str(end_time - start_time)

    if username != 'Visitor':
        # FIX: Chỉ lưu tên file (output_filename) vào DB
        num_log = add_image_log(username, output_filename, datetime.datetime.now())
        processed_image["process_count"] = num_log

    processed_image["process_time"] = process_time
    return jsonify(processed_image), 200


@app.route("/api/reverse_video", methods=["POST"])
def reverse_video_processing():
    """
    Get the processed image with reverse_video
    :return: json dict of image
    :rtype: Request
    """
    r = request.get_json()
    try:
        username = r["username"]
        image_new = r["image"]
        file_type = r["file_type"]
        assert type(image_new) is str
    except KeyError as e:
        logging.warning("Incorrect JSON input: {}".format(e))
        err = {"error": "Incorrect JSON input"}
        return jsonify(err), 400
    except AssertionError as e:
        logging.warning("Incorrect image type given: {}".format(e))
        err = {"error": "Incorrect image type given"}
        return jsonify(err), 400
    stripped_string = strip_image(image_new, file_type)

    suffix = "." + file_type

    # FIX: Tạo đường dẫn file rõ ràng dùng os.path.join
    input_base_name = str(uuid.uuid4())
    output_base_name = str(uuid.uuid4())

    input_temp_path = os.path.join(INPUT_FOLDER, input_base_name + suffix)
    input_png_path = os.path.join(INPUT_FOLDER, input_base_name + ".png")

    output_filename = output_base_name + ".png"
    output_full_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    start_time = datetime.datetime.now()
    decode_image(stripped_string, input_temp_path)
    im = Image.open(input_temp_path)
    im.save(input_png_path)
    
    processed_image = reverse_video(input_png_path, output_full_path)
    
    end_time = datetime.datetime.now()
    process_time = str(end_time - start_time)

    if username != 'Visitor':
        # FIX: Chỉ lưu tên file (output_filename) vào DB
        num_reverse = add_image_reverse(username, output_filename, datetime.datetime.now())
        processed_image["process_count"] = num_reverse

    processed_image["process_time"] = process_time
    return jsonify(processed_image), 200


# New endpoints for temperature, laplacian, and box filter

@app.route("/api/temperature", methods=["POST"])
def temperature_processing():
    """Thay đổi nhiệt độ màu (ấm/lạnh)"""
    r = request.get_json()
    try:
        username = r["username"]
        image_new = r["image"]
        file_type = r["file_type"]
        warm = r.get("warm", True)
        intensity = int(r.get("intensity", 30))
        assert type(image_new) is str
    except KeyError as e:
        return jsonify({"error": f"Thiếu key trong JSON: {e}"}), 400

    stripped_string = strip_image(image_new, file_type)
    suffix = "." + file_type
    input_base_name = str(uuid.uuid4())
    output_base_name = str(uuid.uuid4())

    input_temp_path = os.path.join(INPUT_FOLDER, input_base_name + suffix)
    output_filename = output_base_name + ".png"
    output_full_path = os.path.join(OUTPUT_FOLDER, output_filename)

    start_time = datetime.datetime.now()
    decode_image(stripped_string, input_temp_path)
    apply_temperature(input_temp_path, intensity, warm, output_full_path)
    end_time = datetime.datetime.now()

    process_time = str(end_time - start_time)
    return jsonify({
        "message": "Xử lý nhiệt độ màu thành công!",
        "file_name": output_filename,
        "output_path": output_full_path,
        "process_time": process_time
    }), 200


@app.route("/api/laplacian", methods=["POST"])
def laplacian_processing():
    """Làm sắc nét ảnh bằng Laplacian"""
    r = request.get_json()
    try:
        username = r["username"]
        image_new = r["image"]
        file_type = r["file_type"]
        assert type(image_new) is str
    except KeyError as e:
        return jsonify({"error": f"Thiếu key trong JSON: {e}"}), 400

    stripped_string = strip_image(image_new, file_type)
    suffix = "." + file_type
    input_base_name = str(uuid.uuid4())
    output_base_name = str(uuid.uuid4())

    input_temp_path = os.path.join(INPUT_FOLDER, input_base_name + suffix)
    output_filename = output_base_name + ".png"
    output_full_path = os.path.join(OUTPUT_FOLDER, output_filename)

    start_time = datetime.datetime.now()
    decode_image(stripped_string, input_temp_path)
    apply_laplacian(input_temp_path, output_full_path)
    end_time = datetime.datetime.now()

    process_time = str(end_time - start_time)
    return jsonify({
        "message": "Làm sắc nét ảnh (Laplacian) thành công!",
        "file_name": output_filename,
        "output_path": output_full_path,
        "process_time": process_time
    }), 200


@app.route("/api/box_filter", methods=["POST"])
def boxfilter_processing():
    """Làm mờ ảnh bằng Box Filter"""
    r = request.get_json()
    try:
        username = r["username"]
        image_new = r["image"]
        file_type = r["file_type"]
        ksize = int(r.get("ksize", 5))
        assert type(image_new) is str
    except KeyError as e:
        return jsonify({"error": f"Thiếu key trong JSON: {e}"}), 400

    stripped_string = strip_image(image_new, file_type)
    suffix = "." + file_type
    input_base_name = str(uuid.uuid4())
    output_base_name = str(uuid.uuid4())

    input_temp_path = os.path.join(INPUT_FOLDER, input_base_name + suffix)
    output_filename = output_base_name + ".png"
    output_full_path = os.path.join(OUTPUT_FOLDER, output_filename)

    start_time = datetime.datetime.now()
    decode_image(stripped_string, input_temp_path)
    apply_box_filter(input_temp_path, ksize, output_full_path)
    end_time = datetime.datetime.now()

    process_time = str(end_time - start_time)
    return jsonify({
        "message": "Làm mờ ảnh (Box Filter) thành công!",
        "file_name": output_filename,
        "output_path": output_full_path,
        "process_time": process_time
    }), 200

if __name__ == "__main__":
    # FIX: Đảm bảo các thư mục tồn tại khi chạy
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    app.run()
