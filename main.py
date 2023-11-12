import os
import sys
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# Define paths to your style transfer models
style_predict_path = resource_path("./model/tflite/magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite")
style_transform_path = resource_path("./model/tflite/magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite")

# Load your style transfer models here
style_predict_model = tf.lite.Interpreter(model_path=style_predict_path)
style_transform_model = tf.lite.Interpreter(model_path=style_transform_path)

# Function to preprocess an image by resizing and central cropping it.
def preprocess_image(image, target_dim):
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
    return image

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_dim):
    # Load an image from a file and add a batch dimension
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    
    # Preprocess the image
    shape = tf.cast(tf.shape(img)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = tf.image.resize_with_crop_or_pad(img, target_dim, target_dim)
    
    return img

# Function to run style prediction
def run_style_predict(preprocessed_style_image, style_predict_path):
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    return style_bottleneck

# Function to run style transform
def run_style_transform_v2(style_bottleneck, preprocessed_content_image, style_transform_path):
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()
    stylized_image = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    return stylized_image

# API endpoint to apply style transfer
@app.route('/apply_style_transfer', methods=['POST'])
def apply_style_transfer():
    temp_directory = resource_path('./tmp')
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    try:
        if 'content_image' not in request.files or 'style_image' not in request.files:
            return jsonify({'error': 'Both content_image and style_image files are required'}), 400

        content_file = request.files['content_image']
        style_file = request.files['style_image']

        # Ensure file extensions are allowed (e.g., .jpg, .png)
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not (content_file.filename.lower().split('.')[-1] in allowed_extensions and
                style_file.filename.lower().split('.')[-1] in allowed_extensions):
            return jsonify({'error': 'Unsupported file format. Supported formats: .jpg, .jpeg, .png'}), 400

        # content_blending_ratio = 0.6
        content_blending_ratio = float(request.form.get('content_blending_ratio', 0.6))

        # Save the uploaded files to a temporary directory
        content_filename = secure_filename(content_file.filename)
        style_filename = secure_filename(style_file.filename)

        content_path = os.path.join(temp_directory, content_filename)
        style_path = os.path.join(temp_directory, style_filename)

        content_file.save(content_path)
        style_file.save(style_path)

        # Load and preprocess the input images
        preprocessed_content_image = load_and_preprocess_image(content_path, 384)
        preprocessed_style_image = load_and_preprocess_image(style_path, 256)
        style_bottleneck = run_style_predict(preprocessed_style_image, style_predict_path)

        content_image = load_and_preprocess_image(content_path, 256)

        style_bottleneck_content = run_style_predict(preprocess_image(content_image, 256), style_predict_path)

        style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (1 - content_blending_ratio) * style_bottleneck
        # Stylize the content image using the style bottleneck
        stylized_image = run_style_transform_v2(style_bottleneck_blended, preprocessed_content_image, style_transform_path)

        # Save the stylized image to a temporary file
        stylized_image_path = os.path.join(temp_directory, 'stylized_image.jpg')
        stylized_image = tf.image.encode_jpeg(tf.cast(stylized_image[0] * 255, tf.uint8))
        tf.io.write_file(stylized_image_path, stylized_image)

        # Clean up temporary files
        os.remove(content_path)
        os.remove(style_path)

        # Return the stylized image file as a response
        return send_file(stylized_image_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create a temporary directory for file storage
    os.makedirs('temp', exist_ok=True)
    app.run(debug=True, port=8000)
