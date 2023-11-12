import tensorflow as tf
import matplotlib.pyplot as plt

# Paths to input images and style models
content_path = "./images/image.jpg"
style_path = "./images/style-demo.jpeg"
style_predict_path = "./model/tflite/magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite"
style_transform_path = "./model/tflite/magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite"


# Function to preprocess an image by resizing and central cropping it.
def preprocess_image(image, target_dim):
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
    return image

def load_and_preprocess_image(image_path, target_dim):
    """Function to load and preprocess an image"""
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image[tf.newaxis, :]

    # Resize and central crop the image
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image

# Load and preprocess the input images
preprocessed_content_image = load_and_preprocess_image(content_path, 384)
preprocessed_style_image = load_and_preprocess_image(style_path, 256)

# Function to display images
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

# Display the input images
plt.subplot(1, 2, 1)
imshow(preprocessed_content_image, 'Content Image')
plt.subplot(1, 2, 2)
imshow(preprocessed_style_image, 'Style Image')
plt.show()

# Function to run style prediction
def run_style_predict(preprocessed_style_image, style_predict_path):
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    return style_bottleneck

# Calculate style bottleneck for the preprocessed style image
style_bottleneck = run_style_predict(preprocessed_style_image, style_predict_path)

# Function to run style transform
def run_style_transform(style_bottleneck, preprocessed_content_image, style_transform_path):
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()
    stylized_image = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    return stylized_image

# Stylize the content image using the style bottleneck
# stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image, style_transform_path)

# # Visualize the output
# imshow(stylized_image, 'Stylized Image')
# plt.show()

content_image = load_and_preprocess_image(content_path, 256)

# Calculate style bottleneck of the content image
style_bottleneck_content = run_style_predict(preprocess_image(content_image, 256), style_predict_path)

# Define content blending ratio between [0..1]
content_blending_ratio = 0.6  # You can change this value

# Blend the style bottleneck of style image and content image
style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (1 - content_blending_ratio) * style_bottleneck

# Stylize the content image using the style bottleneck
stylized_image_blended = run_style_transform(style_bottleneck_blended, preprocessed_content_image, style_transform_path)

# Visualize the output
imshow(stylized_image_blended, 'Blended Stylized Image')
plt.show()
