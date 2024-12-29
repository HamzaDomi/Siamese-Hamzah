import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2

# # Define the model (U-Net like architecture)
# def build_enhancement_model(input_shape):
#     inputs = Input(shape=input_shape)

#     # Encoder
#     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)

#     # Bottleneck
#     c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)

#     # Decoder
#     c4 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(c3)
#     c5 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(c4)

#     outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(c5)

#     model = Model(inputs, outputs)
#     return model


# input_shape = (1024, 1024, 3)  # Example input shape (Height, Width, Channels)
# # Compile the model
# model = build_enhancement_model(input_shape)
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# # Load and preprocess data


# def preprocess_images(fullpath,image_list, target_size):
#     processed_images = []
#     for img_path in image_list:
#         file_path = os.path.join(fullpath, img_path)
#         img = tf.keras.preprocessing.image.load_img(file_path, target_size=target_size)
#         img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
#         processed_images.append(img_array)
#     return np.array(processed_images)

# # Example lists of file paths for training
# # input_images_list = ['path_to_input_image1.jpg', 'path_to_input_image2.jpg']
# # enhanced_images_list = ['path_to_enhanced_image1.jpg', 'path_to_enhanced_image2.jpg']

# input_images_list = [f for f in os.listdir('/myData/Reconstructed')]
# enhanced_images_list = [f for f in os.listdir('/myData/original')]

# # Preprocess images
# # Load and preprocess data
# input_images = preprocess_images('/myData/Reconstructed',input_images_list, target_size=(1024, 1024))
# enhanced_images = preprocess_images('/myData/original',enhanced_images_list, target_size=(1024, 1024))


# model.fit(
#     input_images,
#     enhanced_images,
#     epochs=50,  # Increase epochs for better learning with a small dataset
#     batch_size=2,  # Small batch size to manage memory
#     validation_split=0.2,
#     callbacks=[tf.keras.callbacks.ModelCheckpoint('enhancement_checkpoint.keras', save_best_only=True)]
# )

# # Save the final model
# model.save('final_image_enhancement_model.keras')


def enhance_image(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(1024, 1024))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    enhanced_img = model.predict(img_array)[0]  # Remove batch dimension
    return (enhanced_img * 255).astype(np.uint8)  # Rescale to 0-255

def EnahnceAllImages(model,Reconstructed_path,EnahncedReconstructed_path):
    for file_name in os.listdir(Reconstructed_path):
        Reconstructed_img_path = os.path.join(Reconstructed_path, file_name)
        Enhanced_image = enhance_image(model, Reconstructed_img_path)
        EnahncedReconstructed_img_path = os.path.join(EnahncedReconstructed_path, file_name)
        # cv2.imwrite(EnahncedReconstructed_img_path, Enhanced_image)
        tf.keras.preprocessing.image.save_img(EnahncedReconstructed_img_path, Enhanced_image)

model = tf.keras.models.load_model('./models/final_image_enhancement_model.keras')
EnahnceAllImages(model,"./assets/Reconstructed","./assets/EnahncedReconstructed")

#tf.keras.preprocessing.image.save_img('./new.jpg', output_image)