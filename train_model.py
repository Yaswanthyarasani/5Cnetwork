from tensorflow.keras.optimizers import Adam
from nested_unet import nested_unet
from attention_unet import attention_unet
from data_preprocessing import preprocess_image
import numpy as np

# Load your data here
train_images, train_masks = load_data()  # Define load_data to read images/masks
train_images = np.array([preprocess_image(img) for img in train_images])

# Train Nested U-Net
model_nested = nested_unet(input_shape=(256, 256, 1))
model_nested.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model_nested.fit(train_images, train_masks, epochs=20)

# Save the trained model weights
model_nested.save_weights('model_weights/nested_unet_weights.h5')

# Train Attention U-Net
model_attention = attention_unet(input_shape=(256, 256, 1))
model_attention.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model_attention.fit(train_images, train_masks, epochs=20)

# Save the trained model weights
model_attention.save_weights('model_weights/attention_unet_weights.h5')
