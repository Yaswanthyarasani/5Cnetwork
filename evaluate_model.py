from nested_unet import nested_unet
from attention_unet import attention_unet
from tensorflow.keras.metrics import MeanIoU

# Load the test set here
test_images, test_masks = load_test_data()

# Load models
model_nested = nested_unet(input_shape=(256, 256, 1))
model_nested.load_weights('model_weights/nested_unet_weights.h5')

model_attention = attention_unet(input_shape=(256, 256, 1))
model_attention.load_weights('model_weights/attention_unet_weights.h5')

# Evaluate models
dice_nested = model_nested.evaluate(test_images, test_masks)
dice_attention = model_attention.evaluate(test_images, test_masks)

print("DICE Score Nested U-Net:", dice_nested)
print("DICE Score Attention U-Net:", dice_attention)
