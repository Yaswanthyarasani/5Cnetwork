from tensorflow.keras import layers, models

def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def nested_unet(input_shape):
    inputs = layers.Input(input_shape)
    
    # Encoder (downsampling path)
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Decoder (upsampling path)
    u1 = layers.UpSampling2D((2, 2))(c2)
    u1 = layers.concatenate([u1, c1])
    c3 = conv_block(u1, 64)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c3)
    
    model = models.Model(inputs, outputs)
    return model
