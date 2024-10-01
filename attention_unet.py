from tensorflow.keras import layers, models

def attention_block(x, g, inter_shape):
    theta_x = layers.Conv2D(inter_shape, (1, 1))(x)
    phi_g = layers.Conv2D(inter_shape, (1, 1))(g)
    f = layers.add([theta_x, phi_g])
    f = layers.Activation('relu')(f)
    psi_f = layers.Conv2D(1, (1, 1), activation='sigmoid')(f)
    return layers.multiply([x, psi_f])

def attention_unet(input_shape):
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), padding='same')(p1)
    
    # Attention block
    g = layers.Conv2D(64, (1, 1))(c2)
    attn1 = attention_block(c1, g, 64)
    
    # Decoder
    u1 = layers.UpSampling2D((2, 2))(g)
    u1 = layers.concatenate([u1, attn1])
    c3 = layers.Conv2D(64, (3, 3), padding='same')(u1)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c3)
    
    model = models.Model(inputs, outputs)
    return model
