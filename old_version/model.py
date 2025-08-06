# src/model.py
import tensorflow as tf

def unet_model(input_shape=(128, 128, 1)):
    """
    Define U-Net model.
    
    Parameters:
    - input_shape (tuple): Input shape of the model.
    
    Returns:
    - tf.keras.Model: U-Net model.
    """
    # Input layer
    inputs = tf.keras.layers.Input(input_shape)
    
    # Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    # ... continue with the rest of the U-Net architecture
    
    # Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    # ... continue with the rest of the U-Net architecture

    # Output layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model
