import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras import regularizers  # For L2 Regularization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This forces TensorFlow to use CPU only

# Clear previous sessions
tf.keras.backend.clear_session()

# Allow GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Using GPU: {[device.name for device in physical_devices]}")
else:
    print("No GPU found. Using CPU.")

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001  # You can adjust this value , 0.01 learning rate is good for beginning first 50 epochs.. maybe..
L2_REG = 0.0001  # L2 regularization strength
TRAIN_DIR = 'c:/venvs/screenshots scoreboard'  # Path to your training data
SAVE_MODEL_PATH = 'vgg19_model_3.keras'
SAVE_MODEL_PATH2 = 'vgg19_model_3.keras'

# Custom callback for logging
class LoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        print(f" - Loss: {logs['loss']:.4f}")
        print(f" - Accuracy: {logs['accuracy']:.4f}")
        print(f" - Validation Loss: {logs['val_loss']:.4f}")
        print(f" - Validation Accuracy: {logs['val_accuracy']:.4f}")

# Load and modify the VGG19 model
def create_model():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = base_model.output
    x = Flatten()(x)
    
    # Adding Dense layer with L2 regularization
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    
    # First Dropout layer
    x = Dropout(0.5)(x)
    
    # Adding another Dense layer with L2 regularization
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    
    # Second Dropout layer
    x = Dropout(0.5)(x)
    
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze the layers of VGG19

    # Compile the model with custom learning rate
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Data augmentation with validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)


# Load training data with multiprocessing
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True  # Make sure it's shuffled to mix augmented with real images
)

validation_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
)


# Example of adding Batch Normalization to the loaded model
def add_batch_normalization(model):
    # Create a new input for the model
    new_model_input = model.input
    x = new_model_input

    # Create a list to hold the layers of the new model
    new_layers = []

    for layer in model.layers:
        # Skip the input layer
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        x = layer(x)  # Apply the layer to the input/output
        new_layers.append(layer)  # Keep track of layers

        # Add Batch Normalization after Conv2D or Dense layers
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            x = tf.keras.layers.BatchNormalization()(x)

    # Create a new model with the inputs and outputs
    new_model = tf.keras.models.Model(inputs=new_model_input, outputs=x)
    
    return new_model



# Load the model if it exists
if os.path.exists(SAVE_MODEL_PATH):
    model = tf.keras.models.load_model(SAVE_MODEL_PATH)  # Load the existing model
    print("Model loaded from disk.")

    # Add Batch Normalization layers after Dense layers
    #model = add_batch_normalization(model)
    #print("Batch Normalization layers added.")
    
    # Create a new optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    print("Model optimizer initalized")

    # Define the loss function (e.g., binary classification)
    loss_function = 'binary_crossentropy'  # Use the appropriate loss function here
    
    # Compile the model with the new optimizer
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    print("Compiled Model loaded from disk with new optimizer learning rate.")
    # Evaluate the model on the validation set to build metrics
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
else:
    model = create_model()  # Create a new model if it doesn't exist
    # Add Batch Normalization layers after Dense layers
    model = add_batch_normalization(model)
    print("Batch Normalization layers added.")
    # Create a new optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    print("Model optimizer initalized")

    # Define the loss function (e.g., binary classification)
    loss_function = 'binary_crossentropy'  # Use the appropriate loss function here
    
    # Compile the model with the new optimizer
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    print("Compiled Model created from disk with new optimizer learning rate.")
     
# Setup TensorBoard and ModelCheckpoint callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

checkpoint_callback = ModelCheckpoint(
    filepath='newbest2_model.keras',   # Path to save the best model
    monitor='val_loss',               # Monitor the validation loss
    save_best_only=True,              # Save only the best model
    mode='min',                       # Save the model with the minimum validation loss
    verbose=1                         # Verbosity mode, 1 for progress messages
)


# Callbacks for learning rate reduction and early stopping
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0000625000029685907, verbose=1)


early_stopping = EarlyStopping(
    monitor='val_loss',      # Monitor validation loss
    patience=50,              # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the best epoch
)


# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[LoggingCallback(), tensorboard_callback, checkpoint_callback, reduce_lr]
)

# Save the final model
model.save(SAVE_MODEL_PATH2)
print("Model trained and saved.")
