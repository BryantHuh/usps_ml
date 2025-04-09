import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

# erstes Layer anzeigen lassen
def show_activations(model, sample):
    activation_model = tf.keras.Model(inputs=model.input, 
                                      outputs=model.layers[0].output)
    activations = activation_model.predict(sample.reshape(1, 16, 16, 1))
    num_filters = activations.shape[-1]
    
    plt.figure(figsize=(12, 4))
    for i in range(num_filters, 8):
        plt.subplot(1, 8, i + 1)
        plt.imshow(activations[0, :, :, i], cmap="viridis")
        plt.axis("off")
    plt.suptitle("Activations of first layer")
    plt.show()

import tensorflow as tf
import matplotlib.pyplot as plt
import time

class LiveVisualCallback(tf.keras.callbacks.Callback):
    def __init__(self, sample, model):
        super().__init__()
        self.sample = sample.reshape(1, 16, 16, 1)
        self.activation_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[0].output  # erste Conv-Schicht
        )

    def on_train_batch_end(self, batch, logs=None):
        activation = self.activation_model.predict(self.sample, verbose=0)
        num_filters = activation.shape[-1]

        plt.figure(figsize=(12, 4))
        for i in range(min(num_filters, 8)):
            plt.subplot(1, 8, i+1)
            plt.imshow(activation[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f"Aktivierungen nach Batch {batch}")
        plt.tight_layout()
        plt.show()

        # Wartezeit für „Live-Gefühl“
        time.sleep(1)


train_data = np.load("data/X_train.npy")
train_target = np.load("data/y_train.npy")
test_data = np.load("data/X_test.npy")
test_target = np.load("data/y_test.npy")

batch_size = 32

# Erstellt tensorflow Datensätze
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_target))
# mischt den Datensatz und erstellt Batches
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_target))
test_dataset = test_dataset.batch(batch_size)



# Model erstellung
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(16, 16, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# logging erstellen
log_dir = "logs/usps_model"
# Erstelle den TensorBoard Callback (Tensorboard muss im Terminal gestartet werden und im Browser geöffnet werden)
tonsorboard_cb = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)  

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    callbacks=[LiveVisualCallback(train_data[0], model)],
    # callbacks=[tonsorboard_cb],
)
