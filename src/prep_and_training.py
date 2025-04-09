import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Datei laden
f = h5py.File("data/usps.h5", "r")

# Daten extrahieren
train_data = f["train"]["data"][...]
train_target = f["train"]["target"][...]
test_data = f["test"]["data"][...]
test_target = f["test"]["target"][...]

# Reshapen und Typen konvertieren
X_train = train_data.reshape(-1, 16, 16, 1)
y_train = train_target.astype("int")

X_test = test_data.reshape(-1, 16, 16, 1)
y_test = test_target.astype("int")

print("X_train shape:", X_train.shape, X_train.dtype)
print("y_train shape:", y_train.shape, np.unique(y_train))

batch_size = 32

# Erstellt tensorflow Datensätze
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# mischt den Datensatz und erstellt Batches
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Erstelle den Test-Datensatz
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size)


"""
Model erstellung
"einfaches" Convolutional Neural Network mit 2 Convolutional, 2 MaxPooling und 2 Dense Layern
input Shape ist 16x16x1 (Graustufenbild) 
und output ist 10 Klassen (0-9)
maxpooling reduziert die Dimensionen der Daten um die Hälfte
2 mal conv2d bedeutet, dass 2 Convolutional Layer verwendet werden 
Mehr Filter = mehr Merkmale, die extrahiert werden
flatten bedeutet, dass die 2D-Daten in 1D umgewandelt werden
dense Layer sind vollverbundene Schichten, die alle Neuronen der vorherigen Schicht mit allen Neuronen der nächsten Schicht verbinden
"relu" ist die Aktivierungsfunktion, die nicht-lineare Transformationen anwendet
und die Aktivierungsfunktion ist softmax, die die Wahrscheinlichkeiten für jede Klasse berechnet

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(16, 16, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# Model kompilieren
# Adam ist ein Optimierer, der den Lernprozess anpasst
# sparse_categorical_crossentropy ist der Verlust, der für mehrklassige Klassifikation verwendet wird
# accuracy ist die Metrik, die verwendet wird, um die Leistung des Modells zu bewerten
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
# Early Stop einf+hren, stoppt wenn val_loss nicht mehr sinkt
# patience ist die Anzahl der Epochen, die gewartet werden, bevor das Training gestoppt wird
# restore_best_weights stellt die besten Gewichte wieder her, die während des Trainings gespeichert wurden
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

# Model trainieren
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
)

# Training Verlauf plotten
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss plotten
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss") 
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# finale Auswertung
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Model speichern
model.save("models/usps_model.h5")
print("Model saved as usps_model.h5")