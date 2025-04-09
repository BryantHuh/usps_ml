
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- 1. Modell und Daten laden ---
import os

model_path = "models/usps_model.h5"
if not os.path.exists(model_path):
	raise FileNotFoundError(f"Model file not found at {model_path}")
model = tf.keras.models.load_model(model_path)

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# --- 2. Vorhersagen erzeugen ---
# Ausgabe: Wahrscheinlichkeiten für jede Klasse
y_pred_prob = model.predict(X_test, verbose=0)
# Die vorhergesagte Klasse = Index mit höchster Wahrscheinlichkeit
y_pred = np.argmax(y_pred_prob, axis=1)

# --- 3. Confusion Matrix erzeugen ---
cm = confusion_matrix(y_test, y_pred)

# --- 4. Visualisieren ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix – USPS Ziffernerkennung")
plt.show()
