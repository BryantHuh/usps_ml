import h5py
import numpy as np

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


# daten als npy array speichern
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

