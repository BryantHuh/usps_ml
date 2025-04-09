import h5py
import matplotlib.pyplot as plt
import random

with  h5py.File("data/usps.h5", "r") as f:
    print("Keys: ", list(f.keys()))
    
    print("Train Strukture: ", f["train"].keys())
    print("Test Strukture: ", f["test"].keys())
    
    train_data = f["train"]["data"]
    train_target = f["train"]["target"]
    
    print("Train Data Shape: ", train_data.shape)
    print("Train Target Shape: ", train_target.shape)
    print("Train Data Type: ", train_data.dtype)
    
    print("Pixel Value: ", train_data[...].min(), train_data[...].max())
    
    random_index = random.randint(0, 7291)
    
    plt.imshow(train_data[random_index].reshape(16, 16), cmap="gray")
    plt.title(f"Label: {train_target[random_index]}")
    plt.axis("off")
    plt.show()