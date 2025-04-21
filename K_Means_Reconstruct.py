import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# Load MNIST (for unnormalized images)
raw = datasets.MNIST('data', download=True)
imgs = raw.data.float().numpy() / 255.0  # normalized to [0,1]
labels = raw.targets.numpy()

# Load saved cluster centers for K=5000
centers_10000 = np.load("outputs/kmeans_centers_10000.npy") #Load appropriate cluster file for reconstruction 

# Create mock KMeans object with cluster_centers_
class DummyKMeans:
    def __init__(self, cluster_centers):
        self.cluster_centers_ = cluster_centers

    def predict(self, X):
        return pairwise_distances_argmin(X, self.cluster_centers_)

kmeans = DummyKMeans(centers_10000)

# Reconstruction function
def reconstruct_image(img, kmeans, patch_size=5):
    H, W = img.shape
    reconstructed = np.zeros((H, W))
    weight = np.zeros((H, W))

    for i in range(0, H - patch_size + 1):
        for j in range(0, W - patch_size + 1):
            patch = img[i:i+patch_size, j:j+patch_size]
            if np.std(patch) == 0:
                continue
            patch_flat = patch.flatten().reshape(1, -1)
            nearest = kmeans.predict(patch_flat)[0]
            reconstructed_patch = kmeans.cluster_centers_[nearest].reshape(patch_size, patch_size)

            reconstructed[i:i+patch_size, j:j+patch_size] += reconstructed_patch
            weight[i:i+patch_size, j:j+patch_size] += 1

    weight[weight == 0] = 1
    return reconstructed / weight

# Get indices of first occurrence of each digit 0–9
digit_indices = [np.where(labels == d)[0][0] for d in range(10)]

# Plot side-by-side
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i, idx in enumerate(digit_indices):
    original = imgs[idx]
    reconstructed = reconstruct_image(original, kmeans)

    axes[0, i].imshow(original, cmap='gray')
    axes[0, i].set_title(f"Digit: {i}")
    axes[0, i].axis('off')

    axes[1, i].imshow(reconstructed, cmap='gray')
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis('off')

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/reconstructed_digits_0_to_9_k10000.png")
plt.show()
