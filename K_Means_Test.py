import numpy as np
import os
import torch
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# 1) Load raw MNIST (uint8 tensor of shape [60000,28,28])
raw = datasets.MNIST('data', download = True)
imgs = raw.data.float() / 255.0   # now a float tensor in [0,1]

# 2) Compute statistics with built‑in tensor methods
mean = imgs.mean().item()
std  = imgs.std(unbiased=False).item()  # population std

# 3) Define your transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

mnist_data = datasets.MNIST(root='./data', transform=transform)
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=len(mnist_data))
images_tensor, _ = next(iter(data_loader))
images = images_tensor.squeeze().numpy()
def extract_patches(images, patch_size=5):
    patches = []
    for img in images:
        for i in range(0, img.shape[0] - patch_size + 1):
            for j in range(0, img.shape[1] - patch_size + 1):
                patch = img[i:i+patch_size, j:j+patch_size]
                if np.std(patch)>0:
                    patches.append(patch.flatten())
    patches = np.array(patches)
    return patches

patches = extract_patches(images, patch_size=5)

k = 5000
kmeans = KMeans(n_clusters=k)
kmeans.fit(patches)

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(kmeans.cluster_centers_[i].reshape(5, 5), cmap='gray')
    ax.axis('off')
os.makedirs("outputs", exist_ok=True)
plt.suptitle("K-means Cluster Centers of 5x5 MNIST Patches")
plt.savefig("outputs/kmeans_patch_clusters_5000_actual.png")
plt.show()
from sklearn.metrics import pairwise_distances_argmin_min

closest_indices, distances = pairwise_distances_argmin_min(patches, kmeans.cluster_centers_)

average_distance = distances.mean()
print(f"Average Euclidean distance to cluster centers 5000: {average_distance:.4f}")

k = 1000
kmeans = KMeans(n_clusters=k)
kmeans.fit(patches)

fig, axes = plt.subplots(20, 25, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
     ax.imshow(kmeans.cluster_centers_[i].reshape(5, 5), cmap='gray')
     ax.axis('off')
plt.suptitle("K-means Cluster Centers of 5x5 MNIST Patches")
plt.savefig("outputs/kmeans_patch_clusters_1000.png")
plt.show()

closest_indices, distances = pairwise_distances_argmin_min(patches, kmeans.cluster_centers_)

average_distance = distances.mean()
print(f"Average Euclidean distance to cluster centers 1000: {average_distance:.4f}")


