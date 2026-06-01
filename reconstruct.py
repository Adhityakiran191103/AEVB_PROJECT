import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vae import VAE

# Load model
model = VAE()

model.load_state_dict(
    torch.load("vae_model_10epochs.pth")
)

model.eval()

# Load MNIST test data
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=True
)

# Get one batch
images, _ = next(iter(test_loader))

images_flat = images.view(-1, 784)

# Reconstruct
with torch.no_grad():

    reconstructions, _, _ = model(images_flat)

# Plot
fig, axes = plt.subplots(2, 10, figsize=(12, 3))

for i in range(10):

    axes[0, i].imshow(
        images[i].squeeze(),
        cmap="gray"
    )

    axes[0, i].axis("off")

    axes[1, i].imshow(
        reconstructions[i].view(28, 28),
        cmap="gray"
    )

    axes[1, i].axis("off")

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")

plt.tight_layout()

plt.savefig("reconstruction_results.png")

plt.show()