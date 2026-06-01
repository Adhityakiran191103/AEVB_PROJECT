import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vae import VAE

# Load trained model
model = VAE()

model.load_state_dict(
    torch.load("vae_model_10epochs.pth")
)

model.eval()

# Load test data
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

latent_points = []
labels = []

with torch.no_grad():

    for images, targets in test_loader:

        images = images.view(-1, 784)

        mu, logvar = model.encode(images)

        latent_points.append(mu)

        labels.append(targets)

latent_points = torch.cat(latent_points)
labels = torch.cat(labels)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    latent_points[:,0],
    latent_points[:,1],
    c=labels,
    cmap="tab10",
    s=5
)

plt.colorbar(scatter)

plt.title("VAE Latent Space")

plt.savefig("latent_space.png")

plt.show()