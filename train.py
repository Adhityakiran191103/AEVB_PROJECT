import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae import VAE
def loss_function(recon_x, x, mu, logvar):

    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x,
        x.view(-1, 784),
        reduction="sum"
    )

    KLD = -0.5 * torch.sum(
        1 + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return BCE + KLD

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

# Model
model = VAE().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

epoch_losses = []

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):

        data = data.view(-1, 784).to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        loss = loss_function(
            recon_batch,
            data,
            mu,
            logvar
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataset)

    epoch_losses.append(avg_loss)

    print(
        f"Epoch {epoch+1}/{epochs} | Average Loss = {avg_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "vae_model_10epochs.pth"
)

print("Improved model saved!")

import matplotlib.pyplot as plt

plt.plot(epoch_losses)

plt.xlabel("Epoch")
plt.ylabel("Average Loss")

plt.title("Training Loss Curve")

plt.grid(True)

plt.savefig("training_curve.png")

plt.show()