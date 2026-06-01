import torch
import matplotlib.pyplot as plt

from vae import VAE

# Load trained model
model = VAE()

model.load_state_dict(
    torch.load("vae_model_10epochs.pth")
)

model.eval()

# Generate random latent vectors
z = torch.randn(16, 2)

with torch.no_grad():

    generated = model.decode(z)

# Plot
fig, axes = plt.subplots(
    4,
    4,
    figsize=(6, 6)
)

for i, ax in enumerate(axes.flat):

    ax.imshow(
        generated[i].view(28, 28),
        cmap="gray"
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "generated_digits.png"
)

plt.show()