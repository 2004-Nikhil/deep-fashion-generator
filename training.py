from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # scale to [-1, 1] for Tanh
])

dataset = datasets.FashionMNIST(root='./content/data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2,pin_memory=True)



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import Generator, Discriminator, weights_init
import os

# Hyperparameters (from original DCGAN paper)
NOISE_DIM  = 100
BETA1      = 0.5       # Adam momentum — important for stability
BATCH_SIZE = 128
EPOCHS     = 50
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# AFTER — save to Drive so files survive session restarts
os.makedirs('/content/drive/MyDrive/dcgan_outputs', exist_ok=True)
os.makedirs('/content/drive/MyDrive/dcgan_checkpoints', exist_ok=True)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset    = datasets.FashionMNIST('./content/data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Models
G = Generator(NOISE_DIM).to(DEVICE)
D = Discriminator().to(DEVICE)
G.apply(weights_init)
D.apply(weights_init)

criterion  = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(BETA1, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(BETA1, 0.999))

fixed_noise = torch.randn(64, NOISE_DIM, device=DEVICE)  # for visualization

G_losses, D_losses = [], []

for epoch in range(EPOCHS):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(DEVICE)
        b = real_imgs.size(0)

        real_labels = torch.ones(b, 1, device=DEVICE)
        fake_labels = torch.zeros(b, 1, device=DEVICE)

        # --- Train Discriminator ---
        opt_D.zero_grad()
        d_real = D(real_imgs)
        loss_d_real = criterion(d_real, real_labels)

        z = torch.randn(b, NOISE_DIM, device=DEVICE)
        fake_imgs = G(z).detach()  # detach: don't backprop through G
        d_fake = D(fake_imgs)
        loss_d_fake = criterion(d_fake, fake_labels)

        loss_D = loss_d_real + loss_d_fake
        loss_D.backward()
        opt_D.step()

        # --- Train Generator ---
        opt_G.zero_grad()
        z = torch.randn(b, NOISE_DIM, device=DEVICE)
        fake_imgs = G(z)
        d_fake = D(fake_imgs)
        loss_G = criterion(d_fake, real_labels)  # G wants D to say "real"
        loss_G.backward()
        opt_G.step()
        #second pass
        opt_G.zero_grad()
        z = torch.randn(b, NOISE_DIM, device=DEVICE)
        fake_imgs = G(z)
        loss_G2 = criterion(D(fake_imgs), real_labels)
        loss_G2.backward()
        opt_G.step()

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

    # Save sample images every epoch
    with torch.no_grad():
        samples = G(fixed_noise)
        save_image(samples, f'/content/drive/MyDrive/dcgan_outputs/epoch_{epoch+1:03d}.png',
                   nrow=8, normalize=True)

    print(f"Epoch {epoch+1}/{EPOCHS} | D loss: {loss_D.item():.4f} | G loss: {loss_G.item():.4f}")

    # Save checkpoints every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(G.state_dict(), f'/content/drive/MyDrive/dcgan_checkpoints/G_epoch{epoch+1}.pth')
        torch.save(D.state_dict(), f'/content/drive/MyDrive/dcgan_checkpoints/D_epoch{epoch+1}.pth')


import matplotlib.pyplot as plt
import torch

def plot_losses(G_losses, D_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(G_losses, label='Generator Loss', alpha=0.7)
    plt.plot(D_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('DCGAN Training Losses — Fashion MNIST')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/dcgan_outputs/loss_curve.png', dpi=150)
    plt.show()

def generate_grid(generator, noise_dim=100, n=64, device='cpu'):
    from torchvision.utils import make_grid
    import numpy as np
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n, noise_dim, device=device)
        imgs = generator(z)
        grid = make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Generated Fashion Items')
    plt.savefig('/content/drive/MyDrive/dcgan_outputs/final_grid.png', dpi=150, bbox_inches='tight')
    plt.show()


plot_losses(G_losses, D_losses)
generate_grid(G, noise_dim=NOISE_DIM, n=64, device=DEVICE)


# Add this after save_image() calls to see results directly in the notebook
from IPython.display import display, Image
display(Image(f'/content/drive/MyDrive/dcgan_outputs/epoch_{epoch+1:03d}.png'))
