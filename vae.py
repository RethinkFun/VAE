import os
from datetime import datetime
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

# ========== Paths ==========
data_root = "/mnt/d/data/face/img/img_align_celeba"
save_dir = "./vae_test_checkpoints"
os.makedirs(save_dir, exist_ok=True)

# ========== Hyperparams ==========
batch_size = 1024
lr = 2e-5
num_epochs = 300
latent_dim = 128
sample_every = 5
num_sample_images = 8
image_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])


# ========== Dataset ==========
class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


dataset = CelebADataset(root=data_root, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=8, pin_memory=True)


# ========== Model ==========
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, ch=64, image_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = ch
        self.image_size = image_size

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat_dim = self.enc(dummy).view(1, -1).size(1)

        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, feat_dim)
        self._feat_shape = self.enc(dummy).shape[1:]

        # Decoder with Sigmoid output
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),
            nn.Sigmoid()  # 添加Sigmoid激活
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), *self._feat_shape)
        x_recon = self.dec(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# ========== Loss ==========
def vae_loss(recon_x, x, mu, logvar):
    # 使用 MSE 重建损失
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld


# ========== Utilities ==========
def save_checkpoint(model, optim, epoch, path):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict()
    }
    torch.save(state, path)


def save_image_grid(tensor, filename, nrow=8):
    tensor = torch.clamp(tensor, 0, 1)
    utils.save_image(tensor, filename, nrow=nrow, padding=2)


# ========== Training ==========
def train():
    model = ConvVAE(latent_dim=latent_dim,image_size=image_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0
    for epoch in range(136, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0

        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            optim.zero_grad()
            recon_imgs, mu, logvar = model(imgs)
            loss, recon_l, kld = vae_loss(recon_imgs, imgs, mu, logvar)
            loss.backward()
            optim.step()

            batch_size = imgs.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_recon += recon_l.item() * batch_size
            epoch_kld += kld.item() * batch_size
            global_step += 1

            if batch_idx % 100 == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Epoch {epoch}/{num_epochs} Batch {batch_idx}/{len(loader)} "
                      f"Loss {loss.item():.4f} "
                      f"(recon {recon_l.item():.4f}, kld {kld.item():.4f})")

        n_samples = len(loader.dataset)
        print(f"=== Epoch {epoch} finished. Avg loss: {epoch_loss / n_samples:.4f} "
              f"(recon {epoch_recon / n_samples:.4f}, kld {epoch_kld / n_samples:.4f}) ===")

        # 保存样本
        if epoch % sample_every == 0 or epoch == 1:
            # 保存检查点
            ckpt_path = os.path.join(save_dir, f"vae_epoch{epoch}.pth")
            save_checkpoint(model, optim, epoch, ckpt_path)
            model.eval()
            with torch.no_grad():
                # 重建样本
                imgs = next(iter(loader))
                imgs = imgs.to(device)[:num_sample_images]
                recon_imgs, _, _ = model(imgs)
                combined = torch.cat([imgs, recon_imgs], dim=0)  # 不再需要clamp
                save_image_grid(combined, os.path.join(save_dir, f"recon_epoch{epoch}.png"), nrow=8)

                # 生成样本
                z = torch.randn(num_sample_images, latent_dim).to(device)
                samples = model.decode(z)
                save_image_grid(samples, os.path.join(save_dir, f"sample_epoch{epoch}.png"), nrow=8)
            model.train()

    print("Training complete.")


if __name__ == "__main__":
    print("Starting training on device:", device)
    print("Dataset size:", len(dataset))
    train()
