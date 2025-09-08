import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminador para imágenes 3x64x64.
    Ahora usamos Conv(stride=2) para reducir tamaño (mejor que MaxPool en GANs).
    Salida: logits (usa BCEWithLogitsLoss), NO aplicar Sigmoid aquí.
    """
    def __init__(self, img_channels=3, base=96, max_ch=768 , p_drop=0.1 , hinge = True):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, max_ch  # 96, 192, 384, 768

        if hinge:
          def dblock(in_c, out_c, stride=2):
              # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
              return nn.Sequential(
                  SN(nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=True)),

                  nn.LeakyReLU(0.2, inplace=True))
        else:
          def dblock(in_c, out_c, stride=2):
              # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
              return nn.Sequential(
                  nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=True),
                  nn.LeakyReLU(0.2, inplace=True))


        self.features = nn.Sequential(
            # 64x64 - (96 , k=4,s=2,p=1)  = (N, 96, 32, 32)
            dblock(img_channels, c1, stride=2),

            # 32x32 - (192 , k=4,s=2,p=1) = (N, 192, 16, 16)
            dblock(c1, c2, stride=2),

          # 16x16 - (384 , k=4,s=2,p=1) = (N, 384,  8,  8)
            dblock(c2, c3, stride=2),

            #  8x8  - (768 , k=4,s=2,p=1) = (N, 768,  4,  4)
            dblock(c3, c4, stride=2))

        self.classifier = SN(nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=0, bias=True)) # (N, 768, 4, 4) -> (N, 1, 1, 1)


    def forward(self, x):
        # x: (N, 3, 64, 64)
        x = self.features(x)  # -> (N, 768, 4, 4)
        x = self.classifier(x) # -> (N, 1, 1, 1)
        return x.view(x.size(0), 1)  # -> (N, 1) logits


class Generator(nn.Module):

    """
    Generador tipo DCGAN para imágenes 3x64x64.
    Input: vector z (N, latent_dim)
    Output: imagen (N, 3, 64, 64)
    """

    def __init__(self, latent_dim=100, ngf=128, img_channels=3 , img_dim = (64,64)):
        super().__init__()

        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # Z = (N, latent_dim, 1, 1) a (512, 4x4) = (N, 512, 4, 4)
            nn.ConvTranspose2d(latent_dim, ngf*8, 4, 1, 0, bias=False), # kernel=4 , stride = 1 pad = 0
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            # 4 a 8 - (256, 4x4) = (N, 256, 8, 8)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), # kernel=4 , , stride = 2 , pad = 1  (duplica tama;o imagen)
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # 8 a 16 - (128, 4x4) = (N, 128, 16, 16)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # 16 a 32 - (64, 4x4) = (N, 64, 32, 32)
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 32 a 64 - (3, 4x4)= (N, 3, 64, 64)
            nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


def weights_init_dcgan_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if hasattr(m, 'weight_orig'):
            nn.init.normal_(m.weight_orig, 0.0, 0.02)
        else:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

def weights_init_normal(m):
    name = m.__class__.__name__
    if 'Conv' in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if 'BatchNorm' in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

