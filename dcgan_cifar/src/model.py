import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminador  para imágenes 3x64x64.
    Conv(3x3, padding=1) mantiene tamaño; solo MaxPool(2) reduce.
    Salida: probabilidad en (0,1) vía Sigmoid.
    """

    def __init__(self, img_channels=3, base=64, max_ch=1024, p_drop=0.3):

        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, max_ch  # 64, 128, 256, 1024 en numeros de filtros por conv

        def block(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)] # Same padding
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            # 64x64 - (64 , 3x3)  = (N, 64, 64, 64)
            block(img_channels, c1, bn=False),
            block(c1, c1, bn=True),
            nn.Dropout2d(p_drop),
            nn.MaxPool2d(2),  # 64 -> 32

            # 32x32 - (128 , 3x3) = (N, 128, 32, 32)
            block(c1, c2, bn=True),
            nn.Dropout2d(p_drop),
            nn.MaxPool2d(2),  # 32 -> 16

            # 16x16 - (256 , 3x3) = (N, 256, 16, 16)
            block(c2, c3, bn=True),
            nn.Dropout2d(p_drop),
            nn.MaxPool2d(2),  # 16 -> 8

            # 8x8 - (1024 , 3x3)  = ( N, 1024, 8, 8)
            block(c3, c4, bn=True),
            nn.Dropout2d(p_drop))


        self.classifier = nn.Sequential(
            nn.Flatten(), # vector (1024x8x8)
            nn.LazyLinear(1, bias=True)) # Capa automarica que detecta el input 

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class Generator(nn.Module):
    """
    Generador tipo DCGAN para imágenes 3x64x64.
    Input: vector z (N, latent_dim)
    Output: imagen (N, 3, 64, 64)
    """
    def __init__(self, latent_dim=100, ngf=64, img_channels=3 , img_dim = (64,64)):
        super().__init__()

        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # Z = (N, latent_dim, 1, 1) a (512, 4x4) = (N, 512, 4, 4)
            nn.ConvTranspose2d(latent_dim, ngf*8, 4, 1, 0, bias=False), # kernel=4 , pad = 0 , stride = 1
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            # 4 a 8 - (256, 4x4) = (N, 256, 8, 8)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), # kernel=4 , pad = 1 , stride = 2 (duplica tama;o imagen)
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
            nn.ConvTranspose2d(ngf, img_channels, 3, 2, 1, bias=False),
            nn.AdaptiveAvgPool2d(img_dim),
            nn.Tanh())

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


def weights_init_normal(m):
    name = m.__class__.__name__
    if 'Conv' in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if 'BatchNorm' in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

