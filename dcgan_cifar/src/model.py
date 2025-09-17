import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminador  para imágenes 3x64x64.
    Conv(3x3, padding=1) mantiene tamaño; en DCGAN la reducción se hace con Conv(4x4, stride=2).
    Salida: logits para usar BCEwithLogits.
    """

    def __init__(self, img_channels=3, base=64, max_ch=1024, p_drop=0.3):

        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, max_ch  # 64, 128, 256, 1024 en numeros de filtros por conv


        # downsample DCGAN (Conv 4x4, stride 2, padding 1) en la primera capa del D.
        def conv_down(in_c, out_c, bn=True, first=False):
            """
            Bloque DCGAN:
            - Conv2d(4x4, stride=2, padding=1) menos tamaño
            - BatchNorm2d (NO en la primera capa del D)
            - LeakyReLU(0.2)
            """
            layers = []
            # En la primera capa conviene bias=True porque no hay BN; con BN, bias=False
            use_bias = first or (not bn)
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=use_bias))
            if bn and not first:
                layers.append(nn.BatchNorm2d(out_c))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(

            # 64x64 - (64 , 3x3) = (N, 64, 64, 64)
            conv_down(img_channels, c1, bn=False, first=True),  # 64 -> 32

            # 32x32 - (128 , 3x3) = (N, 128, 32, 32)
            conv_down(c1, c2, bn=True,  first=False),  # 32 -> 16

            # 16x16 - (256 , 3x3) = (N, 256, 16, 16)
            conv_down(c2, c3, bn=True,  first=False), # 16 -> 8

            # 8x8 - (1024 , 3x3) = ( N, 1024, 8, 8)
            conv_down(c3, c4, bn=True,  first=False),  # 8 -> 4
        )

        # Cabeza conv: de (N, c4, 4, 4) -> (N, 1, 1, 1) logits (sin sigmoide).
        self.classifier = nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.features(x)          # (N, c4, 4, 4)
        x = self.classifier(x)        # (N, 1, 1, 1) logits
        return x.view(x.size(0), 1)   # vector (N, 1) para BCEWithLogitsLoss


class Generator(nn.Module):
    """
    Generador tipo DCGAN para imágenes 3x64x64.
    Input: vector z (N, latent_dim)
    Output: imagen (N, 3, 64, 64)
    """
    def __init__(self, latent_dim=100, ngf=64, img_channels=3 , img_dim = (64,64)):
        super().__init__()

        self.latent_dim = latent_dim

        layers = [
            # Z = (N, latent_dim, 1, 1) a (512, 4x4) = (N, 512, 4, 4)
            nn.ConvTranspose2d(latent_dim, ngf*8, 4, 1, 0, bias=False),  # kernel=4 , pad=0 , stride=1
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            # 4 a 8 - (256, 4x4) = (N, 256, 8, 8)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),  # kernel=4 , pad=1 , stride=2 (duplica tamaño imagen)
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
            # Ultima capa sin BN, ConvT(4,2,1) y Tanh
            nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()]
        
        # Por si se usa otra configuracion que no sea 64x64 se hace AdaptiveAvgPool2d
        if tuple(img_dim) != (64, 64):
            layers.insert(-1, nn.AdaptiveAvgPool2d(img_dim))  # lo ingresamos antes de Tanh

        # Construir la red
        self.net = nn.Sequential(*layers)


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

