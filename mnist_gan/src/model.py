import torch.nn as nn
import torch

class Generator(nn.Module):

    """
    Generador MLP estilo Goodfellow (2014) para imágenes 1x32x32 (1024 pixeles).
    - Entrada: z ~ N(0, I), dim=latent_dim
    - Capas ocultas: FC + (BatchNorm1d opcional) + ReLU
    - Salida: FC -> Tanh, reshape a (B, 1, 32, 32)
    """
    
    def __init__(self, latent_dim = 100, img_channels = 1, img_size = 32,
                 hidden_dims=(256, 512, 1024) , batch_norm=True):
        
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.out_dim = img_channels * img_size * img_size

        layers = []
        in_dim = latent_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h, bias=True))

            if batch_norm:
                layers.append(nn.BatchNorm1d(h))

            layers.append(nn.ReLU(inplace=True))
            in_dim = h

        layers.append(nn.Linear(in_dim, self.out_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        """
        z: (B, latent_dim)
        return: (B, 1, 32, 32) en [-1, 1]
        """

        x = self.net(z)  
        # Re acomodar el vector en forma de imagen (batch , 1×32×32)
        x = x.view(z.size(0), self.img_channels, self.img_size, self.img_size)
        return x

    @torch.no_grad()
    def sample(self, n: int, device=None):
        """
        Muestra n imágenes sintetizadas desde ruido estándar.
        """
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.forward(z)

class Discriminator(nn.Module):
    """
    Discriminador MLP para imágenes 1x32x32.
    Salida: logits (B, 1) -> usar con BCEWithLogitsLoss
    """

    def __init__(self, img_channels = 1, img_size= 32,
                 hidden_dims=(1024, 512, 256), dropout_p = 0.3 , batch_norm = True):
        
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        in_dim = img_channels * img_size * img_size  # 1*32*32 = 1024

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, bias=True))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))

            prev = h

        layers += [nn.Linear(prev, 1), nn.Sigmoid()] 

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, 1, 32, 32) en [-1, 1]
        return: logits (B, 1)  
        """
        b = x.size(0)
        x = x.view(b, -1)    
        probs = self.net(x)      
        return probs


def weights_init_normal(m):
    """
    Inicialización tipo DCGAN (funciona bien también para MLPs):
    - Linear: N(0, 0.02), bias = 0
    - BatchNorm: gamma ~ N(1,0.02), beta = 0
    """

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)

        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)



