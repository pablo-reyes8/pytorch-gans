import math
import torch.nn as nn
import torch.nn.functional as F
import torch

class PixelNorm(nn.Module):
    """
    Normalización por píxel (Pixel-wise feature vector normalization).

    Propósito:
    - Escalar cada vector de características (en cada posición espacial)
      a una norma similar, reduciendo variaciones de escala.
    - Evita que ciertos canales dominen por magnitudes grandes.
    - Fue introducida en ProGAN y se usa principalmente sobre el vector
      latente `z`, no tanto en capas profundas.

    Funcionamiento:
    - Para cada muestra y cada posición espacial (pixel),
      calcula la media de los cuadrados de los valores en los canales.
    - Divide (normaliza) el vector de canales por la raíz cuadrada de esa media.
    - Así, cada vector de características en cada pixel queda con una magnitud
      (norma) aproximadamente constante.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)

class EqualLinear(nn.Module):
    """
    Capa lineal con *Equalized Learning Rate* (Equalized LR).

    Propósito:
    - Estabilizar el entrenamiento de GANs controlando explícitamente
      la escala de los pesos y del bias durante el forward.
    - Introducido en *Progressive Growing of GANs* (Karras et al., 2017).
    - Alternativa a normalizaciones como BatchNorm, pero diseñada
      específicamente para generadores/discriminadores.

    Funcionamiento:
    - Los pesos se inicializan como N(0,1).
    - En el forward se reescala dinámicamente por:
          (lr_mul / sqrt(in_dim))
      lo que controla la varianza de las activaciones.
    - El bias también se escala por `lr_mul`.
    - Esto desacopla la inicialización y la tasa de aprendizaje efectiva:
      se puede usar un `lr_mul` mayor/menor sin modificar el optimizador.
    """

    def __init__(self, in_dim, out_dim, lr_mul=1.0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.lr_mul = lr_mul
        self.scale = (1.0 / math.sqrt(in_dim)) * lr_mul

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            bias = self.bias * self.lr_mul
        else:
            bias = None

        return F.linear(x, self.weight * self.scale, bias) # y=xA^t + b


class MappingNetwork(nn.Module):
    """
    z -> w
    - dims: 512
    - n_layers: 8 por defecto (paper)
    - activación: LeakyReLU(0.2)
    - sin BatchNorm
    - Equalized LR con lr_mul=0.01 (convención StyleGAN)
    - PixelNorm opcional en z
    """
    def __init__(self, z_dim=512, w_dim=512, n_layers=8, use_pixelnorm=True, lr_mul=0.01):
        super().__init__()
        self.use_pixelnorm = use_pixelnorm

        if use_pixelnorm:
            self.pn = PixelNorm()

        layers = []
        in_dim = z_dim
        for i in range(n_layers):
            layers.append(EqualLinear(in_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = w_dim

        # Quitar la activación de la última
        layers = layers[:-1] + []
        self.net = nn.Sequential(*layers)

        # Media móvil de w
        self.register_buffer("w_avg", torch.zeros(w_dim))

    @torch.no_grad()
    def update_w_avg(self, w, beta=0.995):
        batch_avg = w.mean(dim=0)
        self.w_avg.lerp_(batch_avg, 1.0 - beta)

    def forward(self, z, update_avg=False):
        if self.use_pixelnorm:
            z = self.pn(z)

        # Forward
        w = self.net(z)

        if update_avg and self.training:
            self.update_w_avg(w)

        return w