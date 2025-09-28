import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
import torch 
import math

def maybe_sn(module: nn.Module, use_sn: bool):
    return SN(module) if use_sn else module


class MinibatchStdDev(nn.Module):
    """
    Capa que incentiva la diversidad en los outputs de la red añadiendo un canal
    con la desviación estándar del minibatch.

    Idea central:
    - Si todas las muestras generadas son muy parecidas, la desviación estándar
      entre ellas será pequeña.
    - Al incluir explícitamente esta métrica como un canal extra en el tensor
      de activaciones, el discriminador aprende a detectar falta de diversidad.
    - Esto fuerza al generador a producir outputs más variados para engañarlo.

    Funcionamiento:
    1. Divide el batch en grupos (group_size).
    2. Calcula la desviación estándar por grupo sobre todos los canales y píxeles.
    3. Promedia esa desviación para obtener un único valor escalar por grupo.
    4. Replica ese valor en forma de un mapa 2D y lo concatena como un canal extra.
    5. Así, cada muestra lleva información de cuánta variabilidad hay entre las imágenes.
    """
    def __init__(self, group_size=32, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        g = min(self.group_size, B)
        if B % g != 0:
            g = B
        if g > 1:
            y = x.view(g, -1, C, H, W)
            y = y - y.mean(dim=0, keepdim=True)
            y = torch.sqrt(y.pow(2).mean(dim=0) + self.eps)
            y = y.mean(dim=(1, 2, 3), keepdim=True)
            y = y.repeat(g, 1, H, W)
            return torch.cat([x, y], dim=1)
        else:
            y = x.new_zeros(B, 1, H, W)
            return torch.cat([x, y], dim=1)


class ResDownBlock(nn.Module):
    """
    Bloque residual con reducción de resolución (downsampling).

    Propósito:
    - Reducir la resolución espacial (H, W) a la mitad mientras se transforman
      las características.
    - Mantener un camino residual (skip) para estabilizar el entrenamiento
      y facilitar el flujo del gradiente.
    - Similar a los bloques residuales de ResNet, pero incorporando un downsample.

    Estructura:
    - Rama principal (main path):
        Conv 3x3 -> LeakyReLU -> Conv 3x3 -> AvgPool(2x2)
    - Rama de atajo (skip path):
        Conv 1x1 -> AvgPool(2x2)
    - Se suman ambas ramas y se aplica una activación final.

    """
    def __init__(self, in_ch, out_ch, use_sn: bool):
        super().__init__()
        self.conv1 = maybe_sn(nn.Conv2d(in_ch, in_ch, 3, padding=1), use_sn)
        self.conv2 = maybe_sn(nn.Conv2d(in_ch, out_ch, 3, padding=1), use_sn)
        self.skip  = maybe_sn(nn.Conv2d(in_ch, out_ch, 1), use_sn)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.down  = nn.AvgPool2d(2)

        # Inits para LeakyReLU (SN no las “rompe”; sigue siendo útil)
        for m in [self.conv1, self.conv2, self.skip]:
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        y = self.down(y)
        s = self.down(self.skip(x))
        return self.act((y + s) / math.sqrt(2))


class Discriminator64(nn.Module):
    """
    Discriminador estilo StyleGAN para 64×64:
    - FromRGB 3x3 -> pila de ResDownBlocks (64→32→16→8→4)
    - MinibatchStdDev
    - Conv final + Linear a logit escalar
    """
    def __init__(self, base_ch=64, max_ch=512, in_channels=3, use_sn: bool=True):
        super().__init__()
        C1 = base_ch
        C2 = min(base_ch * 2, max_ch)
        C3 = min(base_ch * 4, max_ch)
        C4 = min(base_ch * 8, max_ch)

        # FromRGB
        self.from_rgb = maybe_sn(nn.Conv2d(in_channels, C1, 3, padding=1), use_sn)

        # Bloques residuales con SN opcional
        self.b64 = ResDownBlock(C1, C2, use_sn)
        self.b32 = ResDownBlock(C2, C3, use_sn)
        self.b16 = ResDownBlock(C3, C4, use_sn)
        self.b08 = ResDownBlock(C4, C4, use_sn)

        self.mbstd = MinibatchStdDev(group_size=32)

        # Conv final tras mbstd (+1 canal)
        self.conv_final = maybe_sn(nn.Conv2d(C4 + 1, C4, 3, padding=1), use_sn)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Head: GAP → Linear
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = maybe_sn(nn.Linear(C4, 1), use_sn)

        # Inicializaciones
        nn.init.kaiming_normal_(self.from_rgb.weight, a=0.2, nonlinearity='leaky_relu')
        if self.from_rgb.bias is not None: nn.init.zeros_(self.from_rgb.bias)

        nn.init.kaiming_normal_(self.conv_final.weight, a=0.2, nonlinearity='leaky_relu')
        if self.conv_final.bias is not None: nn.init.zeros_(self.conv_final.bias)

        # Con SN en Linear, una init pequeña es suficiente
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        if self.fc.bias is not None: nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        y = self.act(self.from_rgb(x))   # [B, C1, 64, 64]
        y = self.b64(y)                  # [B, C2, 32, 32]
        y = self.b32(y)                  # [B, C3, 16, 16]
        y = self.b16(y)                  # [B, C4,  8,  8]
        y = self.b08(y)                  # [B, C4,  4,  4]

        y = self.mbstd(y)                # [B, C4+1, 4, 4]
        y = self.act(self.conv_final(y)) # [B, C4,   4,  4]
        y = self.gap(y).view(y.size(0), -1)  # [B, C4]
        logit = self.fc(y)               # [B, 1]
        return logit