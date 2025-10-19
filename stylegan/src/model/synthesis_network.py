import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
import torch 
import math
from model.mapping_network import *

class EqualConv2d(nn.Module):
    """
    Convolución con *Equalized Learning Rate* (Equalized LR).

    Propósito:
    - Estabilizar el entrenamiento en GANs ajustando la escala efectiva
      de los pesos durante el forward.
    - Fue introducida en *Progressive Growing of GANs* (Karras et al., 2017),
      junto con EqualLinear, como alternativa a normalizaciones como BatchNorm.

    Funcionamiento:
    - Inicializa los pesos con distribución N(0,1).
    - En el forward, los pesos se reescalan dinámicamente por:
          (lr_mul / sqrt(in_ch * kernel^2))
      lo cual mantiene controlada la varianza de las activaciones.
    - El bias, si existe, también se escala por `lr_mul`.
    - Esto permite separar la inicialización del control de la tasa de
      aprendizaje efectiva, facilitando la estabilidad sin necesidad
      de capas de normalización adicionales.

    Beneficio:
    - Ayuda a que la magnitud de las activaciones no dependa de la
      dimensionalidad de entrada ni del tamaño del kernel.
    """

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, lr_mul=1.0, bias=True):
        super().__init__()
        weight = torch.randn(out_ch, in_ch, kernel, kernel)
        self.weight = nn.Parameter(weight)
        self.lr_mul = lr_mul
        self.scale = (lr_mul / math.sqrt(in_ch * (kernel * kernel)))
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x):
        w = self.weight * self.scale
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)


class NoiseInjection(nn.Module):
    """
    Capa de inyección de ruido en mapas de características.

    Propósito:
    - Introducir variaciones estocásticas locales en las imágenes generadas,
      permitiendo que el generador aprenda a modelar detalles finos (textura,
      arrugas, pelo, imperfecciones) sin que estén codificados directamente
      en el vector latente `z`.
    """

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise


class AffineStyle(nn.Module):
    """
    Proyección afín del vector latente intermedio w → [scale, bias].

    Propósito:
    - Traducir el vector latente `w` (dimensión 512 en StyleGAN) en parámetros
      de normalización adaptativa: una escala (γ) y un sesgo (β) por canal.
    - Estos parámetros controlan la modulación en *AdaIN*, permitiendo que
      el estilo global de la imagen (colores, texturas, rasgos) dependa
      directamente de `w`.
    - Esto genera el yb yc que se usa en AdaIn

    Funcionamiento:
    - Capa lineal (EqualLinear) que mapea w ∈ R^w_dim → R^(2*C).
      Los primeros C valores corresponden a escalas (γ),
      los siguientes C a sesgos (β).
    - Inicialización especial:
        • bias de la mitad "escala" inicializado en 1.0 (para que empiece
          como una identidad: γ≈1).
        • bias de la mitad "sesgo" inicializado en 0.0 (β≈0).
      → Esto asegura que al inicio AdaIN no modifique demasiado la activación.

    Beneficio:
    - El generador puede controlar de forma precisa cómo se aplican los estilos
      capa por capa, combinando información global (w) con normalización local.
    """
    def __init__(self, w_dim, channels, lr_mul=1.0):
        super().__init__()
        self.fc = EqualLinear(w_dim, 2 * channels, lr_mul=lr_mul)
        nn.init.zeros_(self.fc.bias)
        with torch.no_grad():
            self.fc.bias[:channels].fill_(1.0)

    def forward(self, w):
        return self.fc(w)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN).

    Propósito:
    - Combinar normalización de características con un control directo de estilo
      proveniente del vector latente `w`.
    - Introducido originalmente para transferencia de estilo (Huang & Belongie, 2017),
      adaptado en *StyleGAN* (Karras et al., 2019) para controlar estilos en el generador.

    Funcionamiento:
    1. Normalización:
       - Se aplica InstanceNorm (por canal y por muestra), es decir:
         restar la media y dividir por la desviación estándar de cada canal
         en cada imagen → elimina información de escala e iluminación.
    2. Estilo:
       - El vector latente intermedio `w` es proyectado por `AffineStyle`
         en dos vectores: `scale (γ)` y `bias (β)`, ambos de dimensión C.
    3. Reescalado:
       - El tensor normalizado se modula con estos parámetros:
         x' = γ * x_norm + β
       - Esto reinyecta control de estilo capa a capa.

    Beneficio:
    - Elimina la dependencia de BatchNorm/LayerNorm.
    - Permite que el estilo de cada capa dependa explícitamente de `w`,
      lo que otorga un control jerárquico sobre la imagen generada
      (colores en capas bajas, formas globales en capas altas).
    """

    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.channels = channels

    def forward(self, x, style):
        B, C, H, W = x.shape
        assert C == self.channels
        style = style.view(B, 2, C)    # [B,2,C]
        scale, bias = style[:, 0], style[:, 1]  # [B,C], [B,C]

        # InstanceNorm: normaliza por canal en cada imagen
        mean = x.mean(dim=(2,3), keepdim=True)
        var = x.var(dim=(2,3), unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Aplicamos AdaIn
        x_styled = x_norm * scale.view(B, C, 1, 1) + bias.view(B, C, 1, 1)
        return x_styled


class Blur2d(nn.Module):
    """
    Filtro de desenfoque 2D separable (Gaussian-like) usado comúnmente en GANs
    para estabilizar el entrenamiento al suavizar aliasing en imágenes.

    Args:
        kernel (tuple): Coeficientes del filtro 1D. Por defecto (1, 3, 3, 1).
        pad (tuple): Padding asimétrico (left, right, top, bottom), típicamente (1, 2, 1, 2).

    Atributos:
        weight (Tensor): Kernel 2D normalizado, registrado como buffer (no entrenable).
        pad (tuple): Especifica el padding reflejado aplicado antes de la convolución.

    Forward:
        Aplica padding reflejado y luego convolución por canal (groups=C)
        para suavizar la entrada sin alterar el número de canales.

    Output:
        Tensor con el mismo tamaño espacial que la entrada, pero con bordes suavizados.
    """
    def __init__(self, kernel=(1,3,3,1), pad=(1,2,1,2)):
        super().__init__()
        k = torch.tensor(kernel, dtype=torch.float32)
        k2d = (k[:, None] * k[None, :])          # kernel 2D separable
        k2d /= k2d.sum()                         # normalización
        self.register_buffer("weight", k2d[None, None, :, :])
        self.pad = pad                           # padding asimétrico

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.weight.repeat(C, 1, 1, 1)
        x = F.pad(x, self.pad, mode="reflect")   # alternativa: "replicate"
        return F.conv2d(x, w, padding=0, groups=C)


class StyledConv(nn.Module):
    """
    Bloque convolucional modulado al estilo StyleGAN.

    Propósito (visión de alto nivel):
    - Combinar tres ideas clave para controlar “forma + textura + estilo” por capa:
      1) Convolución con Equalized LR (estabilidad en la escala de activaciones).
      2) Ruido estocástico canal-específico (detalles finos locales).
      3) Modulación de estilo vía AdaIN con parámetros derivados de w (control global).
    - El orden Conv → Noise → Activación → AdaIN separa:
        • extracción de características (conv),
        • inyección de estocasticidad (noise),
        • no linealidad (LReLU),
        • y normalización/estilización (AdaIN con [γ, β] desde w).
      Así, la normalización actúa sobre activaciones ya “excitadas” por la no linealidad.

    Flujo:
    - (Opcional) Upsample por interpolación: duplica H y W antes de la conv.
    - EqualConv2d: extracción de features con escala controlada.
    - NoiseInjection: añade variación local per-pixel escalada por canal.
    - LeakyReLU: introduce no linealidad estable.
    - AffineStyle(w) → [scale, bias]; AdaIN normaliza por instancia y modula con [γ, β].
    """

    def __init__(self, in_ch, out_ch, kernel, w_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.blur = Blur2d((1,3,3,1)) if upsample else None
        self.conv = EqualConv2d(in_ch, out_ch, kernel, padding=kernel//2) # Convolución con Equalized LR (padding para mantener tamaño si no hay upsample)
        self.noise = NoiseInjection(out_ch)# Ruido estocástico (parámetro por canal que escala el ruido)
        self.affine = AffineStyle(w_dim, out_ch, lr_mul=1.0)  # Proyección afín del estilo w → [scale, bias] por canal yb yc
        self.adain = AdaIN(out_ch)  # Normalización adaptativa por instancia
        self.act = nn.LeakyReLU(0.2, inplace=True) # No linealidad estable

    def forward(self, x, w, noise=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.blur(x) 
            
        x = self.conv(x)
        x = self.noise(x, noise=noise)
        x = self.act(x)
        x = self.adain(x, self.affine(w))
        return x

class ToRGB(nn.Module):
    """
    Capa de salida que convierte un mapa de características en una imagen RGB.

    Propósito:
    - Proyectar las activaciones de un bloque convolucional al espacio de imagen
      con 3 canales (R, G, B).
    - No aplica activación final como Tanh; devuelve logits (valores sin normalizar),
      lo que deja al discriminador aprender la escala correcta de intensidades.

    Funcionamiento:
    - Convolución 1x1 con Equalized LR que ajusta los canales internos a 3.
    - No cambia resolución (solo canales), actuando como una proyección lineal
      sobre el espacio de color.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.conv = EqualConv2d(in_ch, 3, kernel=1, padding=0, lr_mul=0.1)

    def forward(self, x):
        return self.conv(x)


class ConstantInput(nn.Module):
    """Tensor aprendido 4x4xC, compartido por el batch."""
    def __init__(self, channels, size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, channels, size, size))

    def forward(self, batch_size):
        return self.weight.repeat(batch_size, 1, 1, 1)


class SynthesisNetwork64(nn.Module):
    """
    StyleGAN v1-like (AdaIN) a 64x64.
    - Empieza en constante 4x4x512
    - Por resolución: 2 convs “styled” + upsample entre escalas
    - ToRGB al final (64x64)
    - Soporta style mixing: pasar w por capa [B, L, 512] o uno único [B, 512]
    """

    def __init__(self, w_dim=512, fmap_base=512, ch_schedule=(512, 512, 256, 128, 64)):
        """
        ch_schedule define canales por resolución:
        4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        """

        super().__init__()
        assert len(ch_schedule) == 5, "Para 64x64 se esperan 5 resoluciones."
        c4, c8, c16, c32, c64 = ch_schedule

        self.const = ConstantInput(c4, size=4)

        # 4x4 (dos convs, sin upsample en la primera)
        self.conv4_1 = StyledConv(c4, c4, 3, w_dim, upsample=False)
        self.conv4_2 = StyledConv(c4, c8, 3, w_dim, upsample=True)   # upsample hacia 8x8

        # 8x8
        self.conv8_1 = StyledConv(c8, c8, 3, w_dim, upsample=False)
        self.conv8_2 = StyledConv(c8, c16, 3, w_dim, upsample=True)  # → 16x16

        # 16x16
        self.conv16_1 = StyledConv(c16, c16, 3, w_dim, upsample=False)
        self.conv16_2 = StyledConv(c16, c32, 3, w_dim, upsample=True)  # → 32x32

        # 32x32
        self.conv32_1 = StyledConv(c32, c32, 3, w_dim, upsample=False)
        self.conv32_2 = StyledConv(c32, c64, 3, w_dim, upsample=True)  # → 64x64

        # 64x64
        self.conv64_1 = StyledConv(c64, c64, 3, w_dim, upsample=False)
        self.conv64_2 = StyledConv(c64, c64, 3, w_dim, upsample=False)

        self.to_rgb = ToRGB(c64)
        self.num_layers = 10


    def forward(self, w):
        """
        w: [B, 512]  -> usa el mismo estilo para todas las capas
           [B, L, 512] con L=self.num_layers -> style mixing por capa
        return: imagen [B, 3, 64, 64] (logits, sin tanh)
        """

        if w.dim() == 2:
          w = w.unsqueeze(1).repeat(1, self.num_layers, 1)  # brodcasting para tener L vectores w =  [B,L,512]
        else:
            assert w.size(1) == self.num_layers

        B = w.size(0)
        x = self.const(B)

        w_iter = iter(w.unbind(dim=1))  # genera 10 tensores [B,512], uno por capa

        # 4x4
        x = self.conv4_1(x, next(w_iter))
        x = self.conv4_2(x, next(w_iter))  # upsample→8

        # 8x8
        x = self.conv8_1(x, next(w_iter))
        x = self.conv8_2(x, next(w_iter))  # →16

        # 16x16
        x = self.conv16_1(x, next(w_iter))
        x = self.conv16_2(x, next(w_iter)) # →32

        # 32x32
        x = self.conv32_1(x, next(w_iter))
        x = self.conv32_2(x, next(w_iter)) # →64

        # 64x64
        x = self.conv64_1(x, next(w_iter))
        x = self.conv64_2(x, next(w_iter))


        return self.to_rgb(x)
