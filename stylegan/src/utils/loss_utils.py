import torch
import torch.nn.functional as F


def loss_hinge_discriminator(d_real, d_fake):
    """
    Hinge loss para el discriminador (SNGAN/BigGAN-style).

    Idea:
    - Empuja a D(x) ≥ 1 para reales y D(G(z)) ≤ -1 para falsas.
    - Márgenes ±1 estabilizan y evitan saturación típica de BCE.
    d_real: logits de D(x)   -> (N,1)
    d_fake: logits de D(G(z)) -> (N,1)
    """
    loss_real = torch.mean(F.relu(1.0 - d_real)) # max(0, 1 - D(x))
    loss_fake = torch.mean(F.relu(1.0 + d_fake)) # max(0, 1 + D(G(z)))
    return loss_real + loss_fake

def loss_hinge_generator(d_fake):
    """
    Hinge loss para el generador.

    Idea:
    - G quiere que D(G(z)) sea grande (idealmente > 1).
    - Se implementa como L_G = -E[D(G(z))].
    """

    return -torch.mean(d_fake) # G quiere subir D(G(z))


def r1_penalty(d_out_real, real):
    """
    Penalización R1 en datos reales: E[ ||∂D/∂x||^2 ].

    Idea:
    - Regulariza la suavidad de D alrededor de la variedad de datos reales.
    - Mejora estabilidad (reduce oscilaciones/gradientes patológicos).

    Implementación:
    - Toma gradiente de la suma de logits reales wrt. la imagen real.
    - Cuadra y promedia la norma L2 por muestra.

    Tip práctico:
    - Es costoso; aplicar cada N pasos (e.g., 16) y escalar por N en la loss total
      para mantener el valor esperado (L = ... + 0.5 * gamma * r1 * N).
    """
    grad = torch.autograd.grad(
        outputs=d_out_real.sum(), inputs=real,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.reshape(grad.size(0), -1)
    return (grad.pow(2).sum(dim=1)).mean()