import torch
import os 

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Ema = Exponential Moving Averege. Copiaos el generador y actualizamos sus pesos mediante
    un decay (parecido a Momentum pero no con gradientes con pesos). Esto se hace para que las imagenes
    que produce el generador sean mas suaves.
    """
    for (n_ema, p_ema), (n, p) in zip(ema_model.named_parameters(), model.named_parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    for (n_ema, b_ema), (n, b) in zip(ema_model.named_buffers(), model.named_buffers()):
        b_ema.copy_(b)


def diff_augment(x):
    """
    DiffAugment ligero para robustecer al discriminador en datasets pequeños.

    Idea:
    - Añadir transformaciones estocásticas simples (flip, translación pequeña, jitter de color)
      para que D no se sobreajuste a píxeles exactos y aprenda invariancias útiles.

    Operaciones (aplicadas con probabilidad 0.5):
    - Flip horizontal
    - Translación discreta pequeña (±1 píxel) usando torch.roll (conserva tamaño sin recortes)
    - Color jitter multiplicativo → reescala intensidades levemente y clamp a [-1, 1]
      (asumiendo entradas normalizadas a ese rango)
    """
    # Flip
    if torch.rand(1, device=x.device).item() < 0.5:
        x = torch.flip(x, dims=[3])

    # Translation
    if torch.rand(1, device=x.device).item() < 0.5:
        tx = int(torch.randint(-1, 2, (1,), device=x.device))
        ty = int(torch.randint(-1, 2, (1,), device=x.device))
        x = torch.roll(x, shifts=(ty, tx), dims=(2, 3))

        if ty != 0:
          x[:, :, ty:, :] = x[:, :, ty:, :].clone()

    # Color jitter
    if torch.rand(1, device=x.device).item() < 0.5:
        gain = 1.0 + 0.2 * (2*torch.rand(1, device=x.device).item() - 1)
        x = x * gain
        x = x.clamp(-1, 1)
    return x

def make_unique_dir(base="samples"):
    if not os.path.exists(base):
        os.makedirs(base)
        return base

    i = 1
    while os.path.exists(f"{base}_{i}"):
        i += 1
    new_dir = f"{base}_{i}"
    os.makedirs(new_dir)
    return new_dir