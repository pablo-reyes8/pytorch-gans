import torch
import torch.nn.functional as F


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Ema = Exponential Moving Averege. Copiaos el generador y actualizamos sus pesos mediante 
    un decay (parecido a Momentum pero no con gradientes con pesos). Esto se hace para que las imagenes 
    que produce el generador sean mas suaves. 
    """
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def diff_augment(x):
    """
    Introduce pequenas diferencias a las imagenes como rotaciones o traslaciones, 
    para que el discriminador no se sobreajuste en conjuntos pequenos como CIFRAR-10
    """
    # Flip 
    if torch.rand(1, device=x.device).item() < 0.5:
        x = torch.flip(x, dims=[3])

    # Translation
    if torch.rand(1, device=x.device).item() < 0.5:
        pad = 1
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        tx = int(torch.randint(-1, 2, (1,), device=x.device))
        ty = int(torch.randint(-1, 2, (1,), device=x.device))
        x = x[:, :, pad+ty:pad+ty+x.shape[2]-2*pad, pad+tx:pad+tx+x.shape[3]-2*pad]

    # Color jitter 
    if torch.rand(1, device=x.device).item() < 0.5:
        gain = 1.0 + 0.2 * (2*torch.rand(1, device=x.device).item() - 1) 
        x = x * gain
        x = x.clamp(-1, 1)
    return x
