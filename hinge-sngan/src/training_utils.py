import torch
import torch.nn.functional as F


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

def diff_augment(x):
    # Flip horizontal (0.5)
    if torch.rand(1, device=x.device).item() < 0.5:
        x = torch.flip(x, dims=[3])
    # Translation pequeña (±1 px) con padding/reflexión
    if torch.rand(1, device=x.device).item() < 0.5:
        pad = 1
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        # shifts en {-1,0,1}
        tx = int(torch.randint(-1, 2, (1,), device=x.device))
        ty = int(torch.randint(-1, 2, (1,), device=x.device))
        x = x[:, :, pad+ty:pad+ty+x.shape[2]-2*pad, pad+tx:pad+tx+x.shape[3]-2*pad]
    # Color jitter simple (ganancia)
    if torch.rand(1, device=x.device).item() < 0.5:
        gain = 1.0 + 0.2 * (2*torch.rand(1, device=x.device).item() - 1)  # ~[0.8,1.2]
        x = x * gain
        x = x.clamp(-1, 1)
    return x
