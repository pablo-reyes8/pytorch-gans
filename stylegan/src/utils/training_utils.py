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


def diff_augment1(x, p=0.5, max_shift=1):
    """
    DiffAugment ligero:
      - Flip horizontal
      - Translación discreta sin wrap (relleno con 0)
      - Jitter multiplicativo de color (broadcast por batch)
    Supone imágenes en [-1, 1].
    """
    B, C, H, W = x.shape
    device = x.device

    #  Flip horizontal (por-batch)
    if torch.rand(1, device=device).item() < p:
        x = torch.flip(x, dims=[3])

    #  Translación sin wrap
    if torch.rand(1, device=device).item() < p and max_shift > 0:
        tx = int(torch.randint(-max_shift, max_shift + 1, (1,), device=device))
        ty = int(torch.randint(-max_shift, max_shift + 1, (1,), device=device))

        if tx != 0 or ty != 0:
            pad_left   = max(tx, 0)
            pad_right  = max(-tx, 0)
            pad_top    = max(ty, 0)
            pad_bottom = max(-ty, 0)
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
            x = x[:, :, pad_top:pad_top+H, pad_left:pad_left+W]

    #  Jitter multiplicativo de color
    if torch.rand(1, device=device).item() < p:
        gain = 1.0 + 0.2 * torch.empty(B, 1, 1, 1, device=device).uniform_(-1, 1)
        x = (x * gain).clamp(-1, 1)

    return x


def diff_augment(x, p=0.5, max_shift=1):
    B, C, H, W = x.shape
    dev = x.device

    # Flip
    if torch.rand(1, device=dev).item() < p:
        x = torch.flip(x, dims=[3])

    # Translation (sin wrap, padding replicate)
    if torch.rand(1, device=dev).item() < p and max_shift > 0:
        tx = int(torch.randint(-max_shift, max_shift+1, (1,), device=dev))
        ty = int(torch.randint(-max_shift, max_shift+1, (1,), device=dev))
        if tx or ty:
            pad_left, pad_right  = max(tx,0), max(-tx,0)
            pad_top,  pad_bottom = max(ty,0), max(-ty,0)
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
            x = x[:, :, pad_top:pad_top+H, pad_left:pad_left+W]
    return x


def diff_augment2(x, p=0.6, max_shift=1, color_gain=0.05, assume='logits'):
    """
    assume: 'logits' -> no clamp; 'minus1_1' -> clamp a [-1,1] (si entrenaras en ese rango).
    """
    B, C, H, W = x.shape
    dev = x.device

    # Flip
    if torch.rand(1, device=dev).item() < p:
        x = torch.flip(x, dims=[3])

    # Translation (sin wrap)
    if torch.rand(1, device=dev).item() < p and max_shift > 0:
        tx = int(torch.randint(-max_shift, max_shift+1, (1,), device=dev))
        ty = int(torch.randint(-max_shift, max_shift+1, (1,), device=dev))
        if tx or ty:
            pad_left, pad_right  = max(tx,0), max(-tx,0)
            pad_top,  pad_bottom = max(ty,0), max(-ty,0)
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
            x = x[:, :, pad_top:pad_top+H, pad_left:pad_left+W]

    # Color 
    if color_gain and torch.rand(1, device=dev).item() < p:
        gain = 1.0 + color_gain * torch.empty(B, 1, 1, 1, device=dev).uniform_(-1, 1)
        x = x * gain
        if assume == 'minus1_1':
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
