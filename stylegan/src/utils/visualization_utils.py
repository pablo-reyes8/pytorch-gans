from contextlib import contextmanager

@contextmanager
def _eval_mode(module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        if was_training:
            module.train()

@torch.no_grad()
def _to_vis_range(x, assume_logits=True):
    if assume_logits:
        x = torch.tanh(x)
    return (x * 0.5 + 0.5).clamp(0, 1)

@torch.no_grad()
def sample_with_ema(G,
    gen_ema=None,
    z=None,
    z_dim=512,
    n=64,
    rows=8,
    seed=None,
    out_path=None,      
    assume_logits=True,
    style_mixing_prob=0.0,
    truncation=None):
    device = next(G.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    sampler = gen_ema if gen_ema is not None else G
    if z is None:
        z = torch.randn(n, z_dim, device=device)
    else:
        z = z.to(device)

    with _eval_mode(sampler):
        fake = sampler(
            z,
            style_mixing_prob=style_mixing_prob,
            update_w_avg=False,
            truncation=bool(truncation) if truncation is not None else False)
        vis = _to_vis_range(fake, assume_logits=assume_logits)

    if out_path is not None:
        dirname = os.path.dirname(out_path)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True) 
        vutils.save_image(vis, out_path, nrow=rows)

    return vis  

@torch.no_grad()
def show_grid(imgs, rows=8, title=None):
    """Muestra una grilla en pantalla sin guardar."""
    grid = vutils.make_grid(imgs, nrow=rows)
    plt.figure(figsize=(6,6))
    if title: plt.title(title)
    plt.axis("off")
    plt.imshow(grid.detach().cpu().permute(1,2,0).numpy())
    plt.show()
