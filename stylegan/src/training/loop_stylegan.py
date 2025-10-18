import torchvision.utils as vutils
import torch.nn.functional as F
from torch import nn
import os, copy
import torch
from utils.loss_utils import * 
from utils.training_utils import *
import matplotlib.pyplot as plt

###### Utils for training ########## 
@torch.no_grad()
def _to_vis_range(x):
    x = torch.tanh(x)
    return (x * 0.5 + 0.5).clamp(0, 1)

def to_logits(x):
  """
  Convierte salidas del D a logits (B,1):
  - Si viene como mapa (B,1,H,W) o (B,C,H,W): GAP sobre H,W.
  - Si viene como (B,) o (B,1): lo re-forma a (B,1).
  """
  if x.dim() == 4:
      x = x.mean(dim=(2,3), keepdim=False) 
  return x.view(-1, 1)

######### Main Train Loop ############ 

def train_gan(
    train_loader, generador, discriminador,
    optimizerG, optimizerD, criterion,
    latent_dim=512, epochs=20, gamma=10,
    fixed_z=None, smooth=False, smooth_advance=False,
    monitor_img=5, monitor_loss=2,
    train_gen=1, hinge=False,
    disc_steps=1,
    g_warmup_epochs=0,
    g_warmup_train_gen=2,
    use_ema=True, ema_decay=0.999,
    use_diffaug=True,
    r1_every=16, use_softplus=False, style_mixing_prob=0.0,
    ema_warmup=5,
    start_epoch=1,            # epoch absoluta inicial (para logging / warmups)
    base_global_step=0,       # step absoluto inicial (para R1 % r1_every)
    gen_ema=None              # pasa aquí tu EMA previa para no recrearla
):
    """
    Entrenamiento de (Style)GAN con Hinge o BCE/Softplus, R1, EMA y DiffAug.
    NUEVO:
      - start_epoch: para llevar conteo absoluto de épocas (logging, EMA warmup).
      - base_global_step: para que R1 no 'reinicie' su mod r1_every entre tandas.
      - gen_ema: pasa tu EMA previa para continuar suavización global.
    """


    if smooth and smooth_advance:
        raise ValueError('No se puede tener dos suavizamientos al tiempo')
    if hinge and (smooth or smooth_advance):
        raise ValueError('No se puede usar suavizado y hinge al mismo tiempo')
    if use_softplus and (criterion is not None):
        # criterion se ignora en modo softplus
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generador.to(device).train()
    discriminador.to(device).train()

    #  EMA 
    if use_ema:
        if gen_ema is None:
            gen_ema = copy.deepcopy(generador).to(device)
            gen_ema.eval()
            for p in gen_ema.parameters():
                p.requires_grad_(False)
    else:
        gen_ema = None

    #  fixed_z consistente entre tandas 
    if fixed_z is None:
        fixed_z = torch.randn(64, latent_dim, device=device)

    save_dir = make_unique_dir("samples")
    os.makedirs(save_dir, exist_ok=True)

    # snapshot inicial (para comparar progreso)
    with torch.no_grad():
        sampler0 = generador if (not use_ema or start_epoch <= ema_warmup or gen_ema is None) else gen_ema
        fake0 = sampler0(fixed_z).detach()
        fake0_vis = _to_vis_range(fake0)
    vutils.save_image(fake0_vis, f"{save_dir}/epoch_{start_epoch-1:04d}.png", nrow=8)

    grid = vutils.make_grid(fake0_vis, nrow=8)
    plt.figure(figsize=(6,6)); plt.axis("off") 
    plt.title("Fake samples (init) [G]")
    plt.imshow(grid.detach().cpu().permute(1,2,0).numpy())
    plt.show()

    history = {'loss_D': [], 'loss_G': []}
    global_step = int(base_global_step)

    #  loop de entrenamiento
    for e in range(epochs):
        absolute_epoch = start_epoch + e
        running_D = 0.0
        running_G = 0.0
        n_batches = 0

        g_steps_this_epoch = g_warmup_train_gen if absolute_epoch <= g_warmup_epochs else train_gen

        for real, _ in train_loader:
            b = real.size(0)
            real = real.to(device, non_blocking=True)

            # ========== Discriminador (disc_steps veces) ==========
            for _ in range(disc_steps):
                optimizerD.zero_grad(set_to_none=True)

                # Reales
                real_in = diff_augment(real) if use_diffaug else real
                out_real = to_logits(discriminador(real_in)) 

                # Falsas
                z = torch.randn(b, latent_dim, device=device)
                fake = generador(z, style_mixing_prob=0.0, update_w_avg=True, truncation=False).detach()
                fake_in = diff_augment(fake) if use_diffaug else fake
                out_fake = to_logits(discriminador(fake_in))

                if hinge:
                    # Hinge
                    loss_D_core = loss_hinge_discriminator(out_real, out_fake)

                    # R1 SIN augment en reales 
                    if (global_step % r1_every) == 0:
                        real_r1 = real.detach().requires_grad_(True)
                        out_real_r1 = to_logits(discriminador(real_r1))
                        r1 = r1_penalty(out_real_r1, real_r1)
                        loss_D = loss_D_core + 0.5 * gamma * r1 * r1_every
                    else:
                        loss_D = loss_D_core

                else:
                    if use_softplus:
                        # No-saturante (StyleGAN-like)
                        loss_D_core = F.softplus(-out_real).mean() + F.softplus(out_fake).mean()
                        if (global_step % r1_every) == 0:
                            real_r1 = real.detach().requires_grad_(True)
                            out_real_r1 = to_logits(discriminador(real_r1))
                            r1 = r1_penalty(out_real_r1, real_r1)
                            loss_D = loss_D_core + 0.5 * gamma * r1 * r1_every
                        else:
                            loss_D = loss_D_core
                    else:
                        # BCE con posibles suavizados (no recomendado con R1 moderno)
                        if smooth:
                            real_labels = torch.full((b, 1), 0.9, device=device, dtype=out_real.dtype)
                        elif smooth_advance:
                            real_labels = torch.empty(b, 1, device=device, dtype=out_real.dtype).uniform_(0.8, 1.0)
                        else:
                            real_labels = torch.ones(b, 1, device=device, dtype=out_real.dtype)
                        fake_labels = torch.zeros(b, 1, device=device, dtype=out_fake.dtype)
                        loss_D_real = criterion(out_real, real_labels)
                        loss_D_fake = criterion(out_fake, fake_labels)
                        loss_D_core = 0.5 * (loss_D_real + loss_D_fake)

                        if (global_step % r1_every) == 0:
                            real_r1 = real.detach().requires_grad_(True)
                            out_real_r1 = to_logits(discriminador(real_r1))
                            r1 = r1_penalty(out_real_r1, real_r1)
                            loss_D = loss_D_core + 0.5 * gamma * r1 * r1_every
                        else:
                            loss_D = loss_D_core

                loss_D.backward()
                optimizerD.step()
                global_step += 1

            # ========== Generador (g_steps_this_epoch veces) ==========
            loss_G_total = 0.0
            for _ in range(g_steps_this_epoch):
                optimizerG.zero_grad(set_to_none=True)
                z = torch.randn(b, latent_dim, device=device)
                fake = generador(z, style_mixing_prob=style_mixing_prob, update_w_avg=True, truncation=False)
                fake_in_for_G = diff_augment(fake) if use_diffaug else fake
                out_fake_for_G = to_logits(discriminador(fake_in_for_G))

                if hinge:
                    loss_G = loss_hinge_generator(out_fake_for_G)
                else:
                    if use_softplus:
                        loss_G = F.softplus(-out_fake_for_G).mean() # Softplus
                    else:
                        real_labels = torch.ones_like(out_fake_for_G, device=device, dtype=out_fake_for_G.dtype)
                        loss_G = criterion(out_fake_for_G, real_labels)

                loss_G.backward()
                optimizerG.step()
                loss_G_total += loss_G.item()

                # EMA update
                if use_ema and (gen_ema is not None):
                    # decay_t = min(ema_decay, 1 - 1.0/(global_step+1))  # opcional
                    update_ema(gen_ema, generador, decay=ema_decay)

            # métricas batch
            running_D += loss_D.item()
            running_G += (loss_G_total / max(1, g_steps_this_epoch))
            n_batches += 1

        # fin de época: logging y muestras
        epoch_loss_D = running_D / max(1, n_batches)
        epoch_loss_G = running_G / max(1, n_batches)
        history['loss_D'].append(epoch_loss_D)
        history['loss_G'].append(epoch_loss_G)

        if (absolute_epoch % monitor_loss) == 0:
            if hinge or use_softplus:
                with torch.no_grad():
                    d_real_m = out_real.mean().item()
                    d_fake_m = out_fake.mean().item()
            else:
                with torch.no_grad():
                    real_labels = torch.ones_like(out_real, device=device, dtype=out_real.dtype)
                    fake_labels = torch.zeros_like(out_fake, device=device, dtype=out_fake.dtype)
                    d_real_m = criterion(out_real, real_labels).item()
                    d_fake_m = criterion(out_fake, fake_labels).item()

            print(f"[Epoch {absolute_epoch:03d}] "
                  f"loss_D={epoch_loss_D:.4f} | loss_G={epoch_loss_G:.4f} | "
                  f"D(real)={d_real_m:.3f} | D(fake)={d_fake_m:.3f} | "
                  f"Dsteps={disc_steps} | Gsteps={g_steps_this_epoch}")

        if (absolute_epoch % monitor_img) == 0:
            with torch.no_grad():
                use_live_G = (absolute_epoch <= ema_warmup) or (not use_ema)
                sampler = generador if (use_live_G or gen_ema is None) else gen_ema
                fake = sampler(fixed_z).detach()
                fake_vis = _to_vis_range(fake)
            vutils.save_image(fake_vis, f"{save_dir}/epoch_{absolute_epoch:04d}.png", nrow=8)

            grid = vutils.make_grid(fake0_vis, nrow=8)
            plt.figure(figsize=(6,6))
            plt.axis("off") 
            plt.title(f"Fake samples ({absolute_epoch}) [EMA]")
            plt.imshow(grid.detach().cpu().permute(1,2,0).numpy())
            plt.show()

    # Devuelve también fixed_z y global_step para facilitar checkpoint
    return history, gen_ema, fixed_z, global_step, (start_epoch + epochs - 1)
