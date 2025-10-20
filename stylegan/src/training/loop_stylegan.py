import torchvision.utils as vutils
import torch.nn.functional as F
from torch import nn
import os, copy
import torch
from utils.loss_utils import * 
from utils.training_utils import *
import matplotlib.pyplot as plt
import time

###### Utils for training ########## 
def tensor_mean_cpu(t):
    """
    Devuelve la media escalar del tensor `t` como float en CPU.
    Útil para registrar métricas sin mantener gradientes ni usar GPU.
    """
    return float(t.detach().mean().cpu())


def grad_norm(module):
    """
    Calcula la norma L2 total de todos los gradientes de un módulo PyTorch.
    Sirve para monitorear la magnitud del gradiente y detectar saturación o explosión.
    """
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += (g.pow(2).sum()).item()
    return (total ** 0.5)


def param_norm(module):
    """
    Calcula la norma L2 de todos los parámetros del módulo.
    Permite evaluar la escala general de los pesos en entrenamiento.
    """
    total = 0.0
    for p in module.parameters():
        d = p.detach()
        total += (d.pow(2).sum()).item()
    return (total ** 0.5)


def sched_stylemix(epoch):
    """
    Scheduler del parámetro de style-mixing:
    - Epochs 1–2: bajo (0.1)
    - 3–6: incremento lineal (0.15 → 0.375)
    - 7+: estable (0.45)
    Controla la intensidad del mixing progresivamente.
    """
    if epoch <= 2:  
        return 0.1
    if epoch <= 6:  
        return 0.15 + 0.075*(epoch-3)
    return 0.45


def sched_lrd(history_D):
    """
    Ajusta dinámicamente el learning rate del discriminador.
    Si el D se vuelve demasiado dominante (promedio de aciertos < 0.40 en las últimas 3 iteraciones),
    reduce el lr en un 20%. Devuelve un multiplicador (0.8 o 1.0).
    """
    if len(history_D) >= 3 and sum(history_D[-3:])/3 < 0.40:
        return 0.8
    return 1.0

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
    gen_ema=None, # pasa aquí tu EMA previa para no recrearla
    style_decay = False, 
    lr_Disc_decay =False,
    control_noise = False
):
    """
    Entrenamiento de (Style)GAN con Hinge o BCE/Softplus, R1, EMA y DiffAug:
      - start_epoch: para llevar conteo absoluto de épocas (logging, EMA warmup).
      - base_global_step: para que R1 no 'reinicie' su mod r1_every entre tandas.
      - gen_ema: pasa tu EMA previa para continuar suavización global.

    (Mejoras de logging y monitoreo sin costo relevante de GPU)
      - Promedios por época de D(real) y D(fake) (no solo el último batch).
      - Normas de gradiente de G y D (promedio por época, computadas tras backward).
      - Normas L2 de parámetros (||θ||) de G y D (cada época, baratas).
      - Tiempos por época y pasos procesados.
      - Historia extendida en 'history': loss_D, loss_G, D_real, D_fake,
        grad_norm_G, grad_norm_D, param_norm_G, param_norm_D, epoch_time_sec.
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

    # snapshot inicial 
    with torch.no_grad():
        sampler0 = generador if (not use_ema or gen_ema is None) else gen_ema
        fake0 = sampler0(fixed_z).detach()
        fake0_vis = _to_vis_range(fake0)
    vutils.save_image(fake0_vis, f"{save_dir}/epoch_{start_epoch-1:04d}.png", nrow=8)

    grid = vutils.make_grid(fake0_vis, nrow=8)
    plt.figure(figsize=(6,6)); plt.axis("off")
    plt.title("Fake samples (init) [G or G_EMA]")
    plt.imshow(grid.detach().cpu().permute(1,2,0).numpy())
    plt.show()

    # Historia 
    history = {
        'loss_D': [], 'loss_G': [],
        'D_real': [], 'D_fake': [],
        'grad_norm_G': [], 'grad_norm_D': [],
        'param_norm_G': [], 'param_norm_D': [],
        'epoch_time_sec': [], 'steps_per_epoch': []}

    # Pequeña EMA para loss (suavizado visual de logs)
    def ema_update(old, new, alpha=0.98):
        return new if old is None else (alpha * old + (1 - alpha) * new)

    lossD_ema, lossG_ema = None, None


    global_step = int(base_global_step)

    #  loop de entrenamiento
    for e in range(epochs):
        t0 = time.time()
        absolute_epoch = start_epoch + e

        # Acumuladores por época
        running_D = 0.0
        running_G = 0.0
        n_batches = 0

        dreal_acc, dfake_acc, count_acc = 0.0, 0.0, 0
        gradG_acc, gradD_acc = 0.0, 0.0
        gradG_cnt, gradD_cnt = 0, 0

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

                # Monitoreo ligero: norma de gradiente de D
                gnormD = grad_norm(discriminador)
                gradD_acc += gnormD; gradD_cnt += 1

                optimizerD.step()
                global_step += 1

                # Acumular promedios de D(real)/D(fake) 
                with torch.no_grad():
                    dreal_acc += tensor_mean_cpu(out_real)
                    dfake_acc += tensor_mean_cpu(out_fake)
                    count_acc += 1

            # ========== Generador (g_steps_this_epoch veces) ==========
            loss_G_total = 0.0
            for _ in range(g_steps_this_epoch):
                optimizerG.zero_grad(set_to_none=True)
                z = torch.randn(b, latent_dim, device=device)

                style_mixing_prob_epoch = style_mixing_prob
                if style_decay:
                  style_mixing_prob_epoch = sched_stylemix(absolute_epoch)

                fake = generador(z, style_mixing_prob=style_mixing_prob_epoch, update_w_avg=True, truncation=False)
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

                # Monitoreo ligero: norma de gradiente de G
                gnormG = grad_norm(generador)
                gradG_acc += gnormG; gradG_cnt += 1

                optimizerG.step()
                loss_G_total += loss_G.item()

                # EMA update
                if use_ema and (gen_ema is not None):
                    update_ema(gen_ema, generador, decay=ema_decay)

            # métricas batch (loss promediado por pasos de G)
            running_D += loss_D.item()
            running_G += (loss_G_total / max(1, g_steps_this_epoch))
            n_batches += 1

        #  fin de época: logging y muestras 
        epoch_loss_D = running_D / max(1, n_batches)
        epoch_loss_G = running_G / max(1, n_batches)
        lossD_ema = ema_update(lossD_ema, epoch_loss_D)
        lossG_ema = ema_update(lossG_ema, epoch_loss_G)

        d_real_m = dreal_acc / max(1, count_acc)
        d_fake_m = dfake_acc / max(1, count_acc)
        gnormG_m = (gradG_acc / max(1, gradG_cnt))
        gnormD_m = (gradD_acc / max(1, gradD_cnt))

        pnormG = param_norm(generador)
        pnormD = param_norm(discriminador)

        epoch_time = time.time() - t0

        history['loss_D'].append(epoch_loss_D)
        history['loss_G'].append(epoch_loss_G)
        history['D_real'].append(d_real_m)
        history['D_fake'].append(d_fake_m)
        history['grad_norm_G'].append(gnormG_m)
        history['grad_norm_D'].append(gnormD_m)
        history['param_norm_G'].append(pnormG)
        history['param_norm_D'].append(pnormD)
        history['epoch_time_sec'].append(epoch_time)
        history['steps_per_epoch'].append(len(train_loader) * (disc_steps + g_steps_this_epoch))

        if (absolute_epoch % monitor_loss) == 0:
            print(
                f"[Epoch {absolute_epoch:03d}] "
                f"loss_D={epoch_loss_D:.4f} (EMA {lossD_ema:.4f}) | "
                f"loss_G={epoch_loss_G:.4f} (EMA {lossG_ema:.4f}) | "
                f"D(real)={d_real_m:.3f} | D(fake)={d_fake_m:.3f} | "
                f"||∇G||={gnormG_m:.3f} | ||∇D||={gnormD_m:.3f} | "
                f"||θ_G||={pnormG:.1f} | ||θ_D||={pnormD:.1f} | "
                f"batches={n_batches} | time={epoch_time:.1f}s")

        if control_noise:
          with torch.no_grad():
            for m in generador.modules():
              if isinstance(m, NoiseInjection) and hasattr(m, "weight"):
                  m.weight.clamp_(-0.5, 0.5)

            if (use_ema and gen_ema is not None):
              for mn in gen_ema.modules():
                if isinstance(mn, NoiseInjection) and hasattr(m, "weight"):
                   mn.weight.clamp_(-0.5, 0.5)

        if (absolute_epoch % monitor_img) == 0:
            with torch.no_grad():
                use_live_G = (absolute_epoch <= ema_warmup) or (not use_ema) or (gen_ema is None)
                sampler = generador if use_live_G else gen_ema
                fake = sampler(fixed_z).detach()
                fake_vis = _to_vis_range(fake)
            vutils.save_image(fake_vis, f"{save_dir}/epoch_{absolute_epoch:04d}.png", nrow=8)

        if lr_Disc_decay:
          scale = sched_lrd(history['loss_D'])
          if scale < 1.0:
            for pg in optimizerD.param_groups:
                pg['lr'] *= scale
            print(f"[Epoch {absolute_epoch:03d}] ↓ lr_D x{scale:.2f} -> {optimizerD.param_groups[0]['lr']:.2e}")

      
    return history, gen_ema, fixed_z, global_step, (start_epoch + epochs - 1)

