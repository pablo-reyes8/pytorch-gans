import torchvision.utils as vutils
import os, copy
import torch
from utils.loss_utils import * 
from utils.training_utils import *
import matplotlib.pyplot as plt

def train_gan(train_loader, generador, discriminador,
    optimizerG, optimizerD, criterion,
    latent_dim = 512, epochs = 20, gamma = 10, # Hiperparammetro para R1 pennalty
    fixed_z = None , smooth = False , smooth_advance = False ,
    monitor_img = 5 , monitor_loss = 2 ,
    train_gen = 1 , hinge = False,
    disc_steps = 1,                         # pasos de D por batch
    g_warmup_epochs = 0,                    # warm-up de G: épocas con extra-steps
    g_warmup_train_gen = 2,                 # train_gen durante el warm-up. Si g_warmup_epochs = epochs se entrena dos veces el generador por epoca
    use_ema = True,
    ema_decay = 0.999,
    use_diffaug = True,
    r1_every = 16 , use_softplus = False , style_mixing_prob=0 , ema_warmup =5 ,
    gen_ema=None):

    """
    Entrenamiento de GAN (hinge o BCE) con soporte práctico para:
    - **Actualizaciones asimétricas**: varios pasos de D por batch (disc_steps)
      y warm-up de G con más pasos al inicio (g_warmup_epochs).
    - **Regularización R1**: penalización en reales (||∂D/∂x||^2) aplicada
      esporádicamente para estabilizar D (sin DiffAug en el término R1).
    - **EMA del generador**: una copia de G con promedio exponencial para muestrear,
      lo que produce imágenes más estables que el G “en línea”.
    - **DiffAugment**: data augmentation ligero y consistente (reales/falsas)
      previo a D para robustecer entrenamiento cuando el dataset es pequeño.
    - **Monitoreo**: impresión periódica de pérdidas y grillas de imágenes.

    Intuición del bucle:
    1) D aprende a distinguir reales vs. G(z) (con Hinge o BCE).
    2) G aprende a “engañar” a D (maximiza D(G(z)) en Hinge o minimiza BCE).
    3) R1 sujeta a D a mantener una geometría suave alrededor de datos reales.
    4) EMA promedia pesos de G para muestrear con menor varianza.
    """

    def to_logits(x):
        """Convierte mapas (B,1,H,W) a logits (B,1) con GAP; si ya es (B,1) lo deja igual."""
        return x.mean(dim=(2,3), keepdim=False) if x.dim() == 4 else x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generador.to(device).train()
    discriminador.to(device).train()

    if smooth and smooth_advance:
        raise ValueError('No se puede tener dos suavizamientos al tiempo')
    if hinge and (smooth or smooth_advance):
        raise ValueError('No se puede usar suavizado y hinge al mismo tiempo')
    if use_softplus and (criterion is not None):
        # criterion se ignora en modo softplus (StyleGAN-like)
        pass

    # EMA
    if use_ema:
        if gen_ema is None:
            gen_ema = copy.deepcopy(generador).to(device)
            gen_ema.eval()
            for p in gen_ema.parameters():
                p.requires_grad_(False)
    else:
        gen_ema = None

    # Vector fijo para visualización
    save_dir = make_unique_dir("samples")
    EMA_WARMUP_EPOCHS = ema_warmup

    os.makedirs("samples", exist_ok=True)
    if fixed_z is None:
        fixed_z = torch.randn(64, latent_dim, device=device)
        with torch.no_grad():
            sampler = generador  # <-- NO EMA al inicio
            fake0_logits = sampler(fixed_z).detach()
            fake0_vis = torch.tanh(fake0_logits)
            fake0_vis = (fake0_vis * 0.5 + 0.5).clamp(0, 1)
        vutils.save_image(fake0_vis, f"{save_dir}/epoch_init.png", nrow=8)

        grid = vutils.make_grid(fake0_vis, nrow=8)
        plt.figure(figsize=(6,6)); plt.axis("off")
        plt.title("Fake samples (init) [G]")
        plt.imshow(grid.detach().cpu().permute(1,2,0).numpy())
        plt.show()
    else:
        fixed_z = fixed_z.to(device)

    history = {'loss_D': [], 'loss_G': []}
    global_step = 0

    # LOOP
    for epoch in range(1, epochs + 1):
        running_D = running_G = 0.0
        n_batches = 0
        g_steps_this_epoch = g_warmup_train_gen if epoch <= g_warmup_epochs else train_gen

        for real, _ in train_loader:
            b = real.size(0)
            real = real.to(device, non_blocking=True)

            #  Update D (disc_steps veces)
            for _ in range(disc_steps):
                optimizerD.zero_grad(set_to_none=True)

                # Reales
                real_in = diff_augment(real) if use_diffaug else real
                out_real = discriminador(real_in)
                out_real = to_logits(out_real)  # (B,1)

                # Falsas
                z = torch.randn(b, latent_dim, device=device)
                fake = generador(z, style_mixing_prob=0.0, update_w_avg=True, truncation=False).detach()
                fake_in = diff_augment(fake) if use_diffaug else fake
                out_fake = discriminador(fake_in)
                out_fake = to_logits(out_fake)  # (B,1)

                if hinge:
                    # Hinge
                    loss_D_core = loss_hinge_discriminator(out_real, out_fake)

                    # R1 en reales SIN augment
                    if (global_step % r1_every) == 0:
                        real_r1 = real.detach().requires_grad_(True)
                        out_real_r1 = discriminador(real_r1)
                        out_real_r1 = to_logits(out_real_r1)
                        r1 = r1_penalty(out_real_r1, real_r1)
                        loss_D = loss_D_core + 0.5 * gamma * r1 * r1_every
                    else:
                        loss_D = loss_D_core

                else:
                    if use_softplus:
                        # Logística no-saturante (StyleGAN-like)
                        loss_D_core = F.softplus(-out_real).mean() + F.softplus(out_fake).mean()

                        if (global_step % r1_every) == 0:
                            real_r1 = real.detach().requires_grad_(True)
                            out_real_r1 = discriminador(real_r1)
                            out_real_r1 = to_logits(out_real_r1)
                            r1 = r1_penalty(out_real_r1, real_r1)
                            loss_D = loss_D_core + 0.5 * gamma * r1 * r1_every
                        else:
                            loss_D = loss_D_core

                    else:
                        # BCE con logits (opcional smoothing; no recomendado con R1 moderno)
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
                            out_real_r1 = discriminador(real_r1)
                            out_real_r1 = to_logits(out_real_r1)
                            r1 = r1_penalty(out_real_r1, real_r1)
                            loss_D = loss_D_core + 0.5 * gamma * r1 * r1_every
                        else:
                            loss_D = loss_D_core

                loss_D.backward()
                optimizerD.step()
                global_step += 1

            #  Update G
            loss_G_total = 0.0
            for _ in range(g_steps_this_epoch):
                optimizerG.zero_grad(set_to_none=True)

                z = torch.randn(b, latent_dim, device=device)
                fake = generador(z, style_mixing_prob=0.0, update_w_avg=True, truncation=False)
                fake_in_for_G = diff_augment(fake) if use_diffaug else fake

                out_fake_for_G = discriminador(fake_in_for_G)
                out_fake_for_G = to_logits(out_fake_for_G)

                if hinge:
                    loss_G = loss_hinge_generator(out_fake_for_G)
                else:
                    if use_softplus:
                        # Logística no-saturante (StyleGAN-like)
                        loss_G = F.softplus(-out_fake_for_G).mean()
                    else:
                        # BCE no-saturante: objetivo "reales" para G
                        real_labels = torch.ones(b, 1, device=device, dtype=out_fake_for_G.dtype)
                        loss_G = criterion(out_fake_for_G, real_labels)

                loss_G.backward()
                optimizerG.step()
                loss_G_total += loss_G.item()

                if use_ema and (gen_ema is not None):
                    update_ema(gen_ema, generador, decay=ema_decay)

            # Acumular métricas
            running_D += loss_D.item()
            running_G += loss_G_total / g_steps_this_epoch
            n_batches += 1

        #  Métricas por época
        epoch_loss_D = running_D / max(n_batches, 1)
        epoch_loss_G = running_G / max(n_batches, 1)
        history['loss_D'].append(epoch_loss_D)
        history['loss_G'].append(epoch_loss_G)

        if (epoch % monitor_loss) == 0:
            if hinge:
                with torch.no_grad():
                    d_real_m = out_real.mean().item()
                    d_fake_m = out_fake.mean().item()

                print(f"[Epoch {epoch:03d}/{epochs}] "
                      f"loss_D={epoch_loss_D:.4f} | loss_G={epoch_loss_G:.4f} | "
                      f"D(real)={d_real_m:.2f} | D(fake)={d_fake_m:.2f} | "
                      f"Dsteps={disc_steps} | Gsteps={g_steps_this_epoch}")
            else:
                with torch.no_grad():
                    # calcular pérdidas separadas de BCE para interpretarlas mejor
                    real_labels = torch.ones(out_real.shape, device=device, dtype=out_real.dtype)
                    fake_labels = torch.zeros(out_fake.shape, device=device, dtype=out_fake.dtype)
                    loss_D_real = criterion(out_real, real_labels).item()
                    loss_D_fake = criterion(out_fake, fake_labels).item()

                print(f"[Epoch {epoch:03d}/{epochs}] "
                      f"loss_D={epoch_loss_D:.4f} | loss_G={epoch_loss_G:.4f} | "
                      f"loss_D_real={loss_D_real:.4f} | loss_D_fake={loss_D_fake:.4f} | "
                      f"Dsteps={disc_steps} | Gsteps={g_steps_this_epoch}")


        if (epoch % monitor_img) == 0:
            with torch.no_grad():
                use_live_G = (epoch <= ema_warmup) or (not use_ema)
                sampler = generador if (use_live_G or gen_ema is None) else gen_ema
                fake_logits = sampler(fixed_z).detach()
                fake_vis = torch.tanh(fake_logits)
                fake_vis = (fake_vis*0.5 + 0.5).clamp(0, 1)
            vutils.save_image(fake_vis, f"{save_dir}/epoch_{epoch:04d}.png", nrow=8)

            grid = vutils.make_grid(fake_vis, nrow=8)
            plt.figure(figsize=(6,6))
            plt.axis("off")
            plt.title(f"Fake samples (epoch {epoch}) [{'G' if use_live_G else 'EMA'}]")
            plt.imshow(grid.detach().cpu().permute(1,2,0).numpy())
            plt.show()

    return history, gen_ema