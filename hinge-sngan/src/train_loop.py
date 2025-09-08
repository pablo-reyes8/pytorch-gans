from training_utils import * 
from loss_hinge import * 
import torch 
import torchvision.utils as vutils
import os
import copy
import matplotlib.pyplot as plt 


def train_gan(train_loader, generador, discriminador,
    optimizerG, optimizerD, criterion,
    latent_dim = 100, epochs = 20, gamma = 10,
    fixed_z = None , smooth = False , smooth_advance = False ,
    monitor_img = 5 , monitor_loss = 2 ,
    train_gen = 1 , hinge = False,
    disc_steps = 1,                         # pasos de D por batch
    g_warmup_epochs = 0,                    # warm-up de G: épocas con extra-steps
    g_warmup_train_gen = 2,                 #    train_gen durante el warm-up (p.ej., 2 o 3)
    use_ema = True,
    ema_decay = 0.999,
    use_diffaug = True,
    r1_every = 16 ):

    """
    Entrena una GAN : BCE+logits o Hinge (SNGAN) con R1 opcional.
      - disc_steps: D se actualiza varias veces por batch
      - warm-up: más pasos de G en las primeras 'g_warmup_epochs'
      - EMA del generador para muestrear
      - DiffAugment (flip/translation/color) antes de D en reales y falsas

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generador.to(device)
    discriminador.to(device)
    generador.train()
    discriminador.train()

    if smooth and smooth_advance:
        raise ValueError('No se puede tener dos suavizamientos al tiempo')
    if hinge and (smooth or smooth_advance):
        raise ValueError('No se puede usar suavizado y hinge al mismo tiempo')

    # --- EMA (copia del generador para muestrear) ---
    gen_ema = None
    if use_ema:
        gen_ema = copy.deepcopy(generador).to(device)
        gen_ema.eval()
        for p in gen_ema.parameters():
            p.requires_grad_(False)

    # Vector fijo para visualizar progreso
    os.makedirs("samples", exist_ok=True)
    if fixed_z is None:
        fixed_z = torch.randn(64, latent_dim, device=device)

        with torch.no_grad():
            fake0 = (gen_ema if use_ema else generador)(fixed_z).detach()
        vutils.save_image(fake0, "samples/epoch_init.png", normalize=True, nrow=8)
        grid = vutils.make_grid(fake0, normalize=True, nrow=8)
        plt.figure(figsize=(6,6))
        plt.axis("off")
        plt.title("Fake samples (init)")
        plt.imshow(grid.detach().cpu().permute(1,2,0).numpy())
        plt.show()
    else:

        fixed_z = fixed_z.to(device)

    history = {'loss_D': [], 'loss_G': []}
    global_step = 0

    for epoch in range(1, epochs + 1):
        running_D = running_G = 0.0
        n_batches = 0

        # --- 2) warm-up de G: más pasos de G por batch durante X épocas ---
        g_steps_this_epoch = g_warmup_train_gen if epoch <= g_warmup_epochs else train_gen

        for real, _ in train_loader:
            b = real.size(0)
            real = real.to(device, non_blocking=True)

            # ========== (1) Update D  ==========
            for _ in range(disc_steps):
                optimizerD.zero_grad(set_to_none=True)

                if hinge:
                    need_r1 = (global_step % r1_every == 0)
                    if need_r1:
                        real_for_r1 = real.detach().requires_grad_(True)
                        real_in = real_for_r1
                    else:
                        real_for_r1 = real
                        real_in = real
                else:
                    real_in = real  # BCE

                # --- DiffAugment en entradas de D ---
                if use_diffaug:
                    real_in = diff_augment(real_in)

                # forward D en reales
                out_real = discriminador(real_in)

                # falsas para D
                z = torch.randn(b, latent_dim, device=device)
                fake = generador(z).detach()
                fake_in = diff_augment(fake) if use_diffaug else fake
                out_fake = discriminador(fake_in)

                # --- Loss D ---
                if hinge:
                    loss_D_core = loss_hinge_discriminator(out_real, out_fake)
                    if need_r1:
                        r1 = r1_penalty(out_real, real_for_r1)
                        loss_D = loss_D_core + 0.5 * gamma * r1
                    else:
                        loss_D = loss_D_core
                else:
                    # BCE con labels
                    if smooth:
                        real_labels = torch.full((b, 1), 0.9, device=device)
                    elif smooth_advance:
                        real_labels = torch.empty(b, 1, device=device).uniform_(0.8, 1.0)
                    else:
                        real_labels = torch.ones(b, 1, device=device)
                    fake_labels = torch.zeros(b, 1, device=device)
                    loss_D_real = criterion(out_real, real_labels)
                    loss_D_fake = criterion(out_fake, fake_labels)
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimizerD.step()
                global_step += 1

            # ========== (2) Update G (con warm-up de pasos) ==========
            loss_G_total = 0.0
            for _ in range(g_steps_this_epoch):
                optimizerG.zero_grad(set_to_none=True)
                z = torch.randn(b, latent_dim, device=device)
                fake = generador(z)

                # Para el paso de G, también pasamos por D con la misma “vista” (DiffAugment) que vio D
                fake_in_for_G = diff_augment(fake) if use_diffaug else fake
                out_fake_for_G = discriminador(fake_in_for_G)

                if hinge:
                    loss_G = loss_hinge_generator(out_fake_for_G)
                else:
                    # no-saturante con BCE
                    if smooth:
                        real_labels = torch.full((b, 1), 0.9, device=device)
                    elif smooth_advance:
                        real_labels = torch.empty(b, 1, device=device).uniform_(0.8, 1.0)
                    else:
                        real_labels = torch.ones(b, 1, device=device)
                    loss_G = criterion(out_fake_for_G, real_labels)

                loss_G.backward()
                optimizerG.step()
                loss_G_total += loss_G.item()

                # --- 3) actualizar EMA del generador tras cada paso de G ---
                if use_ema:
                    update_ema(gen_ema, generador, decay=ema_decay)

            # acumular métricas
            running_D += loss_D.item()
            running_G += loss_G_total / g_steps_this_epoch
            n_batches += 1

        # epoch metrics
        epoch_loss_D = running_D / max(n_batches, 1)
        epoch_loss_G = running_G / max(n_batches, 1)
        history['loss_D'].append(epoch_loss_D)
        history['loss_G'].append(epoch_loss_G)

        if (epoch % monitor_loss) == 0:
            with torch.no_grad():
                d_real_m = out_real.mean().item()
                d_fake_m = out_fake.mean().item()

            print(f"[Epoch {epoch:03d}/{epochs}] loss_D={epoch_loss_D:.4f} | "
                  f"loss_G={epoch_loss_G:.4f} | D(real)={d_real_m:.2f} | D(fake)={d_fake_m:.2f} | "
                  f"Dsteps={disc_steps} | Gsteps={g_steps_this_epoch}")

        if (epoch % monitor_img) == 0:
            with torch.no_grad():
                sampler = gen_ema if use_ema else generador
                fake_imgs = sampler(fixed_z).detach()

            vutils.save_image(fake_imgs, f"samples/epoch_{epoch:04d}.png", normalize=True, nrow=8)
            grid = vutils.make_grid(fake_imgs, normalize=True, nrow=8)
            plt.figure(figsize=(6,6))
            plt.axis("off")
            plt.title(f"Fake samples (epoch {epoch}) {'[EMA]' if use_ema else ''}")
            plt.imshow(grid.detach().cpu().permute(1,2,0).numpy())
            plt.show()

    return history

