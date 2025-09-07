import torchvision.utils as vutils
import os
import torch
import matplotlib.pyplot as plt

def train_gan(train_loader, generador, discriminador,
    optimizerG, optimizerD, criterion,
    latent_dim = 100, epochs = 20, sample_every = 1,
    fixed_z = None , smooth = False , smooth_advance = False):

    """
    Entrena una GAN : D con Sigmoid + BCELoss; G con pérdida no-saturante.

    Retorna: diccionario con histórico de pérdidas por epoch: {'loss_D': [...], 'loss_G': [...]}
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generador.to(device)
    discriminador.to(device)

    if smooth == True and smooth_advance == True:
        raise ValueError(f'No se puede tener dos suavizamientos al tiempo')

    # Vector de ruido fijo para ver progreso consistente a través de epochs
    if fixed_z is None:
        os.makedirs("samples", exist_ok=True)
        fixed_z = torch.randn(64, latent_dim, device=device)

        with torch.no_grad():
          fake_imgs_init = generador(fixed_z).detach()

        path = f"samples/epoch_init.png"
        vutils.save_image(fake_imgs_init, path, normalize=True, nrow=8)

        grid = vutils.make_grid(fake_imgs_init, normalize=True, nrow=8)
        grid_cpu = grid.detach().cpu()
        plt.figure(figsize=(6,6))
        plt.axis("off")
        plt.title(f"Fake samples (init)")
        plt.imshow(grid_cpu.permute(1, 2, 0).numpy())
        plt.show()
    else:
        fixed_z = fixed_z.to(device)


    history = {'loss_D': [], 'loss_G': []}

    generador.train()
    discriminador.train()

    for epoch in range(1, epochs + 1):
        running_D, running_G, n_batches = 0.0, 0.0, 0

        for real, _ in train_loader:
            b = real.size(0)
            real = real.to(device, non_blocking=True)
            # Etiquetas: 1 para reales, 0 para falsas


            if smooth:
                real_labels = torch.full((b, 1), 0.9 , device=device)
            elif smooth_advance:
                real_labels = torch.empty(b, 1, device=device).uniform_(0.8, 1.0) # en vez de todos exactamente 0.9, los hacemos variar entre 0.8 y 1.0
            else:
                real_labels = torch.ones(b, 1, device=device)

            fake_labels = torch.zeros(b, 1, device=device)

            # =====================================
            # (1) Update D: max log D(x) + log(1 - D(G(z))) =  O sea verdaderas como verdaderas, falsas como falsas

            # En la practica queremos que su loss sea 0.69 (−log 0.5)
            # Para un G fijo D si logra maximizar (una iteracion), cuando G cambia: D se deberia volver mas ineficiente.
            # Al final el discriminador se vuelve igual que lanzar una moneda 50/50
            # =====================================
            optimizerD.zero_grad(set_to_none=True)

            # Reales: D(x) debe acercarse a 1
            out_real = discriminador(real)           # Probabilidades debido a sigmoid
            loss_D_real = criterion(out_real, real_labels) # −log D(real).

            # Falsas: G(z) -> detach para NO actualizar G en el paso de D
            z = torch.randn(b, latent_dim , device=device)
            fake = generador(z).detach()                      # rompe gradiente hacia G
            out_fake = discriminador(fake)                    # D(G(z)) debe acercarse a 0 para maximizar
            loss_D_fake = criterion(out_fake, fake_labels)    # −log (1 − D(G(z)))

            # Promediamos pérdidas real/fake para estabilidad
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward() # Actualizamos solo D
            optimizerD.step()

            # =====================================
            # (2) Update G:  min log(1 - D(G(z)))  ≡  max log D(G(z)) o sea confundir al discriminador

            # Aca lo importante es notar que ambas cosas no pueden maximizarse a la misma vez son contrarias.
            # =====================================
            optimizerG.zero_grad(set_to_none=True)

            z = torch.randn(b, latent_dim , device=device)
            fake = generador(z)                               # ahora SÍ queremos gradiente hacia G
            out_fake_for_G = discriminador(fake)

            # G quiere que D "piense" que estas falsas son reales (target=1)
            loss_G = criterion(out_fake_for_G, real_labels) # Perdida no saturante

            loss_G.backward() #  retropropaga el gradiente a través de D (porque lo usamos para las probs) pero solo actualiza G
            optimizerG.step()


            # Acumular métricas
            running_D += loss_D.item()
            running_G += loss_G.item()
            n_batches += 1

        epoch_loss_D = running_D / max(n_batches, 1)
        epoch_loss_G = running_G / max(n_batches, 1)
        history['loss_D'].append(epoch_loss_D)
        history['loss_G'].append(epoch_loss_G)


        if (sample_every is not None) and ((epoch + 1) % 5 == 0):
            print(f"[Epoch {epoch:03d}/{epochs}]  loss_D={epoch_loss_D:.4f} | loss_G={epoch_loss_G:.4f}")

        if (sample_every is not None) and ((epoch + 1) % 10 == 0):

            with torch.no_grad():
                fake_imgs = generador(fixed_z).detach()

            path = f"samples/epoch_{epoch+1:04d}.png"
            vutils.save_image(fake_imgs, path, normalize=True, nrow=8)

            grid = vutils.make_grid(fake_imgs, normalize=True, nrow=8)
            grid_cpu = grid.detach().cpu()
            plt.figure(figsize=(6,6))
            plt.axis("off")
            plt.title(f"Fake samples (epoch {epoch+1})")
            plt.imshow(grid_cpu.permute(1, 2, 0).numpy())
            plt.show()


    return history


