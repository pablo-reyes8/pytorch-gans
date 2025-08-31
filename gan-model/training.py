import torchvision.utils as vutils
import os
import torch
import matplotlib.pyplot as plt

def train_gan(train_loader, generador, discriminador,
    optimizerG, optimizerD, criterion,       
    latent_dim = 100,
    epochs = 20,
    sample_every = 1,     
    fixed_z = None , smooth = False , smooth_advance = False) :

    """
    Entrena una GAN 'clásica': D con Sigmoid + BCELoss; G con pérdida no-saturante.

    Retorna: diccionario con histórico de pérdidas por epoch: {'loss_D': [...], 'loss_G': [...]}
    """

    if smooth == True and smooth_advance == True:
        raise ValueError(f'No se puede tener dos suavizamientos al tiempo')
    
    # Vector de ruido fijo para ver progreso consistente a través de epochs
    if fixed_z is None:
        os.makedirs("samples", exist_ok=True)
        fixed_z = torch.randn(64, latent_dim)
        fake_imgs_init = generador(fixed_z).detach()

        path = f"samples/epoch_init.png"
        vutils.save_image(fake_imgs_init, path, normalize=True, nrow=8)

        grid = vutils.make_grid(fake_imgs_init, normalize=True, nrow=8)
        plt.figure(figsize=(6,6))
        plt.axis("off")
        plt.title(f"Fake samples (init)")
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.show()


    history = {'loss_D': [], 'loss_G': []}

    generador.train()
    discriminador.train()

    for epoch in range(1, epochs + 1):
        running_D, running_G, n_batches = 0.0, 0.0, 0

        for real, _ in train_loader:
            b = real.size(0)

            # Etiquetas: 1 para reales, 0 para falsas 

            if smooth: 
                real_labels = torch.full((b, 1), 0.9)
            elif smooth_advance: 
                real_labels = torch.empty(b, 1).uniform_(0.8, 1.0) # en vez de todos exactamente 0.9, los hacemos variar entre 0.8 y 1.0
            else: 
                real_labels = torch.ones(b, 1)

            fake_labels = torch.zeros(b, 1)

            # =====================================
            # (1) Update D: max log D(x) + log(1 - D(G(z)))
            # =====================================
            optimizerD.zero_grad(set_to_none=True)

            # Reales: D(x) debe acercarse a 1
            out_real = discriminador(real)                        # Probabilidades en (0,1) porque D tiene Sigmoid
            loss_D_real = criterion(out_real, real_labels)

            # Falsas: G(z) -> detach para NO actualizar G en el paso de D
            z = torch.randn(b, latent_dim)
            fake = generador(z).detach()                      # rompe gradiente hacia G
            out_fake = discriminador(fake)                        # D(G(z)) debe acercarse a 0
            loss_D_fake = criterion(out_fake, fake_labels)

            # Promediamos pérdidas real/fake para estabilidad
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizerD.step()

            # =====================================
            # (2) Update G:  min log(1 - D(G(z)))  ≡  max log D(G(z))
            #     (pérdida no-saturante): BCE(D(G(z)), 1)
            # =====================================
            optimizerG.zero_grad(set_to_none=True)

            z = torch.randn(b, latent_dim)
            fake = generador(z)                               # ahora SÍ queremos gradiente hacia G
            out_fake_for_G = discriminador(fake)
            # G quiere que D "piense" que estas falsas son reales (target=1)
            loss_G = criterion(out_fake_for_G, real_labels)

            loss_G.backward()
            optimizerG.step()

            # Acumular métricas
            running_D += loss_D.item()
            running_G += loss_G.item()
            n_batches += 1

        epoch_loss_D = running_D / max(n_batches, 1)
        epoch_loss_G = running_G / max(n_batches, 1)
        history['loss_D'].append(epoch_loss_D)
        history['loss_G'].append(epoch_loss_G)

        
        # Muestreo consistente (opcional, para monitorear progreso)
        if (sample_every is not None) and ((epoch + 1) % 2 == 0):

            with torch.no_grad():
                fake_imgs = generador(fixed_z).detach()
            
            path = f"samples/epoch_{epoch+1:04d}.png"
            vutils.save_image(fake_imgs, path, normalize=True, nrow=8)

            grid = vutils.make_grid(fake_imgs, normalize=True, nrow=8)
            plt.figure(figsize=(6,6))
            plt.axis("off")
            plt.title(f"Fake samples (epoch {epoch+1})")
            plt.imshow(grid.permute(1, 2, 0).numpy())
            plt.show()

        print(f"[Epoch {epoch:03d}/{epochs}]  loss_D={epoch_loss_D:.4f} | loss_G={epoch_loss_G:.4f}")

    return history


