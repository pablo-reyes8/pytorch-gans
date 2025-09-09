# Hinge-SNGAN – CIFRAR 10 Clases 

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/pytorch-gans)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/pytorch-gans)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/pytorch-gans)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/pytorch-gans)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/pytorch-gans?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/pytorch-gans?style=social)

An implementation of a **Spectral Normalization GAN (SNGAN)** with **Hinge Loss**, **R1 regularization**, **Exponential Moving Average (EMA)** and **DiffAugment**, trained on the **CIfar10** to generate 64×64 color images.

This project extends the baseline GAN setup with modern stabilization techniques, making it more robust on small datasets.

---

## 📂 Project Structure

```plaintext
hinge-sngan/
├── samples_first_training/     # Generated samples from the first run
│   ├── epoch_0005.png
│   ├── epoch_0020.png
│   └── ...
├── samples_second_training/    # Generated samples from a second run
│   ├── epoch_0010.png
│   ├── epoch_0025.png
│   └── ...
├── src/                        # Core implementation
│   ├── load_data.py            # Cifrar dataloader & preprocessing
│   ├── loss_hinge.py           # Hinge + R1 losses
│   ├── model.py                # Generator & Discriminator (with SN)
│   ├── train_loop.py           # Training loop with R1, EMA, DiffAug
│   └── training_utils.py       # Utility functions (sampling, init, etc.)
└── training/                   # Notebooks for experimentation
    ├── hinge_sngan_first_training.ipynb
    ├── hinge_sngan_second_training.ipynb
    └── hinge_sngan_full.ipynb
```

---

---

## ⚙️ Main Components

- **Generator & Discriminator**

  - Generator: maps latent vectors (_z_ ∈ ℝ^100) into 64×64×3 RGB images.
  - Discriminator: convolutional classifier with **Spectral Normalization** for stable gradients.

- **Loss Functions**  
  Implements **hinge loss** for adversarial training and **R1 gradient penalty** for discriminator regularization.

- **Training Loop**  
  Includes advanced features such as:

  - Multiple discriminator updates per batch.
  - Warm-up phase with extra generator steps.
  - **Exponential Moving Average (EMA)** of generator weights.
  - **Differentiable Augmentation (DiffAugment)** for small datasets.

- **Samples**  
  Generated outputs from multiple training runs are stored to visualize progression over epochs.

- **Notebooks**  
  End-to-end notebooks for training, evaluation, and visualization.

---

## 🚀 Results

After ~50–80 epochs, the generator begins producing recognizable dog/cat-like images at 64×64 resolution.  
EMA and DiffAugment significantly improve stability and sample quality compared to plain hinge-SNGAN.

<p align="center">
  <img src="samples_final/Samples Final.png" alt="Oxford Pets Hinge-SNGAN sample" width="280"/>
</p>

---

## 📚 References

- Miyato et al. (2018). [_Spectral Normalization for Generative Adversarial Networks_](https://arxiv.org/abs/1802.05957).
- Mescheder et al. (2018). [_Which Training Methods for GANs do actually Converge?_](https://arxiv.org/abs/1801.04406).
- Zhao et al. (2020). [_Differentiable Augmentation for Data-Efficient GAN Training_](https://arxiv.org/abs/2006.10738).
- Karras et al. (2019). [_A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)_](https://arxiv.org/abs/1812.04948).

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## ✨ Future Work

- Extend training to larger datasets like **CelebA-HQ**.
- Experiment with **larger architectures (BigGAN-lite)**.
- Evaluate image quality with **FID/KID metrics**.




