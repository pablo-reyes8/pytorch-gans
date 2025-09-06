# DCGAN - CIFAR-10

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/pytorch-gans)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/pytorch-gans)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/pytorch-gans)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/pytorch-gans)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/pytorch-gans?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/pytorch-gans?style=social)

An implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** in PyTorch, trained on the **CIFAR-10 dataset** to generate realistic 32×32 color images.  
This project extends the basic GAN architecture by introducing convolutional and transposed convolutional layers, making it better suited for natural image generation.

---

## 📂 Project Structure

```plaintext
dcgan_cifar/
├── model/                 # Pretrained model checkpoints
│   └── Generador_30epochs.pth
├── notebooks/             # End-to-end notebooks
│   ├── conv_gan_full.ipynb   # Training + visualization
│   └── train_model.ipynb     # Training workflow
├── samples/               # Generated CIFAR-10 samples at different epochs
└── src/                   # Core DCGAN implementation
    ├── load_data.py       # CIFAR-10 data loading & transforms
    ├── model.py           # Generator and Discriminator definitions
    └── training.py        # Training loop with logging & sampling
```

---

## ⚙️ Main Components

- **`model.py`**  
  Implements the **DCGAN Generator** and **Discriminator** using convolutional and transposed convolutional layers.

  - Generator: maps latent vectors (_z_ ∈ ℝ^100) into 32×32×3 RGB images.
  - Discriminator: binary classifier that distinguishes between real CIFAR-10 images and generated samples.

- **`training.py`**  
  Implements the adversarial training loop, including forward/backward passes, optimizer steps, and periodic image sampling. Includes refinements like training the generator multiple times per discriminator update for stability.

- **`load_data.py`**  
  Loads and preprocesses the CIFAR-10 dataset, applying normalization and batching.

- **`conv_gan_full.ipynb`**  
  End-to-end notebook tying together data, models, and training pipeline. Includes visualization of training progress.

- **`samples/`**  
  Stores generated color images at different epochs, showcasing the learning dynamics of the DCGAN.

---

## 🚀 Results

After ~30 epochs of training, the generator produces recognizable CIFAR-10-like images. With more training, hyperparameter tuning, and additional techniques (e.g., label smoothing, spectral normalization), image quality can be further improved.

<p align="center">
  <img src="dcgan_cifar/samples/epoch_0030.png" alt="CIFAR-10 DCGAN sample" width="280"/>
</p>

---

## 📚 References

- Ian Goodfellow et al. (2014). _Generative Adversarial Nets_. NeurIPS.
- Alec Radford, Luke Metz, Soumith Chintala (2015). _Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)_.
- PyTorch Documentation: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## ✨ Future Work

- Train on **larger datasets** like CelebA or LSUN.
- Explore **Wasserstein DCGAN (WGAN-GP)** for improved convergence.
- Add experiment tracking with **TensorBoard**.
- Extend to **conditional DCGANs (cDCGAN)** for class-conditioned generation.
