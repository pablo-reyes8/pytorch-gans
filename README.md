# PyTorch GANs

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/pytorch-gans)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/pytorch-gans)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/pytorch-gans)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/pytorch-gans)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/pytorch-gans?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/pytorch-gans?style=social)

A collection of **Generative Adversarial Networks (GANs)** implemented in **PyTorch**.  
This repository brings together different GAN architectures trained on benchmark datasets, providing a practical and extensible framework for learning and experimentation.  

The implementations are designed to be:  
- **Educational** → Clean code that highlights the fundamentals of adversarial training.  
- **Extensible** → Modular design to adapt to new architectures or datasets.  
- **Reproducible** → Includes notebooks and sample outputs for transparency.  

---


## 📂 Repository Structure

```plaintext
pytorch-gans/
│
├── mnist_gan/                # Baseline GAN on MNIST
│   ├── notebooks/            # Jupyter notebooks for training & visualization
│   │   ├── gan_full.ipynb
│   │   └── train_model.ipynb
│   ├── samples/              # Generated digit samples
│   └── src/                  # Source code
│       ├── load_data.py      # MNIST data loading utilities
│       ├── model.py          # Generator & Discriminator definitions
│       └── training.py       # Training loop implementation
│
├── dcgan_cifar/              # Deep Convolutional GAN on CIFAR-10
│   ├── model/                # Saved models (weights, checkpoints)
│   │   └── Generador_30epochs.pth
│   ├── notebooks/            # Training & visualization notebooks
│   │   ├── conv_gan_full.ipynb
│   │   └── train_model.ipynb
│   ├── samples/              # Generated CIFAR-10 images
│   └── src/                  # Source code
│       ├── load_data.py      # CIFAR-10 data loading utilities
│       ├── model.py          # DCGAN Generator & Discriminator
│       └── training.py       # Training loop with refinements
│
├── LICENSE                   # MIT License
├── pyproject.toml            # Project metadata (Poetry / pip installation)
├── poetry.lock               # Dependency lockfile (if using Poetry)
└── README.md                 # Project documentation
```

---

## 🧩 Implementations

### 1. **MNIST GAN** (`mnist_gan/`)
A fully connected GAN trained on the **MNIST dataset** to generate realistic handwritten digits.  

- **Generator**: maps latent vectors (*z* ∈ ℝ^100) to grayscale images of size 32×32.  
- **Discriminator**: distinguishes between real MNIST digits and generated samples.  
- Serves as the **baseline** implementation, ideal for understanding the core mechanics of GANs.  

<p align="center">
  <img src="mnist_gan/samples/epoch_0100.png" alt="MNIST GAN sample" width="280"/>
</p>

---

### 2. **Deep Convolutional GAN (DCGAN)** (`dcgan_cifar/`)
A convolutional GAN based on the **DCGAN architecture** (Radford et al., 2015), trained on the **CIFAR-10 dataset**.  

- **Generator**: convolutional layers with transposed convolutions, enabling the synthesis of 32×32 **color images**.  
- **Discriminator**: convolutional classifier distinguishing real vs. fake images.  
- Incorporates training refinements such as **two-step generator updates** to stabilize learning.  

<p align="center">
  <img src="dcgan_cifar/samples/generated_cifar.png" alt="CIFAR-10 DCGAN sample" width="280"/>
</p>


---

# ⚙️ Installation & Dependencies

## 1. Clone the repository
```bash
git clone https://github.com/pablo-reyes8/pytorch-gans.git
cd pytorch-gans
```

## 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

## 3. Install dependencies

```bash
poetry install
```




---

## 📚 References

- Ian Goodfellow et al. (2014). *Generative Adversarial Nets*. NeurIPS.  
- Alec Radford, Luke Metz, Soumith Chintala (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)*.  
- PyTorch Documentation: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## ✨ Future Work

- Implement **Wasserstein GAN (WGAN)** with gradient penalty.  
- Extend to larger and more diverse datasets (e.g., **CelebA**).  
- Add experiment tracking with **TensorBoard**.  
- Explore **conditional GANs (cGANs)** for class-conditioned image generation.  
