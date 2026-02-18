# PyTorch GANs

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/pytorch-gans)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/pytorch-gans)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/pytorch-gans)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/pytorch-gans)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/pytorch-gans?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/pytorch-gans?style=social)

> **Status:** Research playground for comparing GAN objectives, architectures & stabilisation tricks from scrath.

A collection of **Generative Adversarial Networks (GANs)** implemented in **PyTorch**.  
This repository brings together different GAN architectures trained on benchmark datasets, with a clear goal:

> **Understand what actually changes when we vary:**
> - **data complexity** (MNIST digits vs. RGB natural images vs. faces),
> - **architecture** (MLP GAN → DCGAN → Hinge-SNGAN → StyleGAN),
> - and **loss / stabilisation tricks**  
>   (original saturating GAN loss, hinge loss, non-saturating / softplus, Spectral Norm, R1, EMA, DiffAugment).

The implementations are designed to be:  
- **Educational** → Clean code that highlights the core mechanics of adversarial training.  
- **Extensible** → Modular design so you can swap architectures, losses or datasets.  
- **Reproducible** → Includes scripts / notebooks, checkpoints, and sample outputs.

---

##  What this repo lets you compare

Across the subprojects you can run **direct, controlled comparisons**, such as:

- **Data complexity**
  - 1-channel grayscale digits (**MNIST**)  
    vs 3-channel **RGB** natural images (**CIFAR-like**, pets)  
    vs more structured **faces** (StyleGAN).
- **Architectures**
  - Fully-connected **MNIST GAN**  
  - Convolutional **DCGAN**  
  - Stabilised **Hinge-SNGAN** with Spectral Norm, R1, EMA, DiffAugment  
  - Style-based **StyleGAN**.
- **Objectives & regularisation**
  - Original **(near) saturating GAN loss** (MNIST GAN baseline).
  - **Hinge loss** + Spectral Normalization + R1 + EMA (Hinge-SNGAN).
  - **Non-saturating / softplus-style loss** with R1, EMA, minibatch-stddev (StyleGAN).
- **Stability vs. sample quality**
  - How easy training is on simple grayscale data vs. how fragile it becomes on RGB images.
  - How sample quality changes when you add SN, R1, EMA, DiffAugment and style-based design.

Each folder is a **self-contained lab** for visualising “what happens if I change X but keep everything else similar”.

---

## Repository Structure

```plaintext
pytorch-gans/
│
├── mnist_gan/             # Baseline GAN on MNIST (32x32 grayscale)
│   ├── training/             # Jupyter notebooks for training & visualization
│   ├── samples/              # Generated digit samples
│   └── src/                  # Source code (data, models, training loop)
│
├── dcgan_cifar/            # Deep Convolutional GAN on CIFAR-10 (32x32 RGB)
│   ├── model/                # Saved models (weights, checkpoints)
│   ├── training/             # Training & visualization notebooks
│   ├── samples/              # Generated CIFAR-10 images
│   └── src/                  # Source code (data, DCGAN models, training loop)
│
├── hinge_sngan/            # Hinge-SNGAN with R1, EMA, DiffAugment (CIFRAR-10, 64x64 RGB)
│   ├── samples_first_training/   # Generated samples from first training run
│   ├── samples_second_training/  # Generated samples from second training run
│   └── src/                      # Core implementation (losses, models, train loop)
│   └── training/                 # Training model notebooks
│ 
├── stylegan/            # StyleGanv1 with Hinge loss (CelebA)
│   └── src/                      # Core implementation (losses, models, train loop)
│   ├──  training/                # Training model notebooks
│   ├── samples/                  # Generated CelebA images
│ 
│
├── LICENSE                    # MIT License
├── pyproject.toml             # Project metadata (Poetry / pip installation)
├── poetry.lock                # Dependency lockfile
└── README.md                  # Project documentation

# Every model has a gan_full.ipynb with all the functions and training loops in one place
```

## Implemented Models

### 1. **MNIST GAN** (`mnist_gan/`)

A fully connected GAN trained on the **MNIST dataset** to generate handwritten digits.  
This is the starting point: simple data, simple architecture, and a loss very close to the **original GAN formulation**.

- **Dataset:** MNIST (32×32, grayscale).  
- **Generator:** maps latent vectors (*z* ∈ ℝ¹⁰⁰) to 32×32×1 images.  
- **Discriminator:** MLP that distinguishes real vs. generated digits.  
- **Comparative role:**  
  - Shows that in a “low-complexity” setting (1 channel, centred digits),  
    even a basic GAN with (almost) original loss can learn the distribution.  
  - Serves as a baseline to see how much harder everything becomes once you:
    - move to RGB data,
    - and increase model depth / capacity.

<p align="center">
  <img src="mnist_gan/samples/epoch_0100.png" alt="MNIST GAN sample" width="280"/>
</p>

---

### 2. **Deep Convolutional GAN (DCGAN)** (`dcgan_cifar/`)

A **DCGAN-style** model trained on the **CIFAR-10 dataset**, where we keep roughly the same resolution as MNIST but switch to **3-channel color images**.

- **Dataset:** CIFAR-10 (32×32, RGB).  
- **Generator:** conv–transpose layers synthesising 32×32×3 images.  
- **Discriminator:** convolutional network for real vs. fake classification.  
- **Training details:**  
  - Includes refinements such as **two-step generator updates** to improve stability.  
- **Comparative role:**  
  - Makes very clear how moving from **grayscale digits → color natural images**  
    dramatically increases the difficulty of GAN training.  
  - Typical DCGAN behaviour becomes visible:
    - rough object shapes and colours appear,  
    - but textures, global structure and diversity are much more fragile.  

<p align="center">
  <img src="dcgan_cifar/samples/epoch_0030.png" alt="CIFAR-10 DCGAN sample" width="280"/>
</p>

---

### 3. **Hinge-SNGAN** (`hinge_sngan/`)

A more modern and **heavily stabilised** GAN configuration:  
**Hinge loss + Spectral Normalization + R1 + EMA + DiffAugment** on 64×64 RGB images.

- **Dataset:** 64×64 RGB images (e.g. CIFAR upsampled / pets / similar-scale datasets).  
- **Generator:** deep conv architecture mapping *z* ∈ ℝ¹⁰⁰ to 64×64×3 images.  
- **Discriminator:**  
  - Convolutional network with **Spectral Normalization** (SN),  
  - **Hinge loss** objective.  
- **Training features:**  
  - Multiple discriminator steps per generator step.  
  - Warm-up phase with extra generator updates.  
  - **R1 gradient penalty** every N steps.  
  - **EMA** of generator weights.  
  - **DiffAugment** for data-efficient training.  
- **Comparative role:**  
  - Direct comparison with DCGAN:
    - same rough data regime,  
    - but with modern stabilisation tricks.  
  - Samples become:
    - sharper,  
    - more coherent,  
    - and training is less prone to violent mode collapse.

<p align="center">
  <img src="hinge-sngan/samples_final/Samples Final.png" alt="Hinge-SNGAN sample" width="280"/>
</p>

---

### 4. **StyleGAN** (`stylegan/`)

A **StyleGAN-inspired** implementation with a mapping network, style modulation and stochastic noise for fine-grained control over generated features.

- **Dataset:** face-like datasets (e.g. CelebA-style).  
- **Generator:**  
  - **Mapping network** turning *z* into an intermediate latent *w*,  
  - style-modulated conv blocks using AdaIN,  
  - per-layer **noise injection** for stochastic details,  
  - supports **truncation** and **style mixing**.  
- **Discriminator:**  
  - Convolutional network with **minibatch-stddev** features,  
  - **R1 regularisation** for stable training.  
- **Training:**  
  - Non-saturating / softplus-style loss with R1,  
  - **EMA** of generator,  
  - optional **DiffAugment**.  
- **Comparative role:**  
  - Represents the “high end” of this playground:  
    from simple MLP GAN on MNIST → DCGAN → Hinge-SNGAN → **StyleGAN on faces**.  
  - Makes it visually obvious how far you can push:
    - global structure (face layout, symmetry),  
    - and local detail (hair, eyes, background)  
    when you combine a strong architecture with stabilised training.

<p align="center">
  <img src="stylegan/samplesv3/epoch_0060.png" alt="StyleGAN sample 1" width="280"/>
  <img src="stylegan/samplesv3/Ema Testing 0.37.png" alt="StyleGAN sample 2" width="280"/>
</p>

---

## Companion Project — Diffusion vs. GANs

If you want to compare GANs against diffusion models, this repo pairs naturally with:

> **[`ddpm-diffusion-model`](https://github.com/pablo-reyes8/ddpm-diffusion-model)**

There you’ll find a **DDPM/DDIM** implementation with experiments on attention, sampling schemes and different resolutions (CelebA / CelebA-HQ).

In combination:

- **This repository (`pytorch-gans`)** shows how:
  - losses (saturating vs. non-saturating vs. hinge),  
  - architectural choices (DCGAN vs. SN-GAN vs. StyleGAN),  
  - and stabilisation tricks (SN, R1, EMA, DiffAugment)  
  affect sample quality and training stability.

- **The DDPM repo** shows how diffusion models behave under similar data regimes:
  - much more stable training,  
  - often higher sample quality,  
  - at the cost of slower sampling (partially mitigated with DDIM and better samplers).

---


# Installation & Dependencies

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

## References

 Ian Goodfellow et al. (2014). *Generative Adversarial Nets*. NeurIPS.  
- Alec Radford, Luke Metz, Soumith Chintala (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)*.  
- Takeru Miyato et al. (2018). *Spectral Normalization for Generative Adversarial Networks*. ICLR.  
- Lars Mescheder, Andreas Geiger, Sebastian Nowozin (2018). *Which Training Methods for GANs do actually Converge?* ICML.  
- Tero Karras, Samuli Laine, Timo Aila (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)*. CVPR.  
- Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, Song Han (2020). *Differentiable Augmentation for Data-Efficient GAN Training*. NeurIPS.  
- PyTorch Documentation: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/) 

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Future Work

- Implement **Wasserstein GANs (WGAN, WGAN-GP, WGAN-div)** for improved training stability.  
- Scale up to larger and higher-resolution datasets (e.g., **CelebA-HQ**, **LSUN Bedrooms**).  
- Incorporate backbones of advanced architectures such as **BigGAN** or **StyleGAN2/3**.  
- Explore **conditional GANs (cGANs, AC-GAN, Projection GANs)** for class-conditioned or multi-label generation.  

