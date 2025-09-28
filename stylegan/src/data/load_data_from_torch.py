import torch
from torchvision import datasets, transforms


def get_celeba_loaders(
    root: str = "./data",
    img_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 1,
    pin_memory: bool = True):


    """
    Crea los DataLoaders de CelebA (train, valid, test) con preprocesamiento estándar.
    
    Parámetros
    ----------
    root : str
        Ruta donde se almacenará/descargará el dataset.
    img_size : int
        Tamaño al que se redimensionarán las imágenes (largo y ancho).
    batch_size : int
        Tamaño del batch para cada DataLoader.
    num_workers : int
        Número de workers para cargar datos en paralelo.
    pin_memory : bool
        Si se fijan los tensores en memoria para mejorar transferencia GPU.

    Retorna
    -------
    train_loader, val_loader, test_loader : torch.utils.data.DataLoader
        DataLoaders de entrenamiento, validación y test.
    """

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(img_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # Datasets
    train_dataset = datasets.CelebA(root=root, split="train", download=True, transform=transform)
    val_dataset   = datasets.CelebA(root=root, split="valid", download=True, transform=transform)
    test_dataset  = datasets.CelebA(root=root, split="test",  download=True, transform=transform)

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader