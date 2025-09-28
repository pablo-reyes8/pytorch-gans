import os, zipfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_celeba_loader_from_zip(
    zip_path: str,
    extract_dir: str = "./celeba",
    img_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 1,
    pin_memory: bool = True,
    shuffle: bool = True):

    """
    Carga imágenes de un zip local de CelebA (img_align_celeba.zip), extrae si es necesario
    y devuelve un DataLoader listo para entrenar.

    Parámetros
    ----------
    zip_path : str
        Ruta al archivo .zip (por ejemplo en Google Drive).
    extract_dir : str
        Directorio donde se extraen las imágenes.
    img_size : int
        Tamaño al que se redimensionan las imágenes.
    batch_size : int
        Tamaño de batch para DataLoader.
    num_workers : int
        Workers para cargar en paralelo.
    pin_memory : bool
        Si fijar tensores en memoria (mejora GPU transfer).
    shuffle : bool
        Si barajar los datos.

    Retorna
    -------
    DataLoader listo para usar.
    """

    os.makedirs(extract_dir, exist_ok=True)

    def _has_any_image(root: Path) -> bool:
        exts = ("*.jpg", "*.jpeg", "*.png")
        return any(root.rglob(ext) for ext in exts)

    # Extraer solo si no hay imágenes ya
    if not _has_any_image(Path(extract_dir)):
        assert os.path.exists(zip_path), f"No encuentro el zip en {zip_path}"
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

    exts = ("*.jpg", "*.jpeg", "*.png")
    all_paths = []
    for ext in exts:
        all_paths += list(Path(extract_dir).rglob(ext))

    if not all_paths:
        raise FileNotFoundError(
            f"No encontré imágenes (.jpg/.jpeg/.png) tras extraer {zip_path}.\n"
            f"Revisa manualmente el contenido de {extract_dir}.")

    class FlatImageDataset(Dataset):
        def __init__(self, paths, transform=None, skip_broken=True):
            self.paths = sorted(map(str, paths))
            self.transform = transform
            self.skip_broken = skip_broken

            if len(self.paths) == 0:
                raise FileNotFoundError("La lista de imágenes está vacía.")

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                if self.skip_broken:
                    return self.__getitem__((idx + 1) % len(self))
                else:
                    raise
            if self.transform:
                img = self.transform(img)
            return img, 0  

    transform = transforms.Compose([
        transforms.Resize(178, antialias=True),
        transforms.CenterCrop(178),
        transforms.Resize(img_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),])

    dataset = FlatImageDataset(all_paths, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)

    print(f"[OK] Imágenes detectadas (recursivo): {len(dataset)}")
    print(f"[OK] DataLoader listo: batch_size={batch_size}, img_size={img_size}")

    return loader
