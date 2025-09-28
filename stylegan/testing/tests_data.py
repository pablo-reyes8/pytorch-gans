# tests/test_data_loading.py
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from src.data.load_data_local import *



def _make_dummy_image(path: Path, size=(200, 200), color=(255, 255, 255)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)


def _zip_dir(src_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(src_dir))


def _create_zip_with_images(tmp_path: Path, n=8, broken=False):
    src = tmp_path / "images_src"
    for i in range(n):
        _make_dummy_image(src / f"img_{i:03d}.jpg")
    if broken:
        (src / "corrupto.jpg").write_text("not an image")

    zip_path = tmp_path / "celeba.zip"
    _zip_dir(src, zip_path)
    return zip_path


def _first_batch(loader):
    it = iter(loader)
    return next(it)


def _all_from_loader(loader, max_batches=2):
    xs = []
    for i, (x, y) in enumerate(loader):
        xs.append(x)
        if i + 1 >= max_batches:
            break
    return xs


@pytest.mark.parametrize("img_size", [32, 64, 128])
def test_loader_shapes_and_types(tmp_path, img_size):
    zip_path = _create_zip_with_images(tmp_path, n=10)
    extract_dir = tmp_path / "extract"

    loader = get_celeba_loader_from_zip(
        zip_path=str(zip_path),
        extract_dir=str(extract_dir),
        img_size=img_size,
        batch_size=4,
        num_workers=0,  
        shuffle=True,
        pin_memory=False,)

    x, y = _first_batch(loader)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dim() == 4  # (B,C,H,W)
    assert x.shape[0] == 4
    assert x.shape[1] == 3
    assert x.shape[2] == img_size
    assert x.shape[3] == img_size
    assert x.dtype == torch.float32
    assert y.dtype in (torch.int64, torch.int32, torch.int16, torch.int8)


def test_normalization_in_minus1_plus1(tmp_path):
    zip_path = _create_zip_with_images(tmp_path, n=8)
    extract_dir = tmp_path / "extract"

    loader = get_celeba_loader_from_zip(
        zip_path=str(zip_path),
        extract_dir=str(extract_dir),
        img_size=64,
        batch_size=8,
        num_workers=0,
        shuffle=False,
        pin_memory=False,)

    x, _ = _first_batch(loader)
    assert x.min().item() >= -1.001
    assert x.max().item() <= 1.001
    assert torch.allclose(x.max(), torch.tensor(1.0), atol=1e-3)


def test_skips_broken_images(tmp_path):
    zip_path = _create_zip_with_images(tmp_path, n=7, broken=True)
    extract_dir = tmp_path / "extract"

    loader = get_celeba_loader_from_zip(
        zip_path=str(zip_path),
        extract_dir=str(extract_dir),
        img_size=64,
        batch_size=4,
        num_workers=0,
        shuffle=False,
        pin_memory=False,)

    xs = _all_from_loader(loader, max_batches=2)
    assert len(xs) > 0  


def test_no_reextract_when_images_exist(tmp_path):
    zip_path = _create_zip_with_images(tmp_path, n=6)
    extract_dir = tmp_path / "extract"

    loader1 = get_celeba_loader_from_zip(
        zip_path=str(zip_path),
        extract_dir=str(extract_dir),
        img_size=64,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,)
    x1, _ = _first_batch(loader1)

    os.remove(zip_path)
    loader2 = get_celeba_loader_from_zip(
        zip_path=str(zip_path),  
        extract_dir=str(extract_dir), 
        img_size=64,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,)
    
    x2, _ = _first_batch(loader2)
    assert x1.shape == x2.shape


def test_raises_when_zip_has_no_images(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_path / "no_images.zip"
    _zip_dir(empty_dir, zip_path)  

    extract_dir = tmp_path / "extract"

    with pytest.raises(FileNotFoundError):
        _ = get_celeba_loader_from_zip(
            zip_path=str(zip_path),
            extract_dir=str(extract_dir),
            img_size=64,
            batch_size=2,
            num_workers=0,)
