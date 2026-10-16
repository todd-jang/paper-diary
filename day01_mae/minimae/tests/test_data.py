"""test_data.py — transform contracts (no mean/std normalization, [0,1], correct shape)."""
import numpy as np
import torch
from PIL import Image

import data


def _pil(h=300, w=320, seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def test_build_transform_output_is_tensor_in_unit_range():
    # Given a random PIL image
    torch.manual_seed(0)
    img = _pil()

    # When applying the train transform
    out = data.build_transform()(img)

    # Then it is a (3,224,224) tensor in [0,1] with no mean/std normalization
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_build_eval_transform_output_shape_and_range():
    img = _pil(seed=1)

    out = data.build_eval_transform()(img)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_transforms_handle_small_images():
    # Given an image smaller than the target size
    small = _pil(h=50, w=60, seed=2)

    # Then both transforms still produce 224x224 tensors in [0,1]
    train_out = data.build_transform()(small)
    eval_out = data.build_eval_transform()(small)
    assert train_out.shape == (3, 224, 224)
    assert eval_out.shape == (3, 224, 224)
    assert float(train_out.min()) >= 0.0 and float(train_out.max()) <= 1.0
    assert float(eval_out.min()) >= 0.0 and float(eval_out.max()) <= 1.0


def test_train_transform_is_deterministic_under_seed():
    # Given the same image and the same torch seed
    img = _pil(seed=3)
    tf = data.build_transform()
    torch.manual_seed(5)
    a = tf(img)
    torch.manual_seed(5)
    b = tf(img)
    # Then the augmented outputs are identical (RNG-driven crop/flip is seeded)
    assert torch.allclose(a, b)
